"""Build the sharded teacher / student / fake denoisers of any adapter (LoRA or full-parameter).

Order: meta construction -> LoRA injection -> ``fully_shard`` over the adapter's units -> ``to_empty`` -> rebuild
derived tensors -> ``set_model_state_dict`` broadcast from rank 0 (base weights + initial adapter values, strict).

Frozen parameters (the teacher, the base under LoRA) are stored in the adapter's ``compute_dtype``, trainable ones in
fp32 so small Adam updates survive. The storage dtypes are set on the meta model before sharding, because loading
copies values into the existing parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn

from .distributed import broadcast_object, is_main_process, materialize_and_load, shard_module
from .lora import LoraSpec, initial_lora_state, inject_lora, is_lora_parameter

ROLE_SEED_OFFSETS = {"student": 1, "fake": 2}


@dataclass(frozen=True)
class RoleSpec:
    """How a trainable role is parameterized; ``train='frozen'`` is the teacher."""

    train: str                      # "lora" | "full" | "frozen"
    lora: LoraSpec | None = None
    activation_checkpointing: bool = True
    load_path: str = ""             # optional initial adapter / weights for a new run

    def __post_init__(self) -> None:
        if self.train not in ("lora", "full", "frozen"):
            raise ValueError(f"role train must be lora, full or frozen, got {self.train!r}.")
        if (self.train == "lora") != (self.lora is not None):
            raise ValueError("A LoRA role needs lora settings, and only a LoRA role may have them.")


def build_role(
    adapter: Any,
    role: str,
    spec: RoleSpec,
    *,
    base_state: Mapping[str, torch.Tensor] | None,
    device: torch.device,
    seed: int,
    reshard_after_forward: bool = True,
    read_adapter: Any = None,
) -> nn.Module:
    """One sharded replica. ``base_state`` (CPU tensors in the compute dtype) is needed on rank 0 only."""
    model = adapter.build_on_meta()
    if spec.train == "lora":
        inject_lora(model, spec.lora)
    elif spec.train == "full":
        model.requires_grad_(True)
    else:
        model.requires_grad_(False)
    compute_dtype = getattr(adapter, "compute_dtype", torch.bfloat16)
    set_storage_dtypes(model, frozen=compute_dtype, trainable=torch.float32)
    full_state: dict[str, torch.Tensor] | None = None
    failure: Exception | None = None
    if is_main_process():
        try:
            full_state = _rank0_state(model, role, spec, base_state=base_state, seed=seed, read_adapter=read_adapter)
        except Exception as error:  # noqa: BLE001 - re-raised on every rank below
            failure = error
    # Rank 0 alone validates the weights; without this every other rank would wait forever in the load broadcast.
    message = broadcast_object(None if failure is None else f"{type(failure).__name__}: {failure}")
    if message is not None:
        if failure is not None:
            raise failure
        raise RuntimeError(f"rank 0 could not prepare the {role} weights: {message}")
    if spec.train != "frozen":
        model.train()
        if spec.activation_checkpointing:
            adapter.enable_gradient_checkpointing(model)
    else:
        model.eval()
    shard_module(model, adapter.fsdp_units(model), reshard_after_forward=reshard_after_forward,
                 param_dtype=compute_dtype)
    materialize_and_load(model, full_state, device=device, reset_derived=adapter.reset_derived)
    if spec.train == "frozen" and any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("The teacher must not have trainable parameters.")
    return model


def _rank0_state(model: nn.Module, role: str, spec: RoleSpec, *, base_state: Mapping[str, torch.Tensor] | None,
                 seed: int, read_adapter: Any) -> dict[str, torch.Tensor]:
    """The full state dict rank 0 broadcasts: base weights (fp32 for a full role), plus initial or loaded weights."""
    expected = set(model.state_dict().keys())
    base_keys = {key for key in expected if not is_lora_parameter(key)}
    if base_state is None:
        raise ValueError("Rank 0 needs the base state dict.")
    missing = sorted(base_keys - set(base_state))
    if missing:
        raise RuntimeError(f"Base state lacks {len(missing)} keys, e.g. {missing[:3]}")
    dtype = torch.float32 if spec.train == "full" else None
    full_state = {key: (value.to(dtype) if dtype is not None else value)
                  for key, value in base_state.items() if key in base_keys}
    if spec.train == "lora":
        adapter_state = initial_lora_state(model, spec.lora, seed=int(seed) * 10 + ROLE_SEED_OFFSETS[role])
        if spec.load_path:
            if read_adapter is None:
                raise ValueError("load_path needs a read_adapter callable.")
            loaded, metadata = read_adapter(spec.load_path)
            # rank, alpha and the set of targets fix what the loaded tensors compute; dropout, init and the order of
            # the target globs do not
            lora = metadata.get("lora") if isinstance(metadata.get("lora"), Mapping) else {}
            theirs = (lora.get("rank"), lora.get("alpha"), sorted(set(lora.get("target_modules") or ())))
            ours = (spec.lora.rank, spec.lora.alpha, sorted(set(spec.lora.target_modules)))
            if theirs != ours:
                raise ValueError(f"{role} load_path {spec.load_path} holds a LoRA of (rank, alpha, targets) {theirs}; "
                                 f"the config has {ours}")
            if set(loaded) != set(adapter_state):
                raise ValueError(f"Adapter {spec.load_path} does not match the {role} LoRA targets.")
            adapter_state = loaded
        full_state.update(adapter_state)
    elif spec.train == "full" and spec.load_path:
        if read_adapter is None:
            raise ValueError("load_path needs a read_adapter callable.")
        loaded, metadata = read_adapter(spec.load_path)
        if metadata.get("train") != "full":
            raise ValueError(f"{role} load_path {spec.load_path} holds {metadata.get('train')!r} weights, not full.")
        # every parameter must be there; buffers may be (an edmd export has none: they come from the base)
        parameters = {name for name, _ in model.named_parameters()}
        if not parameters <= set(loaded) <= base_keys:
            missing, extra = sorted(parameters - set(loaded)), sorted(set(loaded) - base_keys)
            raise ValueError(f"{spec.load_path} does not match the {role} parameters: "
                             f"missing {missing[:3]}, unexpected {extra[:3]}")
        full_state.update({key: value.float() for key, value in loaded.items() if key in parameters})
    return full_state


def set_storage_dtypes(model: nn.Module, *, frozen: torch.dtype, trainable: torch.dtype) -> None:
    """Re-create every parameter of a (meta) model in ``trainable`` or ``frozen`` dtype by its ``requires_grad``."""
    for module in model.modules():
        for name, parameter in list(module._parameters.items()):
            if parameter is None:
                continue
            dtype = trainable if parameter.requires_grad else frozen
            if parameter.dtype != dtype:
                module._parameters[name] = nn.Parameter(parameter.detach().to(dtype),
                                                        requires_grad=parameter.requires_grad)


__all__ = ["ROLE_SEED_OFFSETS", "RoleSpec", "build_role", "set_storage_dtypes"]

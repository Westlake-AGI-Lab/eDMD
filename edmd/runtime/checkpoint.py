"""Checkpoints: trainable parameters (LoRA or full), optimizers, EMA, RNG and sampler position.

Layout of ``<output_dir>/checkpoint-N/``::

    dcp/                          sharded DCP state: {student,fake}_trainable, {student,fake}_optim,
                                  student_ema_trainable (if enabled)
    weights/<role>.safetensors    rank-0 export: LoRA adapter tensors (fp32), or the full weights (bf16)
    weights/<role>.json           the role's train type, LoRA spec and step
    weights/inference.json        how to sample the renderable roles (sampler, grid, resolution, adapter options)
    runtime_state_rank-R.pt       per-rank Python/NumPy/torch/device (NPU or GPU) RNG and sampler state
    training_state.pt             step, LR schedulers, EMA counters
    training_step.txt             written last by rank 0: the completion marker

The frozen bases are never written: every role is rebuilt from the model path and its trainable state.
"""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.distributed as dist
from torch import nn

from . import device as device_lib
from .distributed import barrier, full_tensor, is_main_process

COMPLETE_MARKER = "training_step.txt"
TRAINING_STATE = "training_state.pt"
WEIGHTS_FORMAT = "edmd_weights_v1"
#: training_state key: per role, whether its optimizer had taken a step when the checkpoint was written
OPTIMIZERS_STEPPED = "optimizers_stepped"


def checkpoint_step(checkpoint: Path) -> int:
    """The step of a *complete* checkpoint: its completion marker, which must name the directory's step."""
    checkpoint = Path(checkpoint)
    marker = checkpoint / COMPLETE_MARKER
    if not marker.is_file():
        raise ValueError(f"{checkpoint} is not a complete checkpoint (no {COMPLETE_MARKER})")
    step = int(marker.read_text().strip())
    if checkpoint.name != f"checkpoint-{step}":
        raise ValueError(f"{checkpoint} holds step {step}")
    return step


def complete_checkpoints(output_dir: Path) -> dict[int, Path]:
    """``{step: path}`` of the complete checkpoints of a run directory, in step order."""
    found = {}
    for path in Path(output_dir).glob("checkpoint-*"):
        try:
            found[checkpoint_step(path)] = path
        except (ValueError, OSError):
            continue
    return dict(sorted(found.items()))


def optimizers_stepped(roles: Mapping[str, tuple[nn.Module, torch.optim.Optimizer | None]]) -> dict[str, bool]:
    """``{role: optimizer has state}``: an optimizer that never stepped has an empty ``state``."""
    return {role: bool(optimizer.state) for role, (_, optimizer) in roles.items() if optimizer is not None}


def trainable_sharded_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """Sharded (DTensor) trainable parameters keyed by their model FQNs."""
    from torch.distributed.checkpoint.state_dict import get_model_state_dict

    names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    state = {key: value for key, value in get_model_state_dict(model).items() if key in names}
    if not state:
        raise ValueError("Model has no trainable parameters to checkpoint.")
    return state


def _dcp_state(roles: Mapping[str, tuple[nn.Module, torch.optim.Optimizer | None]], ema: Any | None) -> dict[str, Any]:
    from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict

    state: dict[str, Any] = {}
    for role, (model, optimizer) in roles.items():
        state[f"{role}_trainable"] = trainable_sharded_state(model)
        if optimizer is not None:
            state[f"{role}_optim"] = get_optimizer_state_dict(model, optimizer)
    if ema is not None:
        state["student_ema_trainable"] = ema.shadow
    return state


def save_runtime_state(checkpoint: Path, *, rank: int, world_size: int, device: torch.device,
                       sampler: Any, step: int, batches_consumed: int) -> None:
    torch.save({
        "step": int(step),
        "batches_consumed": int(batches_consumed),
        "world_size": int(world_size),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "npu_rng": None,                   # overwritten below on an NPU
        **device_lib.rng_state(device),    # npu_rng / cuda_rng: the device generator
        "device_type": device.type,
        "sampler": sampler.state_dict() if sampler is not None else None,
    }, checkpoint / f"runtime_state_rank-{rank}.pt")


def restore_runtime_state(checkpoint: Path, *, rank: int, world_size: int, device: torch.device,
                          sampler: Any, step: int) -> tuple[int, bool]:
    """Restore before creating the loader iterator; returns ``(consumed prompt batches, exact)``.

    The same world size continues exactly: every random stream and the data position. Another world size continues
    the data from the same global position and leaves the random streams to the caller (``exact`` False); another
    device type keeps the host streams and the device generator as seeded."""
    saved = sorted(checkpoint.glob("runtime_state_rank-*.pt"))
    if not saved:
        raise FileNotFoundError(f"Missing continuation state in {checkpoint}")
    path = checkpoint / f"runtime_state_rank-{rank}.pt"
    state = torch.load(path if path.is_file() else saved[0], map_location="cpu", weights_only=False)
    if int(state["step"]) != int(step):
        raise ValueError("Checkpoint continuation step does not match this run.")
    exact = int(state["world_size"]) == int(world_size) and path.is_file()
    if sampler is not None:
        if state["sampler"] is None:
            raise ValueError("Checkpoint has no sampler state.")
        sampler.load_state_dict(state["sampler"], strict=exact)
    consumed = int(state["batches_consumed"])
    if not exact:   # the same number of prompts, in batches of this world size
        return consumed * int(state["world_size"]) // int(world_size), False
    if state.get("device_type") == device.type:
        device_lib.restore_rng_state(state, device)
    else:
        print(f"warning: {checkpoint} was saved on {state.get('device_type')}; the {device.type} generator is not "
              "restored", flush=True)
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"])
    return consumed, True


def gather_trainable(model: nn.Module | None = None, *, tensors: Mapping[str, torch.Tensor] | None = None,
                     dtype: torch.dtype = torch.float32) -> dict[str, torch.Tensor]:
    """Full trainable tensors on CPU for rank 0 (collective: call on every rank)."""
    if (model is None) == (tensors is None):
        raise ValueError("Pass exactly one of model or tensors.")
    items = ([(name, parameter.detach()) for name, parameter in model.named_parameters() if parameter.requires_grad]
             if model is not None else list(tensors.items()))
    gathered: dict[str, torch.Tensor] = {}
    for name, value in items:
        full = full_tensor(value)
        if is_main_process():
            gathered[name] = full.to(dtype).cpu()
    return gathered


def write_weights(path: Path, tensors: Mapping[str, torch.Tensor], metadata: Mapping[str, Any]) -> None:
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({key: value.contiguous() for key, value in tensors.items()}, str(path), metadata={"format": WEIGHTS_FORMAT})
    path.with_suffix(".json").write_text(json.dumps(dict(metadata), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_weights(path: str | Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """``(tensors, role record)`` of a role's weights: ``<role>.safetensors`` with its ``<role>.json``, or a
    single-file release (a diffusers LoRA from ``scripts/convert_lora.py``), returned as the role file it was made
    from: ``transformer.`` prefix dropped, ``alpha / rank`` divided back out of ``lora_B`` (exact for the paper's
    alpha / rank = 2)."""
    from safetensors import safe_open

    path = Path(path)
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        if "edmd" in metadata and metadata.get("format") != WEIGHTS_FORMAT:
            record = json.loads(metadata["edmd"])["weights"]
            scale = float(record["lora"]["alpha"]) / int(record["lora"]["rank"])
            for key in handle.keys():
                name = key.removeprefix("transformer.")
                value = handle.get_tensor(key)
                tensors[name] = value / scale if name.endswith(".lora_B.weight") else value
            return tensors, record
        if metadata.get("format") != WEIGHTS_FORMAT:
            raise ValueError(f"{path} is not an edmd weights file.")
        for key in handle.keys():
            tensors[key] = handle.get_tensor(key)
    return tensors, json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))


def save_checkpoint(
    output_dir: Path, step: int, *,
    roles: Mapping[str, tuple[nn.Module, torch.optim.Optimizer | None]],
    role_train: Mapping[str, str],
    ema: Any | None,
    weights_metadata: Mapping[str, Mapping[str, Any]],
    training_state: Mapping[str, Any],
    rank: int, world_size: int, device: torch.device, sampler: Any, batches_consumed: int,
    inference_record: Mapping[str, Any] | None = None,
) -> Path:
    """Write ``checkpoint-<step>`` (layout: module docstring), the completion marker last."""
    import torch.distributed.checkpoint as dcp

    checkpoint = output_dir / f"checkpoint-{int(step)}"
    if is_main_process():
        if checkpoint.exists():
            shutil.rmtree(checkpoint)  # a retry at the same step must not mix old and new shards
        checkpoint.mkdir(parents=True)
    barrier()
    # Before _dcp_state: get_optimizer_state_dict initializes a never-stepped optimizer with a zero-gradient, lr-0 step
    # (Adam's step counter at 1); load_checkpoint drops such a state, so a run from checkpoint-0 does not carry that
    # phantom step's bias correction.
    stepped = optimizers_stepped(roles)
    dcp.save(_dcp_state(roles, ema), checkpoint_id=str(checkpoint / "dcp"))
    save_runtime_state(checkpoint, rank=rank, world_size=world_size, device=device, sampler=sampler,
                       step=step, batches_consumed=batches_consumed)
    exports: dict[str, dict[str, torch.Tensor]] = {}
    for role, (model, _) in roles.items():
        dtype = torch.bfloat16 if role_train[role] == "full" else torch.float32
        exports[role] = gather_trainable(model, dtype=dtype)
    if ema is not None:
        dtype = torch.bfloat16 if role_train["student"] == "full" else torch.float32
        exports["student_ema"] = gather_trainable(tensors=ema.export_tensors(), dtype=dtype)
    if is_main_process():
        for role, tensors in exports.items():
            metadata = dict(weights_metadata["student" if role == "student_ema" else role])
            write_weights(checkpoint / "weights" / f"{role}.safetensors", tensors,
                          {**metadata, "step": int(step), "role": role})
        torch.save({**dict(training_state), OPTIMIZERS_STEPPED: stepped}, checkpoint / TRAINING_STATE)
        if inference_record is not None:   # before the marker: a complete checkpoint always has its record
            (checkpoint / "weights" / "inference.json").write_text(json.dumps(dict(inference_record), indent=2) + "\n",
                                                                 encoding="utf-8")
    barrier()
    if is_main_process():
        (checkpoint / COMPLETE_MARKER).write_text(f"{int(step)}\n", encoding="utf-8")
    barrier()
    return checkpoint


def load_checkpoint(checkpoint: Path, *, roles: Mapping[str, tuple[nn.Module, torch.optim.Optimizer | None]],
                    ema: Any | None) -> dict[str, Any]:
    """Load trainable parameters, optimizer states and the EMA shadow in place; return ``training_state``."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict, set_optimizer_state_dict

    checkpoint = Path(checkpoint)
    if not (checkpoint / COMPLETE_MARKER).is_file():
        raise FileNotFoundError(f"{checkpoint} is not a complete checkpoint (no {COMPLETE_MARKER}).")
    training_state = torch.load(checkpoint / TRAINING_STATE, map_location="cpu", weights_only=False)
    if ema is not None:
        saved_ema = training_state.get("ema")
        if saved_ema is not None:
            ema.validate_state_dict(saved_ema)  # reject a schedule mismatch before mutating models / optimizers
    state = _dcp_state(roles, ema)
    dcp.load(state, checkpoint_id=str(checkpoint / "dcp"))
    for role, (model, optimizer) in roles.items():
        trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
        result = set_model_state_dict(model, state[f"{role}_trainable"], options=StateDictOptions(strict=False))
        unexpected = list(getattr(result, "unexpected_keys", []) or [])
        missing = [key for key in (getattr(result, "missing_keys", []) or []) if key in trainable]
        if unexpected or missing:
            raise RuntimeError(f"{role} trainable-state mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
        if optimizer is not None:
            set_optimizer_state_dict(model, optimizer, optim_state_dict=state[f"{role}_optim"])
    stepped = training_state.get(OPTIMIZERS_STEPPED) or {}
    for role, (_, optimizer) in roles.items():
        if optimizer is not None and stepped.get(role) is False:
            optimizer.state.clear()        # the saved state is torch's initialization step, not a real one
    if dist.is_initialized():
        barrier()
    return training_state


__all__ = ["COMPLETE_MARKER", "OPTIMIZERS_STEPPED", "TRAINING_STATE", "WEIGHTS_FORMAT", "checkpoint_step",
           "complete_checkpoints", "gather_trainable", "load_checkpoint", "optimizers_stepped", "read_weights",
           "restore_runtime_state", "save_checkpoint", "save_runtime_state", "trainable_sharded_state",
           "write_weights"]

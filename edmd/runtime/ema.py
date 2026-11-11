"""Exponential moving average of the student's trainable parameters (LoRA adapters or the full model).

Schedule (CDM's ``EMAModuleWrapper``): after every ``update_every`` successful optimizer steps
``ema = d * ema + (1 - d) * param`` with the warm-up ``d = min((1 + n) / (10 + n), decay)`` (so 0.999 is only ~0.9955
at step 2000).

The shadow is one detached clone per trainable parameter (a DTensor under FSDP2, so the update is local). Training
never reads it; checkpoints store it for export and evaluation.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn

from .lora import LoRALinear, is_lora_parameter


@torch.no_grad()
def effective_weight(module: LoRALinear) -> torch.Tensor:
    """FP32 ``W0 + s B A`` with the base's sharding, without mutating any live weight.

    A is replicated and B redistributed like W0 before the local rows are multiplied; only replicated or row-sharded
    W0 is supported (other layouts are refused rather than gathering W0).
    """
    from torch.distributed.tensor import DTensor, Replicate, Shard

    weight = module.weight.detach()
    a, b = module.lora_A.weight.detach(), module.lora_B.weight.detach()
    result = weight.to(dtype=torch.float32, copy=True)
    if isinstance(weight, DTensor):
        if not isinstance(a, DTensor) or not isinstance(b, DTensor):
            raise ValueError("the effective weight requires W0, A and B on the same device mesh")
        if a.device_mesh != weight.device_mesh or b.device_mesh != weight.device_mesh:
            raise ValueError("the effective weight requires W0, A and B on the same device mesh")
        if any(not (isinstance(p, Replicate) or isinstance(p, Shard) and p.dim == 0) for p in weight.placements):
            raise ValueError("the effective weight supports replicated or row-sharded linear weights")
        a = a.redistribute(placements=[Replicate()] * weight.device_mesh.ndim).to_local()
        b = b.redistribute(placements=weight.placements).to_local()
        local = result.to_local()
    else:
        if isinstance(a, DTensor) or isinstance(b, DTensor):
            raise ValueError("the effective weight requires W0, A and B on the same device mesh")
        local = result
    local.add_(b.float() @ a.float(), alpha=module.lora_scaling)
    return result


def warmup_decay(updates: int, decay: float) -> float:
    if updates < 0:
        raise ValueError("updates must be non-negative")
    return min((1.0 + updates) / (10.0 + updates), decay)


@dataclass(frozen=True)
class EmaSpec:
    decay: float = 0.999
    update_every: int = 1
    warmup: bool = True
    # offset 1 reproduces CDM, whose warm-up starts at d(1) = 2/11 (it counts the step before the update)
    offset: int = 0

    def decay_at(self, updates: int) -> float:
        return warmup_decay(updates + self.offset, self.decay) if self.warmup else self.decay


class ShadowEma:
    def __init__(self, model: nn.Module, spec: EmaSpec) -> None:
        self.spec = spec
        self.model = model
        self.pairs: list[tuple[str, nn.Parameter]] = [
            (name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
        if not self.pairs:
            raise ValueError("No trainable parameters to average")
        self.shadow: dict[str, torch.Tensor] = {name: parameter.detach().clone() for name, parameter in self.pairs}
        self.updates = 0
        self.pending_steps = 0

    @torch.no_grad()
    def copy_from_model(self) -> None:
        for name, parameter in self.pairs:
            self.shadow[name].copy_(parameter.detach())
        self.updates = 0
        self.pending_steps = 0

    def current_decay(self) -> float:
        return self.spec.decay_at(self.updates)

    @torch.no_grad()
    def after_optimizer_step(self, *, skipped: bool = False) -> bool:
        if skipped:
            return False
        self.pending_steps += 1
        if self.pending_steps < self.spec.update_every:
            return False
        decay = self.current_decay()
        for name, parameter in self.pairs:
            self.shadow[name].mul_(decay).add_(parameter.detach(), alpha=1.0 - decay)
        self.updates += 1
        self.pending_steps = 0
        return True

    @contextlib.contextmanager
    def swapped_into_model(self):
        """Temporarily load the shadow into the model's trainable parameters."""
        with torch.no_grad():
            live = {name: parameter.detach().clone() for name, parameter in self.pairs}
            for name, parameter in self.pairs:
                parameter.detach().copy_(self.shadow[name])
        try:
            yield
        finally:
            with torch.no_grad():
                for name, parameter in self.pairs:
                    parameter.detach().copy_(live[name])

    def export_tensors(self) -> dict[str, torch.Tensor]:
        """The shadow tensors (sharded like the trainable parameters) for the ``student_ema`` weights export."""
        return self.shadow

    def state_dict(self) -> dict[str, Any]:
        return {"updates": int(self.updates), "pending_steps": int(self.pending_steps),
                "spec": {"decay": self.spec.decay, "update_every": self.spec.update_every, "warmup": self.spec.warmup,
                         "offset": self.spec.offset}}

    def validate_state_dict(self, state: Mapping[str, Any]) -> None:
        saved = EmaSpec(**state["spec"])
        if saved != self.spec:
            raise ValueError(f"Checkpoint EMA schedule {saved} differs from config {self.spec}.")

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.validate_state_dict(state)
        self.updates = int(state["updates"])
        self.pending_steps = int(state.get("pending_steps", 0))


@torch.no_grad()
def merge_parameters(target: nn.Module, source: nn.Module, decay: float) -> int:
    """CDM's fake <- student merge: ``target = decay * target + (1 - decay) * source`` over matching trainable
    parameters.

    For LoRA roles this interpolates A and B separately, not the low-rank increments (a deviation from CDM's
    full-parameter roles). A full-parameter fake with a LoRA student merges each adapted linear's weight toward the
    student's effective ``W0 + scaling * B @ A`` and every other parameter toward its base value (the student's own).
    """
    source_params = dict(source.named_parameters())
    source_modules = dict(source.named_modules())
    count = 0
    for name, parameter in target.named_parameters():
        if not parameter.requires_grad:
            continue
        other = source_params.get(name)
        if other is None or tuple(other.shape) != tuple(parameter.shape):
            raise ValueError(f"merge_parameters: {name} has no counterpart of the same shape in the source.")
        value = other.data
        if not other.requires_grad:
            if is_lora_parameter(name):
                raise ValueError(f"merge_parameters: the source's adapter {name} is frozen")
            owner, _, leaf = name.rpartition(".")
            module = source_modules.get(owner)
            if isinstance(module, LoRALinear) and leaf == "weight":
                # the frozen base alone would pull the fake to the teacher and ignore the student's adapter
                value = effective_weight(module)
        parameter.data.mul_(decay).add_(value.to(parameter.dtype), alpha=1.0 - decay)
        count += 1
    if count == 0:
        raise ValueError("merge_parameters found no trainable parameters.")
    return count


__all__ = ["EmaSpec", "ShadowEma", "effective_weight", "merge_parameters", "warmup_decay"]

"""Minimal in-place LoRA, FSDP2/DCP friendly: ``y = W x + b + (alpha / r) * B(A(dropout(x)))``.

* A target ``nn.Linear`` keeps its parameter names and gains ``lora_A`` / ``lora_B`` children, so the frozen base
  loads without key mapping.
* ``lora_B`` starts at zero, so student and fake begin as exact copies of the teacher.
* ``gaussian`` draws ``lora_A`` with std ``1/r`` (PEFT's ``init_lora_weights="gaussian"``); ``kaiming`` is PEFT's
  default.
* Every target glob must match at least one ``nn.Linear`` (a silent no-match would train nothing).
"""

from __future__ import annotations

import fnmatch
import math
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn

INIT_MODES = ("gaussian", "kaiming")
LORA_PARAMETER_MARKERS = (".lora_A.", ".lora_B.")


@dataclass(frozen=True)
class LoraSpec:
    rank: int
    alpha: float
    dropout: float
    init: str
    target_modules: tuple[str, ...]

    def __post_init__(self) -> None:
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank <= 0:
            raise ValueError("lora.rank must be a positive integer.")
        if not (math.isfinite(self.alpha) and self.alpha > 0):
            raise ValueError("lora.alpha must be positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("lora.dropout must lie in [0, 1).")
        if self.init not in INIT_MODES:
            raise ValueError(f"lora.init must be one of {INIT_MODES}.")
        if not self.target_modules:
            raise ValueError("lora.target_modules must be non-empty.")

    @property
    def scaling(self) -> float:
        return self.alpha / self.rank

    def identity(self) -> dict:
        return {"rank": self.rank, "alpha": self.alpha, "dropout": self.dropout, "init": self.init,
                "target_modules": list(self.target_modules)}


class LoRALinear(nn.Linear):
    """``nn.Linear`` with a low-rank residual; only produced by ``inject_lora``."""

    lora_A: nn.Linear
    lora_B: nn.Linear
    lora_dropout: nn.Module
    lora_scaling: float

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002 - nn.Linear signature
        result = F.linear(input, self.weight, self.bias)
        update = self.lora_B(self.lora_A(self.lora_dropout(input)))
        return result + update * self.lora_scaling


class PeftOrderLoRALinear(LoRALinear):
    """``LoRALinear`` with PEFT's cast order for an adapter in another dtype than the base (``peft_fp32`` of FLUX.2 and
    SD3): base matmul, input cast to the adapter's dtype, sum promoted, one rounding back."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002 - nn.Linear signature
        result = F.linear(input, self.weight, self.bias)
        result_dtype = result.dtype
        adapter_dtype = self.lora_A.weight.dtype
        x = input if input.dtype == adapter_dtype else input.to(adapter_dtype)
        result = result + self.lora_B(self.lora_A(self.lora_dropout(x))) * self.lora_scaling
        return result.to(result_dtype)


def use_peft_order(model: nn.Module) -> int:
    """Switch every injected ``LoRALinear`` to :class:`PeftOrderLoRALinear` in place; returns how many."""
    count = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.__class__ = PeftOrderLoRALinear
            count += 1
    return count


def _matches(name: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)


def lora_target_names(model: nn.Module, spec: LoraSpec) -> list[str]:
    names = [
        name for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and not isinstance(module, LoRALinear) and _matches(name, spec.target_modules)
    ]
    for pattern in spec.target_modules:
        if not any(fnmatch.fnmatchcase(name, pattern) for name in names):
            raise ValueError(f"LoRA target pattern {pattern!r} matches no nn.Linear in the model.")
    return names


def inject_lora(model: nn.Module, spec: LoraSpec, *, dtype: torch.dtype = torch.float32) -> list[str]:
    """Convert the targeted linears in place and freeze everything but the adapters (meta-device safe)."""
    names = lora_target_names(model, spec)
    model.requires_grad_(False)
    for name in names:
        module = model.get_submodule(name)
        device = module.weight.device
        module.__class__ = LoRALinear
        module.lora_A = nn.Linear(module.in_features, spec.rank, bias=False, device=device, dtype=dtype)
        module.lora_B = nn.Linear(spec.rank, module.out_features, bias=False, device=device, dtype=dtype)
        module.lora_dropout = nn.Dropout(spec.dropout) if spec.dropout > 0 else nn.Identity()
        module.lora_scaling = spec.scaling
        module.lora_A.weight.requires_grad_(True)
        module.lora_B.weight.requires_grad_(True)
    return names


def is_lora_parameter(name: str) -> bool:
    return any(marker in f".{name}" for marker in LORA_PARAMETER_MARKERS)


def lora_parameter_names(model: nn.Module) -> list[str]:
    return [name for name, _ in model.named_parameters() if is_lora_parameter(name)]


def initial_lora_state(model: nn.Module, spec: LoraSpec, *, seed: int,
                       dtype: torch.dtype = torch.float32) -> dict[str, torch.Tensor]:
    """Deterministic CPU initial values for every adapter, identical on every rank."""
    generator = torch.Generator().manual_seed(int(seed))
    state: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if not isinstance(module, LoRALinear):
            continue
        a = torch.empty(spec.rank, module.in_features, dtype=torch.float32)
        if spec.init == "gaussian":
            a.normal_(mean=0.0, std=1.0 / spec.rank, generator=generator)
        else:
            bound = 1.0 / math.sqrt(module.in_features)  # kaiming_uniform_(a=sqrt(5)) bound for fan_in
            a.uniform_(-bound, bound, generator=generator)
        state[f"{name}.lora_A.weight"] = a.to(dtype)
        state[f"{name}.lora_B.weight"] = torch.zeros(module.out_features, spec.rank, dtype=dtype)
    if not state:
        raise ValueError("The model has no LoRA modules to initialize.")
    return state


@torch.no_grad()
def merge_lora_into_base(model: nn.Module) -> int:
    """Fold every adapter into its base weight and restore plain ``nn.Linear`` (inference / export only)."""
    merged = 0
    for _, module in list(model.named_modules()):
        if not isinstance(module, LoRALinear):
            continue
        delta = module.lora_B.weight.float() @ module.lora_A.weight.float()
        module.weight.add_((delta * module.lora_scaling).to(module.weight.dtype))
        del module.lora_A, module.lora_B, module.lora_dropout, module.lora_scaling
        module.__class__ = nn.Linear
        merged += 1
    return merged


__all__ = [
    "INIT_MODES", "LoRALinear", "LoraSpec", "PeftOrderLoRALinear", "initial_lora_state", "inject_lora",
    "is_lora_parameter", "lora_parameter_names", "lora_target_names", "merge_lora_into_base", "use_peft_order",
]

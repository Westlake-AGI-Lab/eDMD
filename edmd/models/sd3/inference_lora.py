"""LoRA on the single-device SD3 inference transformer (``options.inference_lora``): ``unmerged_compute`` (default),
``peft_fp32`` (fp32 adapter in peft's cast order, :class:`PeftLinear`) or ``merged`` (fp32 merge, one rounding); see
``ModelAdapter.load_inference_denoiser``. Under the bf16 evaluation autocast ``unmerged_compute`` and ``peft_fp32``
compute the same.
"""

from __future__ import annotations

from typing import Mapping

import torch
from torch import nn

from ...runtime.lora import LoRALinear, LoraSpec, PeftOrderLoRALinear, inject_lora, lora_parameter_names

MODES = ("peft_fp32", "unmerged_compute", "merged")
PeftLinear = PeftOrderLoRALinear   # the runtime's peft-order forward (shared with FLUX.2)


def spec_from_metadata(identity: Mapping) -> LoraSpec:
    return LoraSpec(rank=int(identity["rank"]), alpha=float(identity["alpha"]),
                    dropout=float(identity.get("dropout", 0.0)), init=str(identity.get("init", "gaussian")),
                    target_modules=tuple(identity["target_modules"]))


def attach_unmerged(model: nn.Module, spec: LoraSpec, tensors: Mapping[str, torch.Tensor], *, mode: str,
                    compute_dtype: torch.dtype) -> list[str]:
    """Inject the adapter into a materialized ``compute_dtype`` model and load ``tensors`` (strict key set)."""
    if mode not in ("peft_fp32", "unmerged_compute"):
        raise ValueError(f"attach_unmerged handles the unmerged modes, got {mode!r}.")
    adapter_dtype = torch.float32 if mode == "peft_fp32" else compute_dtype
    names = inject_lora(model, spec, dtype=adapter_dtype)
    expected = set(lora_parameter_names(model))
    if set(tensors) != expected:
        raise ValueError(f"adapter tensors do not match the LoRA targets: missing={sorted(expected - set(tensors))[:3]} "
                         f"unexpected={sorted(set(tensors) - expected)[:3]}")
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name in expected:
            parameters[name].copy_(tensors[name].to(adapter_dtype))   # fp32 -> bf16 rounds like FSDP's param cast
    if mode == "peft_fp32":
        for module in model.modules():
            if type(module) is LoRALinear:
                module.__class__ = PeftLinear
    return names


__all__ = ["MODES", "PeftLinear", "attach_unmerged", "spec_from_metadata"]

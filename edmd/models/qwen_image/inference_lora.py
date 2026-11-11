"""LoRA numerics of the Qwen-Image-2.1 inference denoiser (``options.inference_lora``, explained in ``models/base.py``).

Qwen's ``unmerged_fp32`` rounds differently from the ``peft_fp32`` of FLUX.2 / SD3: each of the fp32 adapter's two
matmuls is rounded to the compute dtype (what fp32 adapter layers in a bf16 transformer compute on Ascend), not once at
the end. Written out, it also runs on the CPU.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn

from ...runtime.lora import LoRALinear

INFERENCE_LORA_MODES = ("unmerged_fp32", "unmerged_compute", "merged")


class AdapterRoundedLoRALinear(LoRALinear):
    """``LoRALinear`` whose adapter matmuls each run in the adapter's dtype and round back to the input's dtype."""

    def forward(self, input: Tensor) -> Tensor:  # noqa: A002 - nn.Linear signature
        result = F.linear(input, self.weight, self.bias)
        dtype = input.dtype
        hidden = self.lora_dropout(input)
        a = F.linear(hidden.to(self.lora_A.weight.dtype), self.lora_A.weight).to(dtype)
        update = F.linear(a.to(self.lora_B.weight.dtype), self.lora_B.weight).to(dtype)
        return result + update * self.lora_scaling


def use_adapter_rounding(model: nn.Module) -> int:
    """Switch every injected ``LoRALinear`` to :class:`AdapterRoundedLoRALinear` in place; returns how many."""
    count = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.__class__ = AdapterRoundedLoRALinear
            count += 1
    return count


__all__ = ["AdapterRoundedLoRALinear", "INFERENCE_LORA_MODES", "use_adapter_rounding"]

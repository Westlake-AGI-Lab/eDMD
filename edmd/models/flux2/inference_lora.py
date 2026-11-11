"""LoRA numerics of the FLUX.2 inference denoiser (``options.inference_lora``): ``unmerged_compute`` (default),
``peft_fp32`` (fp32 adapter in peft's cast order, :class:`PeftOrderLoRALinear`) or ``merged`` (fp32 delta added in
the base dtype); see ``ModelAdapter.load_inference_denoiser``.
"""

from __future__ import annotations

from ...runtime.lora import PeftOrderLoRALinear, use_peft_order   # one implementation, shared with SD3

INFERENCE_LORA_MODES = ("peft_fp32", "unmerged_compute", "merged")


__all__ = ["INFERENCE_LORA_MODES", "PeftOrderLoRALinear", "use_peft_order"]

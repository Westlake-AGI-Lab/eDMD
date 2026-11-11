"""Ascend fixes for QwenImage21, applied by :func:`install`:

* **RoPE**: CANN cannot index ``complex64`` tensors, so the rope tables are gathered on the CPU and returned as the
  real ``(cos, sin)`` pair of the same rotation (each entry repeated for the interleaved layout, packed on the last
  axis so the KV-cache decode's ``rotary_emb[prefix_len:]`` still slices the sequence); the rotation becomes the
  exactly equal ``x * cos + rotate_half(x) * sin``. Tables are cached per rope module.
* **RMSNorm**: diffusers calls ``torch_npu.npu_rms_norm`` whenever ``torch_npu`` imports, which raises on CPU
  tensors; the patch dispatches on the tensor's device.
* **Attention mask**: NPU queries get a broadcast key-padding mask (:func:`expand_key_padding_mask`).

Each patch keeps the upstream function as ``_edmd_original``, so ``install(force=True)`` never nests patches.
"""

from __future__ import annotations

from collections import OrderedDict

import torch

_INSTALLED = False
ORIGINAL_MARKERS = ("_edmd_original",)
ROPE_CACHE_ATTRIBUTES = ("_edmd_rope_cache",)


def _unwrap(function):
    """Follow patch markers back to the upstream function."""
    seen = 0
    while seen < 8:
        inner = next((getattr(function, marker) for marker in ORIGINAL_MARKERS if hasattr(function, marker)), None)
        if inner is None:
            return function
        function = inner
        seen += 1
    raise RuntimeError("npu_compat: patch markers form a cycle.")


def _mark(patched, original):
    for marker in ORIGINAL_MARKERS:
        setattr(patched, marker, original)
    return patched


def _real_rope_tables(freqs_cis: torch.Tensor, device: torch.device) -> torch.Tensor:
    """``[S, 1, D, 2]`` holding ``(cos, sin)`` from complex ``[S, D/2]`` (head axis inserted here)."""
    cos = torch.repeat_interleave(freqs_cis.real, 2, dim=-1).unsqueeze(1)
    sin = torch.repeat_interleave(freqs_cis.imag, 2, dim=-1).unsqueeze(1)
    return torch.stack([cos, sin], dim=-1).to(device=device, dtype=torch.float32)


def apply_rotary_emb_real(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    use_real: bool = False,
    use_real_unbind_dim: int = -1,
) -> torch.Tensor:
    """Rotate ``[B, S, H, D]`` by the packed ``(cos, sin)`` tables (``apply_rotary_emb_qwen``'s signature)."""
    cos, sin = freqs_cis.unbind(-1)
    values = x.float()
    even, odd = values.reshape(*values.shape[:-1], -1, 2).unbind(-1)
    rotated = torch.stack([-odd, even], dim=-1).flatten(-2)
    return (values * cos + rotated * sin).type_as(x)


#: RoPE tables kept per module (LRU). The key holds the prompt length, so an unbounded cache would grow by a 4-17 MB
#: entry per distinct length; the forwards of one training step share their key, so a few entries suffice.
ROPE_CACHE_ENTRIES = 8


def _cached_rope_forward(original):
    def forward(self, img_shapes, image_pad_mask, device):
        key = (tuple(tuple(shape) for shape in img_shapes), int(image_pad_mask.shape[-1]),
               int(image_pad_mask.sum()), str(device))
        cache = getattr(self, "_edmd_rope_cache", None)
        if cache is None:
            cache = OrderedDict()
            self._edmd_rope_cache = cache
        if key in cache:
            cache.move_to_end(key)
        else:
            cpu = torch.device("cpu")
            freqs_cis = original(self, img_shapes, image_pad_mask.to(cpu), device=cpu)
            cache[key] = _real_rope_tables(freqs_cis, device)
            while len(cache) > ROPE_CACHE_ENTRIES:
                cache.popitem(last=False)
        return cache[key]

    return _mark(forward, original)


def _device_aware_rms_norm(original):
    """Route ``RMSNorm`` to the fused Ascend kernel only for NPU tensors; elsewhere the upstream eager math."""

    def forward(self, hidden_states):
        if hidden_states.device.type != "npu":
            input_dtype = hidden_states.dtype
            variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
            if self.weight is None:
                return hidden_states.to(input_dtype)
            if self.weight.dtype in (torch.float16, torch.bfloat16):
                hidden_states = hidden_states.to(self.weight.dtype)
            hidden_states = hidden_states * self.weight
            if getattr(self, "bias", None) is not None:
                hidden_states = hidden_states + self.bias
            return hidden_states
        return original(self, hidden_states)

    return _mark(forward, original)


def expand_key_padding_mask(attn_mask, query: torch.Tensor):
    """``[B, 1, 1, Skv]`` -> ``[B, 1, Sq, Skv]`` (materialized) for NPU queries ``[B, Sq, H, D]``; else unchanged.

    Ascend's ``FlashAttentionScore`` (torch_npu's SDPA) rejects the broadcast query axis of the key-padding mask the
    processor passes for padded prompt batches; without this, such a batch crashes. Broadcasting changes no value.
    """
    if (isinstance(attn_mask, torch.Tensor) and attn_mask.dim() == 4 and attn_mask.shape[2] == 1
            and query.device.type == "npu" and query.shape[1] > 1):
        batch, heads, _, keys = attn_mask.shape
        return attn_mask.expand(batch, heads, query.shape[1], keys).contiguous()
    return attn_mask


def _npu_attention_mask(original):
    def dispatch(query, key, value, attn_mask=None, *args, **kwargs):
        return original(query, key, value, expand_key_padding_mask(attn_mask, query), *args, **kwargs)

    return _mark(dispatch, original)


def install(force: bool = False) -> bool:
    """Patch the rope table builder, the rotation, RMSNorm and the NPU attention mask. Idempotent."""
    global _INSTALLED
    if _INSTALLED and not force:
        return False
    from diffusers.models import normalization
    from diffusers.models.transformers import transformer_qwenimage21 as module

    module.QwenImage21Rope.forward = _cached_rope_forward(_unwrap(module.QwenImage21Rope.forward))
    # `_qwenimage21_prepare_qkv` and the processors resolve these names as module globals at call time.
    module.apply_rotary_emb_qwen = apply_rotary_emb_real
    module.dispatch_attention_fn = _npu_attention_mask(_unwrap(module.dispatch_attention_fn))
    normalization.RMSNorm.forward = _device_aware_rms_norm(_unwrap(normalization.RMSNorm.forward))
    _INSTALLED = True
    return True


def is_installed() -> bool:
    return _INSTALLED


def drop_rope_cache(rope: torch.nn.Module) -> None:
    """Forget cached tables (after the rope freqs were rebuilt, e.g. by ``reset_derived``)."""
    for name in ROPE_CACHE_ATTRIBUTES:
        if hasattr(rope, name):
            delattr(rope, name)


__all__ = ["apply_rotary_emb_real", "drop_rope_cache", "expand_key_padding_mask", "install", "is_installed"]

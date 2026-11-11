"""Perturbed-attention guidance (PAG) for Qwen-Image-2.1: the frozen teacher read with a degraded attention map.

In the listed blocks every target-image token attends to the prompt prefix and to itself only (identity image map);
everything else is the model's, including RoPE through ``_qwenimage21_prepare_qkv`` (so ``npu_compat``'s RoPE applies).
Prompt rows never attend to the image, so they are unchanged.

Covers prefill (``segments`` given; every training read) and cached decode (``segments is None``). The processor is a
plain attribute of each block's attention module, so the swap also works under FSDP2. Every rank must enter the
context for the same forwards; no-grad reads only.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Iterator, Sequence

import torch


def identity_rows(query, prefix_key, prefix_value, own_key, own_value, prefix_valid=None):
    """Target rows with the identity image map. ``query/own_*``: [B, Li, H, D]; ``prefix_*``: [B, Lp, H, D];
    ``prefix_valid``: [B, Lp] bool or None. Softmax over (the prefix keys, the row's own key), computed in float32
    without an Li x Li mask; returns [B, Li, H, D] in the query's dtype."""
    scale = 1.0 / math.sqrt(query.shape[-1])
    q = query.float().permute(0, 2, 1, 3)                                   # [B, H, Li, D]
    kp, vp = (x.float().permute(0, 2, 1, 3) for x in (prefix_key, prefix_value))
    ko, vo = (x.float().permute(0, 2, 1, 3) for x in (own_key, own_value))
    logits_prefix = torch.matmul(q, kp.transpose(-1, -2)) * scale           # [B, H, Li, Lp]
    if prefix_valid is not None:
        logits_prefix = logits_prefix.masked_fill(~prefix_valid.bool()[:, None, None, :], float("-inf"))
    logit_self = (q * ko).sum(dim=-1, keepdim=True) * scale                 # [B, H, Li, 1]
    weights = torch.softmax(torch.cat([logits_prefix, logit_self], dim=-1), dim=-1)
    lp = prefix_key.shape[1]
    out = torch.matmul(weights[..., :lp], vp) + weights[..., lp:] * vo
    return out.permute(0, 2, 1, 3).to(query.dtype)


class QwenImageIdentityAttnProcessor:
    """``QwenImage21AttnProcessor`` with the target image's image-to-image attention map replaced by the identity."""

    def __call__(self, attn, hidden_states, attention_mask=None, rotary_emb=None, layer_cache=None,
                 kv_cache_mode=None, cache_write_slice=None, segments=None, key_valid=None):
        from diffusers.models.attention_dispatch import dispatch_attention_fn
        from diffusers.models.transformers.transformer_qwenimage21 import _qwenimage21_prepare_qkv

        query, key, value, seq_len_q = _qwenimage21_prepare_qkv(
            attn, hidden_states, rotary_emb, layer_cache, kv_cache_mode, cache_write_slice)
        if segments is None:
            # decode: the queries are the target tokens; the keys are [cached prefix, target]
            prefix_len = key.shape[1] - query.shape[1]
            if prefix_len <= 0:
                raise ValueError("The degraded decode path expects a cached prefix in front of the target keys.")
            valid = None
            if attention_mask is not None:
                if attention_mask.dtype != torch.bool or attention_mask.ndim != 4:
                    raise ValueError("The degraded decode path expects the boolean key-valid mask [B, 1, 1, Lk].")
                valid = attention_mask[:, 0, 0, :prefix_len]
            hidden = identity_rows(query, key[:, :prefix_len], value[:, :prefix_len], key[:, prefix_len:],
                                   value[:, prefix_len:], valid)
        else:
            prefix_len = segments[-1][1] if segments else 0
            if prefix_len <= 0:
                raise ValueError("The degraded prefill path expects a prompt prefix.")
            outputs = []
            for start, end, is_text in segments:            # diffusers' prefix handling (Apache-2.0), verbatim
                seg_mask = None
                if is_text:
                    seg_len = end - start
                    seg_mask = torch.cat([
                        torch.ones(seg_len, start, dtype=torch.bool, device=query.device),
                        torch.tril(torch.ones(seg_len, seg_len, dtype=torch.bool, device=query.device)),
                    ], dim=1)[None, None]
                if key_valid is not None:
                    seg_key_valid = key_valid[:, None, None, :end]
                    seg_mask = seg_key_valid if seg_mask is None else (seg_mask & seg_key_valid)
                outputs.append(dispatch_attention_fn(query[:, start:end], key[:, :end], value[:, :end],
                                                     attn_mask=seg_mask, dropout_p=0.0, backend=None))
            valid = None if key_valid is None else key_valid[:, :prefix_len]
            outputs.append(identity_rows(query[:, prefix_len:], key[:, :prefix_len], value[:, :prefix_len],
                                         key[:, prefix_len:], value[:, prefix_len:], valid))
            hidden = torch.cat(outputs, dim=1)
        hidden = hidden[:, :seq_len_q].flatten(2, 3).type_as(query)
        hidden = attn.to_out[0](hidden)
        return attn.to_out[1](hidden)


def validate_blocks(blocks: Sequence[int], num_blocks: int | None = None) -> tuple[int, ...]:
    """A non-empty tuple of distinct non-negative block indices (below ``num_blocks`` when given)."""
    if isinstance(blocks, (str, bytes)) or not isinstance(blocks, Sequence) or not blocks:
        raise ValueError("Degraded blocks must be a non-empty list of block indices.")
    if any(isinstance(b, bool) or not isinstance(b, int) or b < 0 for b in blocks) or len(set(blocks)) != len(blocks):
        raise ValueError(f"Degraded blocks must be distinct non-negative integers, got {list(blocks)}.")
    if num_blocks is not None and max(blocks) >= num_blocks:
        raise ValueError(f"Degraded block {max(blocks)} does not exist (the model has {num_blocks} blocks).")
    return tuple(int(b) for b in blocks)


def listed_blocks(model) -> list:
    """``model.transformer_blocks``, each a ``QwenImage21TransformerBlock`` or the subclass FSDP2's ``fully_shard``
    swaps in (the sharded training teacher; ``components.transformer_blocks`` matches the exact class only)."""
    from .components import BLOCK_CLASS

    blocks = list(getattr(model, "transformer_blocks", []))
    if not blocks or not all(any(cls.__name__ == BLOCK_CLASS for cls in type(block).__mro__) for block in blocks):
        kinds = sorted({type(block).__name__ for block in blocks})
        raise RuntimeError(f"Expected model.transformer_blocks to list {BLOCK_CLASS} modules (got {len(blocks)}: "
                           f"{kinds}); the upstream model changed.")
    return blocks


@contextmanager
def degraded_attention(model, blocks: Sequence[int]) -> Iterator[None]:
    """Within the block, ``model.transformer_blocks[i].attn`` for i in ``blocks`` use the identity image map; the
    original processor objects come back afterwards (also on error)."""
    listed = listed_blocks(model)
    indices = validate_blocks(blocks, len(listed))
    saved = [(listed[b].attn, listed[b].attn.processor) for b in indices]
    try:
        for module, _ in saved:
            module.processor = QwenImageIdentityAttnProcessor()
        yield
    finally:
        for module, processor in saved:
            module.processor = processor


__all__ = ["QwenImageIdentityAttnProcessor", "degraded_attention", "identity_rows", "listed_blocks", "validate_blocks"]

"""SD3 prompt conditioning: a port of ``StableDiffusion3Pipeline.encode_prompt`` (one image per prompt).

CLIP-L | CLIP-G hidden states (77 tokens; the length and dtype of the *first* tokenizer / encoder for both, as in the
pipeline) are zero-padded to T5's width and put before the T5 tokens: ``prompt_embeds`` is
``[B, 77 + max_sequence_length, 4096]``, ``pooled_prompt_embeds`` CLIP-L pooled | CLIP-G pooled, ``[B, 2048]``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SD3Conditioning:
    """Text conditioning of one batch: what ``SD3Transformer2DModel`` takes besides latents and time."""

    prompt_embeds: Tensor           # [B, 77 + max_sequence_length, joint_attention_dim]
    pooled_prompt_embeds: Tensor    # [B, pooled_projection_dim]

    def __post_init__(self) -> None:
        if self.prompt_embeds.ndim != 3 or self.pooled_prompt_embeds.ndim != 2:
            raise ValueError("prompt_embeds must be [B, L, D] and pooled_prompt_embeds [B, P].")
        if self.prompt_embeds.shape[0] != self.pooled_prompt_embeds.shape[0]:
            raise ValueError("prompt_embeds and pooled_prompt_embeds disagree on the batch size.")

    @property
    def batch_size(self) -> int:
        return int(self.prompt_embeds.shape[0])

    @staticmethod
    def cat(items: Sequence["SD3Conditioning"]) -> "SD3Conditioning":
        """Batch concatenation (the pipeline's CFG batch is ``cat([negative, positive])``)."""
        if not items:
            raise ValueError("Nothing to concatenate.")
        return SD3Conditioning(torch.cat([item.prompt_embeds for item in items], dim=0),
                               torch.cat([item.pooled_prompt_embeds for item in items], dim=0))

    def select(self, index: Tensor | slice | Sequence[int]) -> "SD3Conditioning":
        return SD3Conditioning(self.prompt_embeds[index], self.pooled_prompt_embeds[index])

    def to(self, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> "SD3Conditioning":
        return SD3Conditioning(self.prompt_embeds.to(device=device, dtype=dtype),
                               self.pooled_prompt_embeds.to(device=device, dtype=dtype))


class SD3PromptEncoder:
    """Frozen CLIP-L + CLIP-G + T5 encoders and their tokenizers, encoding exactly like the pipeline."""

    def __init__(self, tokenizers: Sequence, text_encoders: Sequence[nn.Module], *, max_sequence_length: int = 256,
                 clip_skip: int | None = None) -> None:
        if len(tokenizers) != 3 or len(text_encoders) != 3:
            raise ValueError("SD3 needs three tokenizers and three text encoders (CLIP-L, CLIP-G, T5).")
        if not 1 <= int(max_sequence_length) <= 512:   # check_inputs: max_sequence_length cannot exceed 512
            raise ValueError("max_sequence_length must lie in [1, 512].")
        self.tokenizers = tuple(tokenizers)
        self.text_encoders = tuple(text_encoders)
        self.max_sequence_length = int(max_sequence_length)
        self.clip_skip = clip_skip
        self.tokenizer_max_length = int(self.tokenizers[0].model_max_length)   # pipeline.tokenizer_max_length

    @property
    def device(self) -> torch.device:
        for parameter in self.text_encoders[0].parameters():
            return parameter.device
        raise RuntimeError("CLIP-L has no parameters.")

    def _clip(self, index: int, prompts: list[str], clip_skip: int | None) -> tuple[Tensor, Tensor]:
        tokenizer, encoder = self.tokenizers[index], self.text_encoders[index]
        text_input_ids = tokenizer(prompts, padding="max_length", max_length=self.tokenizer_max_length,
                                   truncation=True, return_tensors="pt").input_ids
        untruncated_ids = tokenizer(prompts, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed = tokenizer.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1: -1])
            logger.warning("CLIP truncated the prompt to %d tokens, removing: %s", self.tokenizer_max_length, removed)
        output = encoder(text_input_ids.to(self.device), output_hidden_states=True)
        pooled = output[0]
        hidden = output.hidden_states[-2] if clip_skip is None else output.hidden_states[-(clip_skip + 2)]
        hidden = hidden.to(dtype=self.text_encoders[0].dtype, device=self.device)
        return hidden, pooled

    def _t5(self, prompts: list[str]) -> Tensor:
        tokenizer, encoder = self.tokenizers[2], self.text_encoders[2]
        text_input_ids = tokenizer(prompts, padding="max_length", max_length=self.max_sequence_length,
                                   truncation=True, add_special_tokens=True, return_tensors="pt").input_ids
        untruncated_ids = tokenizer(prompts, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed = tokenizer.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1: -1])
            logger.warning("T5 truncated the prompt to %d tokens, removing: %s", self.max_sequence_length, removed)
        embeds = encoder(text_input_ids.to(self.device))[0]
        return embeds.to(dtype=encoder.dtype, device=self.device)

    @torch.no_grad()
    def encode(self, prompts: Sequence[str], *, clip_skip: int | None | str = "default") -> SD3Conditioning:
        """``encode_prompt``'s positive branch (``clip_skip="default"`` uses the configured value)."""
        prompts = [prompts] if isinstance(prompts, str) else list(prompts)
        if not prompts or not all(isinstance(prompt, str) for prompt in prompts):
            raise ValueError("prompts must be a non-empty list of strings.")
        skip = self.clip_skip if clip_skip == "default" else clip_skip
        prompt_embed, pooled_prompt_embed = self._clip(0, prompts, skip)
        prompt_2_embed, pooled_prompt_2_embed = self._clip(1, prompts, skip)
        clip_prompt_embeds = torch.cat([prompt_embed, prompt_2_embed], dim=-1)
        t5_prompt_embed = self._t5(prompts)
        clip_prompt_embeds = F.pad(clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1]))
        prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)
        pooled_prompt_embeds = torch.cat([pooled_prompt_embed, pooled_prompt_2_embed], dim=-1)
        return SD3Conditioning(prompt_embeds, pooled_prompt_embeds)

    def encode_negative(self, negative_prompt: str, batch_size: int) -> SD3Conditioning:
        """``encode_prompt``'s negative branch: ``[negative_prompt or ""] * batch_size`` with ``clip_skip=None``."""
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        return self.encode([negative_prompt or ""] * batch_size, clip_skip=None)


__all__ = ["SD3Conditioning", "SD3PromptEncoder"]

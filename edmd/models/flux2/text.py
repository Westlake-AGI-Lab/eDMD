"""Prompt encoding for FLUX.2 klein, as ``Flux2KleinPipeline`` encodes.

Each prompt goes through the Qwen chat template (one user turn, ``enable_thinking=False``), is padded / truncated to
``max_sequence_length``, and the encoder's hidden states of layers ``(9, 18, 27)`` (index 0 = embeddings) are
concatenated to ``[B, L, 3 * hidden]``. The only change is ``logits_to_keep=1``: the pipeline computes and discards
the LM-head logits of every position, which the hidden states do not depend on.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class Flux2Conditioning:
    """What the denoiser needs besides the latents and the time (opaque to the algorithm layer)."""

    embeds: Tensor     # [B, L, len(layers) * hidden], encoder dtype, on the encoder device
    text_ids: Tensor   # [B, L, 4] int64 (t, h, w, l) coordinates
    grid: tuple[int, int] | None = None   # image tokens (rows, cols) it was encoded for; None: the adapter's own

    def __post_init__(self) -> None:
        if self.embeds.ndim != 3 or self.text_ids.ndim != 3 or self.text_ids.shape[-1] != 4:
            raise ValueError(f"embeds [B,L,D] / text_ids [B,L,4] expected, got {tuple(self.embeds.shape)} / "
                             f"{tuple(self.text_ids.shape)}")
        if tuple(self.embeds.shape[:2]) != tuple(self.text_ids.shape[:2]):
            raise ValueError("embeds and text_ids disagree on batch or sequence length.")

    @property
    def batch_size(self) -> int:
        return int(self.embeds.shape[0])

    @property
    def sequence_length(self) -> int:
        return int(self.embeds.shape[1])

    def to(self, device: torch.device | str) -> "Flux2Conditioning":
        return Flux2Conditioning(embeds=self.embeds.to(device), text_ids=self.text_ids.to(device), grid=self.grid)

    def expand(self, batch_size: int) -> "Flux2Conditioning":
        """A batch-1 conditioning repeated (as a view) ``batch_size`` times (the negative prompt is encoded once)."""
        if self.batch_size != 1:
            raise ValueError("Only a batch-1 conditioning can be expanded.")
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive.")
        return Flux2Conditioning(embeds=self.embeds.expand(int(batch_size), -1, -1),
                                 text_ids=self.text_ids.expand(int(batch_size), -1, -1), grid=self.grid)

    def at_grid(self, height: int, width: int) -> "Flux2Conditioning":
        """The same conditioning for an image of ``height x width`` tokens."""
        return replace(self, grid=(int(height), int(width)))


def text_ids(batch_size: int, sequence_length: int, *, device: torch.device | str) -> Tensor:
    """``cartesian_prod(arange(1), arange(1), arange(1), arange(L))`` expanded to the batch (the pipeline's
    ``_prepare_text_ids``)."""
    coords = torch.cartesian_prod(
        torch.arange(1, device=device, dtype=torch.long),
        torch.arange(1, device=device, dtype=torch.long),
        torch.arange(1, device=device, dtype=torch.long),
        torch.arange(int(sequence_length), device=device, dtype=torch.long),
    )
    return coords.unsqueeze(0).expand(int(batch_size), -1, -1).contiguous()


def tokenize(tokenizer, prompts: Sequence[str], *, max_sequence_length: int) -> tuple[Tensor, Tensor]:
    """Chat-template each prompt and pad/truncate it to ``max_sequence_length`` (one call per prompt, as upstream)."""
    ids: list[Tensor] = []
    masks: list[Tensor] = []
    for prompt in prompts:
        if not isinstance(prompt, str):
            raise TypeError(f"prompts must be strings, got {type(prompt).__name__}")
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        tokens = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True,
                           max_length=int(max_sequence_length))
        ids.append(tokens["input_ids"])
        masks.append(tokens["attention_mask"])
    return torch.cat(ids, dim=0), torch.cat(masks, dim=0)


@torch.no_grad()
def encode_prompts(
    text_encoder: nn.Module,
    tokenizer,
    prompts: Sequence[str],
    *,
    max_sequence_length: int,
    layers: Sequence[int],
    device: torch.device | str,
    dtype: torch.dtype,
) -> Flux2Conditioning:
    prompts = list(prompts)
    if not prompts:
        raise ValueError("encode needs at least one prompt.")
    layers = tuple(int(layer) for layer in layers)
    input_ids, attention_mask = tokenize(tokenizer, prompts, max_sequence_length=max_sequence_length)
    output = text_encoder(
        input_ids=input_ids.to(device),
        attention_mask=attention_mask.to(device),
        output_hidden_states=True,
        use_cache=False,
        logits_to_keep=1,
    )
    hidden = output.hidden_states
    if hidden is None or max(layers) >= len(hidden):
        raise RuntimeError(f"The text encoder returned {0 if hidden is None else len(hidden)} hidden states; "
                           f"layers {layers} requested.")
    stacked = torch.stack([hidden[layer] for layer in layers], dim=1).to(device=device, dtype=dtype)
    batch, count, length, width = stacked.shape
    embeds = stacked.permute(0, 2, 1, 3).reshape(batch, length, count * width).contiguous()
    return Flux2Conditioning(embeds=embeds, text_ids=text_ids(batch, length, device=device))


__all__ = ["Flux2Conditioning", "encode_prompts", "text_ids", "tokenize"]

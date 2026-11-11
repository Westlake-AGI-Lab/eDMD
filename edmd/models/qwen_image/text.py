"""Prompt encoding exactly as ``QwenImage21Pipeline._get_qwen_prompt_embeds`` (text-to-image branch).

The frozen Qwen3-VL encoder runs once per prompt batch, shared by teacher, student and fake. Details that silently
change what the transformer sees when wrong:

1. The template is a raw string, not ``apply_chat_template`` (the two tokenize differently).
2. The system prefix is dropped from the hidden states; its length comes from the processor's tokenization of the
   system message, never hardcoded.
3. The encoder's final RMSNorm is bypassed with a forward hook, as in the pipeline (transformers >= 5 ties
   ``hidden_states[-1]`` to the normalized output).
4. Left padding in, right padding out: zeros plus a mask, ``None`` when nothing is padded.

No truncation. An empty prompt is encoded as a single space, like the pipeline (also CFG's unconditional ``""``).
As in the pipeline, left padding shifts a shorter prompt's positions in a batch, so it matches its solo encoding only to
rounding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor

SYSTEM_PROMPT = "Comprehend and analyze the provided prompt."
#: Error for a prompt batch of mixed token lengths (see ``layout.build_condition``).
MIXED_LENGTHS = (
    "prompts of different token lengths {lengths} in one batch: QwenImage21 lays every sample out from img_mask[0], "
    "so each image block starts at the padded text length and the shorter prompts' image-text RoPE offsets shift -- "
    "on real weights a 38-token prompt batched with a 1058-token one changed its velocity by 16.7% (relative), the "
    "longest by 0.5%. Use a per-rank prompt batch size of 1 (evaluation renders one image at a time) or "
    "equal-length prompts; encode(..., allow_padding=True) accepts the pipeline's behavior."
)
PROMPT_TEMPLATE_T2I = (
    f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
    f"<|im_start|>user\n{{}}<|im_end|>\n"
    f"<|im_start|>assistant\n"
)


@dataclass(frozen=True)
class PromptBatch:
    """Encoder output for one batch of prompts."""

    embeds: Tensor            # [B, L, context_in_dim], right-padded with zeros
    mask: Tensor | None       # [B, L] bool, or None when nothing is padded
    lengths: tuple[int, ...]  # valid tokens per sample

    def to(self, device: torch.device | str) -> "PromptBatch":
        return PromptBatch(embeds=self.embeds.to(device), mask=None if self.mask is None else self.mask.to(device),
                           lengths=self.lengths)

    @property
    def batch_size(self) -> int:
        return int(self.embeds.shape[0])

    @property
    def sequence_length(self) -> int:
        return int(self.embeds.shape[1])


def text_decoder(text_encoder: torch.nn.Module) -> torch.nn.Module:
    """The Qwen3-VL language model (the module owning ``layers`` and the final ``norm``)."""
    return getattr(text_encoder.model, "language_model", text_encoder.model)


class QwenPromptEncoder:
    """The frozen Qwen3-VL text encoder and its processor, called exactly as the pipeline calls them."""

    def __init__(self, text_encoder: torch.nn.Module, processor, *, device: torch.device | str | None = None) -> None:
        self.text_encoder = text_encoder
        self.processor = processor
        system_message = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]
        tokens = processor.apply_chat_template(system_message, tokenize=True, return_dict=False)
        self.drop_idx = len(tokens[0])
        self.device = torch.device(device) if device is not None else next(text_encoder.parameters()).device

    @staticmethod
    def _extract_masked_hidden(hidden_states: Tensor, mask: Tensor) -> list[Tensor]:
        bool_mask = mask.bool()
        return list(torch.split(hidden_states[bool_mask], bool_mask.sum(dim=1).tolist(), dim=0))

    @torch.no_grad()
    def encode(self, prompts: Sequence[str], *, require_equal_lengths: bool = False) -> PromptBatch:
        """``require_equal_lengths`` raises :data:`MIXED_LENGTHS` before the encoder forward when the prompts' token
        counts differ (off by default: the pipeline's behavior)."""
        if isinstance(prompts, str):
            raise TypeError("encode takes a sequence of prompts, not a single string.")
        prompts = list(prompts)
        if not prompts:
            raise ValueError("encode needs at least one prompt.")
        for prompt in prompts:
            if not isinstance(prompt, str):
                raise TypeError(f"prompts must be strings, got {type(prompt).__name__} (None is not an empty prompt).")
        texts = [PROMPT_TEMPLATE_T2I.format(prompt if prompt else " ") for prompt in prompts]
        inputs = self.processor(text=texts, padding=True, padding_side="left", return_tensors="pt").to(self.device)
        if require_equal_lengths and not bool(inputs.attention_mask.bool().all()):
            valid = (inputs.attention_mask.bool().sum(dim=1) - self.drop_idx).tolist()
            raise ValueError(MIXED_LENGTHS.format(lengths=valid))
        forward_kwargs = {"input_ids": inputs.input_ids, "attention_mask": inputs.attention_mask,
                          "output_hidden_states": True}
        if hasattr(inputs, "mm_token_type_ids"):
            forward_kwargs["mm_token_type_ids"] = inputs.mm_token_type_ids

        handle = text_decoder(self.text_encoder).norm.register_forward_hook(lambda module, args, output: args[0])
        try:
            outputs = self.text_encoder(**forward_kwargs)
        finally:
            handle.remove()

        split = [entry[self.drop_idx:] for entry
                 in self._extract_masked_hidden(outputs.hidden_states[-1], inputs.attention_mask)]
        lengths = tuple(int(entry.shape[0]) for entry in split)
        if min(lengths) <= 0:
            raise ValueError("A prompt encoded to no tokens after dropping the system prefix.")
        width = max(lengths)
        embeds = torch.stack([torch.cat([entry, entry.new_zeros(width - entry.shape[0], entry.shape[1])])
                              for entry in split])
        mask = torch.stack([torch.cat([torch.ones(length, dtype=torch.bool, device=embeds.device),
                                       torch.zeros(width - length, dtype=torch.bool, device=embeds.device)])
                            for length in lengths])
        return PromptBatch(embeds=embeds, mask=None if bool(mask.all()) else mask, lengths=lengths)


__all__ = ["MIXED_LENGTHS", "PROMPT_TEMPLATE_T2I", "PromptBatch", "QwenPromptEncoder", "SYSTEM_PROMPT", "text_decoder"]

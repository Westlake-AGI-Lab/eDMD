"""PickScore (https://github.com/yuvalkirstain/PickScore).

What: a CLIP ViT-H/14 fine-tuned on Pick-a-Pic human preferences. Score = ``logit_scale.exp() * cos(text, image)``,
the unnormalized logit (typically 18-25; no division by 26). A relative scale.

Preprocessing: the float ``[0, 1]`` batch goes back to uint8 RGB PIL images (``round(x * 255)``), then the
``laion/CLIP-ViT-H-14-laion2B-s32B-b79K`` processor (``transformers`` ``AutoProcessor``: resize the short side to
224 bicubic, center-crop 224, CLIP mean/std); text padded and truncated to 77 tokens. Embeddings are L2-normalized.
float32, batch size 8.

Weights: ``yuvalkirstain/PickScore_v1`` (model), ``laion/CLIP-ViT-H-14-laion2B-s32B-b79K`` (processor).

Follows CDM ``cdm/rewards/pickscore_scorer.py`` (MIT License, Copyright (c) 2024 FoundationVision; see eval/NOTICE).
"""

from __future__ import annotations

import torch

from .common import to_pil


class PickScore:
    def __init__(self, model_dir: str, processor_dir: str, device: str, dtype: torch.dtype = torch.float32):
        from transformers import AutoModel, AutoProcessor

        self.device, self.dtype = device, dtype
        self.processor = AutoProcessor.from_pretrained(processor_dir)
        self.model = AutoModel.from_pretrained(model_dir).to(device=device, dtype=dtype).eval()

    @torch.no_grad()
    def __call__(self, images: torch.Tensor, prompts: list[str]) -> torch.Tensor:
        image_inputs = self.processor(images=to_pil(images), padding=True, truncation=True, max_length=77,
                                      return_tensors="pt")
        image_inputs = {key: value.to(self.device) for key, value in image_inputs.items()}
        text_inputs = self.processor(text=prompts, padding=True, truncation=True, max_length=77, return_tensors="pt")
        text_inputs = {key: value.to(self.device) for key, value in text_inputs.items()}

        image_embs = self.model.get_image_features(**image_inputs)
        if not isinstance(image_embs, torch.Tensor):
            image_embs = image_embs.pooler_output
        image_embs = image_embs / image_embs.norm(p=2, dim=-1, keepdim=True)

        text_embs = self.model.get_text_features(**text_inputs)
        if not isinstance(text_embs, torch.Tensor):
            text_embs = text_embs.pooler_output
        text_embs = text_embs / text_embs.norm(p=2, dim=-1, keepdim=True)

        logit_scale = self.model.logit_scale.exp()
        return (logit_scale * (text_embs @ image_embs.T)).diag().float().cpu()

"""CLIP score (https://github.com/openai/CLIP), with OpenAI CLIP ViT-B/32.

What: how well an image matches its prompt in CLIP's joint embedding space. Score = ``logits_per_image / 100`` of
the (image i, prompt i) pair, i.e. ``logit_scale.exp() / 100 * cos(image, text)`` with ``logit_scale.exp() = 100`` for
this checkpoint: the cosine similarity of the L2-normalized embeddings (about 0.2-0.35). This is not CLIPScore
(Hessel et al., https://arxiv.org/abs/2104.08718), which rescales to ``2.5 * max(cos, 0)``.

Preprocessing: on the float ``[0, 1]`` batch itself (no uint8 round trip), the processor's own steps as torchvision
transforms: resize the short side to 224 (``T.Resize``'s default, bilinear and antialiased, as in CDM's code; the
processor itself would resize bicubic), center-crop 224, CLIP mean/std. The prompt is padded and truncated to 77
tokens. float32, batch size 8.

Weights: ``openai/clip-vit-base-patch32``.

Follows CDM ``cdm/rewards/clip_scorer.py`` (MIT License, Copyright (c) 2024 FoundationVision; see eval/NOTICE), whose
code uses ViT-L/14; the eDMD paper reports ViT-B/32.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.transforms as T


def _size(value):
    """A processor size: ``{"shortest_edge": s}`` -> ``s`` (resize the short side), ``{"height", "width"}`` -> a pair."""
    if isinstance(value, dict):
        if "shortest_edge" in value:
            return value["shortest_edge"]
        if "height" in value:
            return (value["height"], value["width"])
    return value


class CLIPScore:
    def __init__(self, model_dir: str, device: str, dtype: torch.dtype = torch.float32):
        from transformers import CLIPModel, CLIPProcessor

        self.device, self.dtype = device, dtype
        self.model = CLIPModel.from_pretrained(model_dir).to(device=device, dtype=dtype).eval()
        self.processor = CLIPProcessor.from_pretrained(model_dir)
        image_processor = self.processor.image_processor
        config = image_processor.to_dict()
        steps = []
        if config.get("do_resize", True):
            steps.append(T.Resize(_size(config.get("size"))))
        if config.get("do_center_crop", True):
            steps.append(T.CenterCrop(_size(config.get("crop_size"))))
        if config.get("do_normalize", True):
            steps.append(T.Normalize(mean=image_processor.image_mean, std=image_processor.image_std))
        self.transform = T.Compose(steps) if steps else nn.Identity()

    @torch.no_grad()
    def __call__(self, images: torch.Tensor, prompts: list[str]) -> torch.Tensor:
        texts = self.processor(text=prompts, padding="max_length", truncation=True, return_tensors="pt").to(self.device)
        pixels = self.transform(images).to(device=self.device, dtype=self.dtype)
        outputs = self.model(pixel_values=pixels, **texts)
        return (outputs.logits_per_image.diagonal() / 100).float().cpu()

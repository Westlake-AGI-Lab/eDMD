"""HPSv2.1 -- Human Preference Score v2.1 (https://github.com/tgxs002/HPSv2).

What: an OpenCLIP ViT-H/14 fine-tuned on HPD v2 human preferences. Score = cosine similarity of the normalized image
and text features (no logit scale), typically 0.2-0.35. A relative scale.

Preprocessing: on the float ``[0, 1]`` tensor (no uint8 round trip): scale the long side to 224 (bicubic, torchvision
tensor resize), pad the short side symmetrically with 0 to 224 x 224 (the HPSv2 / OpenCLIP ``ResizeMaxSize``, so
nothing is cropped), CLIP mean/std. The OpenCLIP tokenizer of the ``hpsv2x`` package (77 tokens). Model created with
``precision="amp"`` (float32 weights), float32 inputs, batch size 8.

Weights: ``xswu/HPSv2`` ``HPS_v2.1_compressed.pt`` (the whole model, loaded strictly over the backbone);
``laion/CLIP-ViT-H-14-laion2B-s32B-b79K`` ``open_clip_pytorch_model.bin`` (the OpenCLIP backbone it is created from).

Follows CDM ``cdm/rewards/hpsv2_scorer.py`` (MIT License, Copyright (c) 2024 FoundationVision; see eval/NOTICE).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.transforms as T

OPENAI_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_STD = (0.26862954, 0.26130258, 0.27577711)


class ResizeMaxSize(nn.Module):
    """Scale the long side to ``max_size``, then pad to a ``max_size`` square (centered)."""

    def __init__(self, max_size: int, interpolation=T.InterpolationMode.BICUBIC, fill: int = 0):
        super().__init__()
        self.max_size, self.interpolation, self.fill = max_size, interpolation, fill

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        import torchvision.transforms.functional as F

        height, width = image.shape[-2:]
        scale = self.max_size / float(max(height, width))
        if scale != 1.0:
            new_size = tuple(round(dim * scale) for dim in (height, width))
            image = F.resize(image, list(new_size), self.interpolation)
            pad_h = self.max_size - new_size[0]
            pad_w = self.max_size - new_size[1]
            image = F.pad(image, padding=[pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2],
                          fill=self.fill)
        return image


class HPSv2:
    def __init__(self, checkpoint: str, openclip_backbone: str, device: str, dtype: torch.dtype = torch.float32):
        from hpsv2.src.open_clip import create_model, get_tokenizer

        self.device, self.dtype = device, dtype
        model = create_model("ViT-H-14", openclip_backbone, precision="amp", device=device, jit=False,
                             force_quick_gelu=False, force_custom_text=False, force_patch_dropout=False,
                             force_image_size=None, pretrained_image=False, output_dict=True)
        image_size = model.visual.image_size
        if isinstance(image_size, tuple):
            image_size = image_size[0]
        mean = getattr(model.visual, "image_mean", None) or OPENAI_MEAN
        std = getattr(model.visual, "image_std", None) or OPENAI_STD
        self.preprocess = T.Compose([ResizeMaxSize(image_size, fill=0), T.Normalize(mean=mean, std=std)])
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state["state_dict"])
        self.model = model.to(device).eval()
        self.tokenizer = get_tokenizer("ViT-H-14")

    @torch.no_grad()
    def __call__(self, images: torch.Tensor, prompts: list[str]) -> torch.Tensor:
        image = self.preprocess(images.to(dtype=self.dtype, device=self.device))
        text = self.tokenizer(prompts).to(self.device)
        out = self.model(image, text)
        logits = out["image_features"] @ out["text_features"].T
        return torch.diagonal(logits, 0).float().cpu()

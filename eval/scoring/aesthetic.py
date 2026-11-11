"""Aesthetic score: the LAION aesthetic predictor (https://github.com/christophschuhmann/improved-aesthetic-predictor).

What: an MLP on the L2-normalized CLIP ViT-L/14 image embedding, regressing the 1-10 human aesthetic rating
(the SAC+LOGOS+AVA1 linear MSE predictor). Image only: the prompt is ignored.

Preprocessing: the float ``[0, 1]`` batch goes back to a uint8 ``(N, 3, H, W)`` tensor (``round(x * 255)``), then the
``openai/clip-vit-large-patch14`` ``CLIPProcessor`` (resize the short side to 224 bicubic, center-crop 224, CLIP
mean/std). float32, batch size 8.

Weights: ``openai/clip-vit-large-patch14``; ``trl-lib/ddpo-aesthetic-predictor`` ``aesthetic-model.pth`` (a mirror of
the predictor's ``sac+logos+ava1-l14-linearMSE.pth``).

Follows CDM ``cdm/rewards/aesthetic_scorer.py`` (MIT License, Copyright (c) 2024 FoundationVision; see eval/NOTICE).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .common import to_uint8


class MLP(nn.Module):
    """The predictor head; 768 = the CLIP ViT-L/14 embedding width."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, embed: torch.Tensor) -> torch.Tensor:
        return self.layers(embed)


class Aesthetic:
    def __init__(self, clip_dir: str, mlp_checkpoint: str, device: str, dtype: torch.dtype = torch.float32):
        from transformers import CLIPModel, CLIPProcessor

        self.device, self.dtype = device, dtype
        self.clip = CLIPModel.from_pretrained(clip_dir).to(device=device, dtype=dtype).eval()
        self.processor = CLIPProcessor.from_pretrained(clip_dir)
        self.mlp = MLP().to(device=device, dtype=dtype).eval()
        self.mlp.load_state_dict(torch.load(mlp_checkpoint, map_location="cpu"))

    @torch.no_grad()
    def __call__(self, images: torch.Tensor, prompts: list[str] | None = None) -> torch.Tensor:
        inputs = self.processor(images=to_uint8(images), return_tensors="pt")
        inputs = {key: value.to(device=self.device, dtype=self.dtype) for key, value in inputs.items()}
        embed = self.clip.get_image_features(**inputs)
        if not isinstance(embed, torch.Tensor):
            embed = embed.pooler_output if hasattr(embed, "pooler_output") else embed[1]
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        return self.mlp(embed).squeeze(1).float().cpu()

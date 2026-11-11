"""ImageReward (https://github.com/THUDM/ImageReward).

What: ImageReward-v1.0, a BLIP backbone (ViT-L image encoder, BERT text encoder with cross-attention to the image)
with an MLP head, trained on human preference rankings of text-to-image outputs. Score = the head's output
standardized by the model's own constants, ``(r - 0.16717) / 1.03334``: roughly -2 to 2, a relative scale.

Computation: the ``image-reward`` package's ``ImageReward.score(prompt, image)``, one (prompt, image) pair at a time,
so the score does not depend on the batch. Preprocessing is the package's own: the float ``[0, 1]`` batch goes back
to uint8 RGB PIL images (``round(x * 255)``), then resize the short side to 224 (bicubic), center-crop 224, CLIP
mean/std; the prompt is padded and truncated to 35 BERT tokens. float32.

Weights: ``THUDM/ImageReward`` (``ImageReward.pt``, ``med_config.json``); the ``google-bert/bert-base-uncased``
tokenizer. ``ImageReward.load`` would fetch both from the Hugging Face Hub, so the model is built here from the local
files the way ``ImageReward.load`` builds it (``load_state_dict(strict=False)``).

Follows CDM ``cdm/rewards/imagereward_scorer.py`` (MIT License, Copyright (c) 2024 FoundationVision; see eval/NOTICE),
scoring each pair on its own where CDM takes the diagonal of an all-pairs matrix (the same numbers).
"""

from __future__ import annotations

import torch

from .common import to_pil


class ImageReward:
    def __init__(self, checkpoint: str, med_config: str, bert_dir: str, device: str, dtype: torch.dtype = torch.float32):
        from ImageReward.ImageReward import ImageReward as Model
        from ImageReward.models.BLIP import blip_pretrain
        from transformers import BertTokenizer

        def init_tokenizer():
            """The package's ``models/BLIP/blip.py`` ``init_tokenizer``, reading the local ``bert-base-uncased``."""
            tokenizer = BertTokenizer.from_pretrained(bert_dir)
            tokenizer.add_special_tokens({"bos_token": "[DEC]"})
            tokenizer.add_special_tokens({"additional_special_tokens": ["[ENC]"]})
            tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]
            return tokenizer

        upstream = blip_pretrain.init_tokenizer
        blip_pretrain.init_tokenizer = init_tokenizer
        try:
            model = Model(med_config=med_config, device=device)
        finally:
            blip_pretrain.init_tokenizer = upstream
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=False)
        self.model = model.to(device).eval().to(dtype=dtype)
        self.model.requires_grad_(False)

    @torch.no_grad()
    def __call__(self, images: torch.Tensor, prompts: list[str]) -> torch.Tensor:
        scores = [self.model.score(prompt, image) for prompt, image in zip(prompts, to_pil(images))]
        return torch.tensor(scores, dtype=torch.float32)

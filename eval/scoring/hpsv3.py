"""HPSv3 -- Human Preference Score v3 (https://github.com/MizzenAI/HPSv3).

What: a Qwen2-VL-7B-Instruct reward model trained with an uncertainty-aware ranking (Bradley-Terry) loss on human
preference pairs. ``HPSv3RewardInferencer.reward`` returns ``(mu, sigma)`` per (image, prompt); the score is ``mu``
(``sigma`` is kept as its uncertainty). A relative scale: only differences are meaningful, negative values are normal.

Preprocessing: the float ``[0, 1]`` batch goes back to uint8 RGB PIL images (``round(x * 255)``); the ``hpsv3``
package (pip ``hpsv3==1.0.0``) does the rest: Qwen2-VL processor at exactly 256 * 28 * 28 pixels and the HPSv3
instruction prompt. Batch size 1.

Weights: ``MizzenAI/HPSv3`` ``HPSv3.safetensors``; base ``Qwen/Qwen2-VL-7B-Instruct``, set as ``model_name_or_path``
in a temporary copy of the package's ``config/HPSv3_7B.yaml``.

Follows CDM ``cdm/rewards/hpsv3_scorer.py`` (MIT License, Copyright (c) 2024 FoundationVision; see eval/NOTICE).
"""

from __future__ import annotations

from pathlib import Path

import torch

from .common import local_config, to_pil


class HPSv3:
    def __init__(self, checkpoint: str, base_model: str, device: str):
        import hpsv3
        from hpsv3 import HPSv3RewardInferencer

        template = Path(hpsv3.__file__).parent / "config" / "HPSv3_7B.yaml"
        with local_config(template, base_model) as config:
            self.inferencer = HPSv3RewardInferencer(device=device, checkpoint_path=checkpoint,
                                                    config_path=str(config))

    @torch.no_grad()
    def __call__(self, images: torch.Tensor, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        rewards = self.inferencer.reward(to_pil(images), prompts).float().cpu()
        return rewards[:, 0], rewards[:, 1]

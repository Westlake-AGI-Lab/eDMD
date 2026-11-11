"""HPSv3++ (https://github.com/PlantPotatoOnMoon/HPSv3-PlusPlus).

What: a Bradley-Terry reward model like HPSv3, on a Qwen3-VL-8B-Instruct base, with FiLM conditioning on the
generator's capability (inferred from the image) and on an RL-iteration scalar ``iter_step``. ``reward`` returns
``(mu, sigma)``; the score is ``mu``. ``iter_step`` is fixed at 0.0, the upstream default for plain preference
scoring (the 0.3 -> 1.0 ramp is for use as an RL reward); 1.0 would shift every score by about +2.

Preprocessing: the image file is passed by path to the upstream ``HPSv3RewardInferencer.reward`` (Qwen3-VL processor
at 256 * 28 * 28 pixels, the HPSv3 instruction prompt). Batch size 1.

Weights: ``Junjun2333/HPSv3-PlusPlus`` ``hpsv3++.pth``; base ``Qwen/Qwen3-VL-8B-Instruct``, set as
``model_name_or_path`` in a temporary copy of the checkout's ``hpsv3/config/train_stage2.yaml``.

Environment: its own (``transformers>=4.57``, ``eval/requirements-hpsv3pp.txt``) with the checkout
``eval/HPSv3PlusPlus`` (not on PyPI) first on ``sys.path`` (``run.use_hpsv3pp_source``). It imports as ``hpsv3`` like
the HPSv3 package, hence ``inferencer_class``'s checks: score HPSv3 and HPSv3++ in separate processes.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import torch

from .common import local_config

ITER_STEP = 0.0


def inferencer_class(source: str | Path):
    """``(hpsv3 module, HPSv3RewardInferencer)`` of the HPSv3++ checkout ``source``; refuses the HPSv3 package."""
    import hpsv3
    from hpsv3.inference import HPSv3RewardInferencer

    origin = Path(getattr(hpsv3, "__file__", "") or "").resolve()
    source = Path(source).resolve()
    if source not in origin.parents:
        raise RuntimeError(f"`hpsv3` was imported from {origin}, not from the HPSv3++ checkout {source}: score "
                           "HPSv3++ in a process of its own (the HPSv3 package imports as `hpsv3` too)")
    if "iter_step" not in inspect.signature(HPSv3RewardInferencer.reward).parameters:
        raise RuntimeError(f"{origin}: reward() has no iter_step; this is HPSv3, not HPSv3++")
    return hpsv3, HPSv3RewardInferencer


class HPSv3PP:
    wants_paths = True      # the upstream interface takes image paths

    def __init__(self, checkpoint: str, base_model: str, device: str, source: str | Path):
        hpsv3, HPSv3RewardInferencer = inferencer_class(source)
        template = Path(hpsv3.__file__).parent / "config" / "train_stage2.yaml"
        with local_config(template, base_model) as config:
            self.inferencer = HPSv3RewardInferencer(config_path=str(config), checkpoint_path=checkpoint,
                                                    device=device)

    @torch.no_grad()
    def __call__(self, image_paths: list[str], prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        # keyword arguments: upstream's signature is reward(prompts, image_paths, iter_step)
        rewards = self.inferencer.reward(prompts=list(prompts), image_paths=list(image_paths), iter_step=ITER_STEP)
        rewards = rewards.float().cpu()
        return rewards[:, 0], rewards[:, 1]

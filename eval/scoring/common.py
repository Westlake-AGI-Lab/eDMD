"""Image conversions shared by the metrics, and the local copy of a reward model's YAML config.

The scorers receive a batch as a float32 tensor ``(N, 3, H, W)`` in ``[0, 1]`` (``pil_to_float_tensor``), the form
CDM's reward functions receive; each metric then converts it exactly as CDM does (to uint8, to PIL, or not at all).
Parts of this file are ported from CDM (MIT License, Copyright (c) 2024 FoundationVision); see eval/NOTICE.
"""

from __future__ import annotations

import contextlib
import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from PIL import Image


def pil_to_float_tensor(images: list[Image.Image]) -> torch.Tensor:
    """RGB images (one size) -> float32 ``(N, 3, H, W)``, ``uint8 / 255``."""
    array = np.stack([np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0 for image in images])
    return torch.from_numpy(array).permute(0, 3, 1, 2).contiguous()


def to_uint8(images: torch.Tensor) -> torch.Tensor:
    """float ``[0, 1]`` -> uint8: ``round(x * 255)``, clamped (CDM ``cdm/rewards/rewards.py``)."""
    return (images * 255).round().clamp(0, 255).to(torch.uint8)


def to_pil(images: torch.Tensor) -> list[Image.Image]:
    """float ``[0, 1]`` ``(N, 3, H, W)`` -> uint8 RGB PIL images (CDM ``cdm/rewards/rewards.py``)."""
    array = to_uint8(images).cpu().numpy().transpose(0, 2, 3, 1)
    return [Image.fromarray(item) for item in array]


@contextlib.contextmanager
def local_config(template: Path, base_model: str) -> Iterator[Path]:
    """A temporary copy of a reward model's YAML config whose ``model_name_or_path`` (a Hugging Face id upstream) is
    the local base-model directory; every other line is the upstream file's. Same file name as the template."""
    text = Path(template).read_text(encoding="utf-8")
    text, count = re.subn(r"^([ \t]*)model_name_or_path:.*$", lambda m: f"{m.group(1)}model_name_or_path: "
                          f"{json.dumps(str(base_model))}", text, flags=re.M)
    if count != 1:
        raise ValueError(f"{template}: expected one model_name_or_path line, found {count}")
    directory = Path(tempfile.mkdtemp(prefix="scorer_config_"))
    try:
        path = directory / Path(template).name
        path.write_text(text, encoding="utf-8")
        yield path
    finally:
        shutil.rmtree(directory, ignore_errors=True)

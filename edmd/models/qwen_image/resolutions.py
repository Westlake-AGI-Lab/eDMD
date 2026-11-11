"""Qwen-Image-2.1 generation sizes by area tier and aspect ratio.

Keys are the prompt enhancer's ``wh_ratio`` (``"w:h"``); values are ``(width, height)`` in pixels, the model card's
order (``Geometry`` takes ``(height, width)``).

* ``2k`` -- official: the model card's ``WH_RATIO_TO_SIZE``.
* ``1k`` -- derived, not official: the pipeline's ``calculate_dimensions(1024 * 1024, w / h)``.

Every size is a multiple of 32, so the latent sides are even and the token count is a multiple of 4.
"""

from __future__ import annotations

import math
import operator

from ..base import Geometry
from .schedule import PIXEL_MULTIPLE

#: The model card's ``WH_RATIO_TO_SIZE`` -- (width, height).
OFFICIAL_2K: dict[str, tuple[int, int]] = {
    "1:1": (2048, 2048), "4:3": (2400, 1792), "3:4": (1792, 2400),
    "3:2": (2528, 1696), "2:3": (1696, 2528), "16:9": (2752, 1536),
    "9:16": (1536, 2752),
}
#: diffusers ``calculate_dimensions(1024 * 1024, w / h)`` for every official ratio -- (width, height). Derived.
DERIVED_1K: dict[str, tuple[int, int]] = {
    "1:1": (1024, 1024), "4:3": (1184, 896), "3:4": (896, 1184),
    "3:2": (1248, 832), "2:3": (832, 1248), "16:9": (1376, 768),
    "9:16": (768, 1376),
}
TIERS: dict[str, dict[str, tuple[int, int]]] = {"1k": DERIVED_1K, "2k": OFFICIAL_2K}
TIER_SOURCES = {"1k": "derived: diffusers calculate_dimensions(1024*1024, w/h) (pipeline_qwenimage21.py)",
                "2k": "official: Qwen-Image-2.1 model card WH_RATIO_TO_SIZE"}


def calculate_dimensions(target_area: float, ratio: float) -> tuple[int, int]:
    """The pipeline's ``calculate_dimensions``: ``(width, height)``."""
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    return round(width / 32) * 32, round(height / 32) * 32


def geometry_table() -> dict[str, dict[str, Geometry]]:
    """``{tier: {"w:h": Geometry(height, width)}}``; every tier has the "1:1" fallback."""
    return {tier: {ratio: Geometry(height=height, width=width) for ratio, (width, height) in sizes.items()}
            for tier, sizes in TIERS.items()}


def check_geometry(geometry: Geometry) -> Geometry:
    """A geometry the layout accepts as is: positive multiples of 32 on both sides (a non-multiple is refused, not
    rounded down as by the pipeline, so the caller knows the size it gets)."""
    sides = {}
    for name in ("height", "width"):
        value = getattr(geometry, name)
        try:
            integer = None if isinstance(value, bool) else operator.index(value)   # ints, numpy ints; not floats
        except TypeError:
            integer = None
        if integer is None or integer <= 0 or integer % PIXEL_MULTIPLE:
            raise ValueError(f"Qwen-Image-2.1 geometry {name} must be a positive multiple of {PIXEL_MULTIPLE} pixels "
                             f"(latent side even, tokens a multiple of 4); got {geometry}.")
        sides[name] = int(integer)
    return Geometry(**sides)


__all__ = ["DERIVED_1K", "OFFICIAL_2K", "TIERS", "TIER_SOURCES", "calculate_dimensions", "check_geometry",
           "geometry_table"]

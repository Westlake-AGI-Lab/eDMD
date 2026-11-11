"""The geometry of each training batch under ``run.data.geometry: buckets`` (``fixed`` keeps the model's own).

An area tier per student step and an aspect ratio per prompt, both pure functions of the data: no RNG stream is
consumed, so resuming reproduces them exactly.

* tier(step) = the tier with the most pixels for the first ``largest_first`` steps (so an OOM shows in minutes),
  afterwards a weighted choice keyed by sha256(seed, step). Every rank uses the same tier, so FSDP2's per-block
  collectives do not wait on a straggler; the fake updates and the student update of a step share it.
* aspect(prompt) = its ``wh_ratio`` when the corpus carries one (blank -> 1:1, the official fallback), else a weighted
  choice keyed by sha256(prompt text) over ``aspect_weights`` (default: ``PE_ASPECT_WEIGHTS``).

The adapter maps (tier, ratio) through ``ModelAdapter.geometry_table``; a ratio missing from the tier is 1:1.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

#: wh_ratio histogram of the Qwen-Image-2.1 prompt enhancer's rewrites of the 2048 PickScore test prompts (the 5
#: blank ratios counted as 1:1): the aspect prior of a corpus without ``wh_ratio``.
PE_ASPECT_WEIGHTS: dict[str, float] = {"1:1": 734.0, "2:3": 664.0, "3:2": 593.0, "16:9": 36.0, "9:16": 14.0,
                                       "3:4": 4.0, "4:3": 3.0}


def _unit(*parts: object) -> float:
    """A uniform number in [0, 1) from sha256 of the parts (stable across processes and Python versions)."""
    digest = hashlib.sha256("\x1f".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2.0 ** 64


def _weighted(weights: Mapping[str, float], u: float) -> str:
    keys = sorted(key for key, value in weights.items() if float(value) > 0)
    if not keys:
        raise ValueError("no positive weight")
    total = sum(float(weights[key]) for key in keys)
    edge = 0.0
    for key in keys:
        edge += float(weights[key]) / total
        if u < edge:
            return key
    return keys[-1]


def tier_for_step(spec: Any, table: Mapping[str, Mapping[str, Any]], *, seed: int, step: int) -> str:
    unknown = sorted(set(spec.tiers) - set(table))
    if unknown:
        raise ValueError(f"data.geometry.tiers {unknown} are not in the adapter's table {sorted(table)}")
    if step < int(spec.largest_first):
        used = [tier for tier, weight in spec.tiers.items() if float(weight) > 0]
        return max(used, key=lambda tier: max(g.height * g.width for g in table[tier].values()))
    return _weighted(spec.tiers, _unit("tier", seed, step))


def aspect_for_prompt(prompt: str, metadata: Mapping[str, Any] | None, weights: Mapping[str, float]) -> str:
    if metadata and "wh_ratio" in metadata:
        ratio = metadata["wh_ratio"]
        return ratio.strip() if isinstance(ratio, str) and ratio.strip() else "1:1"
    return _weighted(weights or PE_ASPECT_WEIGHTS, _unit("aspect", prompt))


class GeometryPlan:
    """``geometry(step, prompts, metadata)`` for one run (``run.data.geometry`` of mode ``buckets``)."""

    def __init__(self, adapter: Any, spec: Any, *, seed: int) -> None:
        self.adapter, self.spec, self.seed = adapter, spec, int(seed)
        self.table = adapter.geometry_table()
        for tier in spec.tiers:
            if tier not in self.table:
                raise ValueError(f"data.geometry.tiers: {tier!r} is not one of {self.adapter.name}'s tiers "
                                 f"{sorted(self.table)}")
            if "1:1" not in self.table[tier]:
                raise ValueError(f"{self.adapter.name} tier {tier!r} has no 1:1 geometry (the fallback ratio)")

    def geometry(self, step: int, prompts: Sequence[str], metadata: Sequence[Mapping[str, Any]] | None) -> Any:
        if len(prompts) != 1:
            raise ValueError("geometry buckets need a per-rank batch of 1")
        tier = tier_for_step(self.spec, self.table, seed=self.seed, step=step)
        ratio = aspect_for_prompt(prompts[0], metadata[0] if metadata else None, self.spec.aspect_weights)
        return self.table[tier].get(ratio, self.table[tier]["1:1"])


__all__ = ["GeometryPlan", "PE_ASPECT_WEIGHTS", "aspect_for_prompt", "tier_for_step"]

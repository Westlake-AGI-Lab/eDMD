"""Check, build and run a scorer in fixed batches, aggregate: what ``eval/score_images.py`` and
``python -m edmd.cli.evaluate score`` (``edmd/evaluation/scoring.py``) call.

Batches are cut in sample order, ``batch_size`` images at a time (``BATCH_SIZES``; HPSv3 and HPSv3++ score
differently at other batch sizes); a batch whose images differ in size is split into runs of equal size, in order.
Everything runs in this interpreter: HPSv3++ needs its own environment (``eval/README.md``), and since it imports as
``hpsv3`` like the HPSv3 package, a process scores one of the two.
"""

from __future__ import annotations

import gc
import importlib
import sys
import time
from typing import Any, Sequence

import torch

try:  # registers the Ascend "npu" device when torch_npu is installed
    import torch_npu  # noqa: F401
except ImportError:
    pass

from . import data
from .common import pil_to_float_tensor
from .hpsv3pp import ITER_STEP

METRICS = data.METRICS
BATCH_SIZES = {"hpsv3": 1, "hpsv3pp": 1, "pickscore": 8, "aesthetic": 8, "hpsv2": 8, "imagereward": 8, "clip": 8}
#: Metrics returning ``{"mu", "sigma"}`` per image instead of a float.
UNCERTAIN = ("hpsv3", "hpsv3pp")
#: What building each metric's scorer imports besides its module (``module`` or ``module:name``), for ``check``.
REQUIRES = {
    "hpsv3": ("hpsv3:HPSv3RewardInferencer",),
    "hpsv3pp": (),                                    # hpsv3pp.inferencer_class checks the checkout
    "pickscore": ("transformers:AutoModel", "transformers:AutoProcessor"),
    "aesthetic": ("transformers:CLIPModel", "transformers:CLIPProcessor"),
    "hpsv2": ("hpsv2.src.open_clip:create_model", "hpsv2.src.open_clip:get_tokenizer"),
    "imagereward": ("ImageReward.ImageReward:ImageReward", "ImageReward.models.BLIP.blip_pretrain",
                    "transformers:BertTokenizer"),
    "clip": ("transformers:CLIPModel", "transformers:CLIPProcessor"),
}


def use_hpsv3pp_source() -> None:
    """Put the HPSv3++ checkout first on ``sys.path`` (it imports as ``hpsv3``, like the HPSv3 package)."""
    if not (data.HPSV3PP_SOURCE / "hpsv3" / "inference.py").is_file():
        raise FileNotFoundError(f"HPSv3++ needs its source checkout at {data.HPSV3PP_SOURCE} (see eval/README.md)")
    if str(data.HPSV3PP_SOURCE) not in sys.path:
        sys.path.insert(0, str(data.HPSV3PP_SOURCE))


def missing_weights(metric: str, weights: dict[str, str]) -> list[str]:
    """The files ``metric`` loads that ``weights`` (``data.resolve_weights``) lacks."""
    return [f"{repo}/{name or ''}" for key, (repo, name) in data.METRIC_WEIGHTS[metric].items() if key not in weights]


def check(metric: str) -> None:
    """Fail now, without loading a model, when ``metric`` cannot be scored here: its weights must be in the manifest
    and what its scorer imports must import in this interpreter."""
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}; choose from {METRICS}")
    missing = missing_weights(metric, data.resolve_weights(data.load_manifest()))
    if missing:
        raise FileNotFoundError(f"{metric} needs {missing}; not found through the weight manifest "
                                f"({data.manifest_path()}): run eval/download_weights.py --metrics {metric}")
    if metric == "hpsv3pp":
        use_hpsv3pp_source()
        from .hpsv3pp import inferencer_class

        inferencer_class(data.HPSV3PP_SOURCE)
    importlib.import_module(f".{metric}", __package__)
    for item in REQUIRES[metric]:
        module, _, name = item.partition(":")
        value = importlib.import_module(module)
        if name:
            getattr(value, name)


def build_scorer(metric: str, weights: dict[str, str], device: str):
    """``weights``: ``data.resolve_weights(data.load_manifest())``."""
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}; choose from {METRICS}")
    missing = missing_weights(metric, weights)
    if missing:
        raise FileNotFoundError(f"{metric} needs {missing}; not found through the weight manifest "
                                f"({data.manifest_path()}): run eval/download_weights.py")
    if metric == "hpsv3":
        from .hpsv3 import HPSv3

        return HPSv3(weights["hpsv3_ckpt"], weights["qwen2_vl"], device)
    if metric == "hpsv3pp":
        use_hpsv3pp_source()
        from .hpsv3pp import HPSv3PP

        return HPSv3PP(weights["hpsv3pp_ckpt"], weights["qwen3_vl"], device, source=data.HPSV3PP_SOURCE)
    if metric == "pickscore":
        from .pickscore import PickScore

        return PickScore(weights["pickscore"], weights["clip_h"], device)
    if metric == "aesthetic":
        from .aesthetic import Aesthetic

        return Aesthetic(weights["clip_l"], weights["aesthetic_mlp"], device)
    if metric == "imagereward":
        from .imagereward import ImageReward

        return ImageReward(weights["imagereward_ckpt"], weights["imagereward_med_config"], weights["bert"], device)
    if metric == "clip":
        from .clip import CLIPScore

        return CLIPScore(weights["clip_b"], device)
    from .hpsv2 import HPSv2

    return HPSv2(weights["hpsv2_ckpt"], weights["openclip_h"], device)


def score_shard(metric: str, scorer, images: data.LazyImages, prompts: Sequence[str], item_ids: Sequence[Any],
                batch_size: int, *, log_every: int = 20) -> list:
    """One value per image, in order: a float, or ``{"mu", "sigma"}`` for HPSv3 / HPSv3++. ``item_ids`` is unused
    (kept for the pipeline's call signature)."""
    count = len(images)
    values: list = []
    for start in range(0, count, batch_size):
        indices = list(range(start, min(start + batch_size, count)))
        batch_prompts = [prompts[index] for index in indices]
        if getattr(scorer, "wants_paths", False):
            outputs = [scorer([images.paths[index] for index in indices], batch_prompts)]
        else:
            loaded = [images[index] for index in indices]
            runs: list[list[int]] = []          # consecutive images of one size share a tensor
            for position, image in enumerate(loaded):
                if runs and loaded[runs[-1][0]].size == image.size:
                    runs[-1].append(position)
                else:
                    runs.append([position])
            outputs = [scorer(pil_to_float_tensor([loaded[p] for p in run]), [batch_prompts[p] for p in run])
                       for run in runs]
        for output in outputs:
            if metric in UNCERTAIN:
                mu, sigma = output
                values += [{"mu": float(a), "sigma": float(b)} for a, b in zip(mu, sigma)]
            else:
                values += [float(value) for value in output]
        if log_every and (start // batch_size) % log_every == 0:
            print(f"  [{metric}] {min(start + batch_size, count)}/{count}", flush=True)
    return values


def set_device(device: str) -> None:
    if device.startswith("npu"):
        torch.npu.set_device(device)
    elif device.startswith("cuda"):
        torch.cuda.set_device(device)


def score_paths(metric: str, paths: Sequence[str], prompts: Sequence[str], device: str) -> list:
    """Score the images at ``paths`` against ``prompts`` on ``device``, in ``BATCH_SIZES[metric]`` batches; the scorer
    is freed afterwards."""
    set_device(device)
    scorer = build_scorer(metric, data.resolve_weights(data.load_manifest()), device)
    started = time.time()
    values = score_shard(metric, scorer, data.LazyImages(list(paths)), list(prompts), [None] * len(paths),
                         BATCH_SIZES[metric])
    print(f"{metric}: {len(values)} images in {time.time() - started:.0f} s on {device}", flush=True)
    del scorer
    gc.collect()
    for backend in ("cuda", "npu"):
        if device.startswith(backend):
            getattr(torch, backend).empty_cache()
    return values


def aggregate(metric: str, values: list) -> dict:
    """``{"n", "mean"}``; HPSv3 / HPSv3++ add ``mean_sigma`` (and HPSv3++ its ``iter_step``)."""
    if not values:
        return {"n": 0}
    if metric in UNCERTAIN:
        mus = [value["mu"] for value in values]
        sigmas = [value["sigma"] for value in values]
        result = {"n": len(mus), "mean": sum(mus) / len(mus), "mean_sigma": sum(sigmas) / len(sigmas)}
        if metric == "hpsv3pp":
            result["iter_step"] = ITER_STEP
        return result
    return {"n": len(values), "mean": sum(values) / len(values)}

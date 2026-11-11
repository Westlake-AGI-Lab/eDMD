"""Inputs shared by the scorers: the weight manifest, the image list and the prompt list.

Weights: scorers load local files only. A manifest (JSON, written by ``eval/download_weights.py``) maps each
Hugging Face repository id to a local directory. It is read from ``$EDMD_SCORER_MANIFEST`` (alias
``$CDM_EVAL_WEIGHTS_MANIFEST``), else ``eval/weights/manifest.json``; the evaluation pipeline reads the same file.

Images and prompts: image ``i`` is scored against prompt row ``i``, an image's row being the last run of digits in its
file name (``prompt_00012.png``, ``12.webp`` and ``img-0012.jpg`` are all row 12); a directory must hold rows
``0..n-1``, each once. Prompt files: ``.txt``, one prompt per non-blank line, stripped; ``.jsonl``, one object per
line, its ``score`` field (the raw request in an eDMD evaluation's ``prompts.jsonl``), else its ``prompt`` field.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

EVAL_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_ENVS = ("EDMD_SCORER_MANIFEST", "CDM_EVAL_WEIGHTS_MANIFEST")
DEFAULT_MANIFEST = EVAL_ROOT / "weights" / "manifest.json"
#: The HPSv3++ source checkout (github.com/PlantPotatoOnMoon/HPSv3-PlusPlus); see eval/README.md.
HPSV3PP_SOURCE = EVAL_ROOT / "HPSv3PlusPlus"

# Hugging Face repositories: the manifest's keys.
HPSV3 = "MizzenAI/HPSv3"
QWEN2_VL = "Qwen/Qwen2-VL-7B-Instruct"
HPSV3PP = "Junjun2333/HPSv3-PlusPlus"
QWEN3_VL = "Qwen/Qwen3-VL-8B-Instruct"
PICKSCORE = "yuvalkirstain/PickScore_v1"
CLIP_H = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
CLIP_L = "openai/clip-vit-large-patch14"
AESTHETIC = "trl-lib/ddpo-aesthetic-predictor"
HPSV2 = "xswu/HPSv2"
IMAGEREWARD = "THUDM/ImageReward"
BERT = "google-bert/bert-base-uncased"
CLIP_B = "openai/clip-vit-base-patch32"

#: The metrics, in the pipeline's order (``edmd.evaluation.protocol.METRICS``).
METRICS = ("hpsv3", "pickscore", "aesthetic", "hpsv2", "hpsv3pp", "imagereward", "clip")
#: What each metric loads: ``resolve_weights`` key -> (repository, file inside it, or None for the directory).
METRIC_WEIGHTS = {
    "hpsv3": {"hpsv3_ckpt": (HPSV3, "HPSv3.safetensors"), "qwen2_vl": (QWEN2_VL, None)},
    "hpsv3pp": {"hpsv3pp_ckpt": (HPSV3PP, "hpsv3++.pth"), "qwen3_vl": (QWEN3_VL, None)},
    "pickscore": {"pickscore": (PICKSCORE, None), "clip_h": (CLIP_H, None)},
    "aesthetic": {"clip_l": (CLIP_L, None), "aesthetic_mlp": (AESTHETIC, "aesthetic-model.pth")},
    "hpsv2": {"hpsv2_ckpt": (HPSV2, "HPS_v2.1_compressed.pt"),
              "openclip_h": (CLIP_H, "open_clip_pytorch_model.bin")},
    "imagereward": {"imagereward_ckpt": (IMAGEREWARD, "ImageReward.pt"),
                    "imagereward_med_config": (IMAGEREWARD, "med_config.json"), "bert": (BERT, None)},
    "clip": {"clip_b": (CLIP_B, None)},
}
#: The repositories each metric reads.
METRIC_REPOS = {metric: tuple(dict.fromkeys(repo for repo, _ in files.values()))
                for metric, files in METRIC_WEIGHTS.items()}


# ============================================================================ weights
def manifest_path() -> Path:
    for name in MANIFEST_ENVS:
        value = os.environ.get(name, "").strip()
        if value:
            return Path(value).expanduser()
    return DEFAULT_MANIFEST


def load_manifest(path: str | Path | None = None) -> dict[str, str]:
    """``{repo id: local directory}``."""
    path = Path(path) if path else manifest_path()
    if not path.is_file():
        raise FileNotFoundError(f"scorer weight manifest not found: {path} (run eval/download_weights.py, or set "
                                f"${MANIFEST_ENVS[0]})")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"{path} must map repo ids to directories")
    return {str(key): str(value) for key, value in manifest.items()}


def resolve_weights(manifest: dict[str, str]) -> dict[str, str]:
    """The files and directories the scorers load (``METRIC_WEIGHTS``) that exist under the manifest's directories.
    A metric with a missing entry fails when its scorer is built (``run.build_scorer``)."""
    weights = {}
    for files in METRIC_WEIGHTS.values():
        for key, (repo, name) in files.items():
            directory = manifest.get(repo)
            if directory and os.path.isdir(directory):
                path = os.path.join(directory, name) if name else directory
                if os.path.exists(path):
                    weights[key] = path
    return weights


# ============================================================================ images and prompts
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")


def list_images(directory: str | Path) -> list[str]:
    """The images of a directory in row order (see the module docstring)."""
    rows: dict[int, str] = {}
    for path in sorted(Path(directory).iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        digits = re.findall(r"\d+", path.stem)
        if not digits:
            raise ValueError(f"{path}: the file name carries no row number")
        row = int(digits[-1])
        if row in rows:
            raise ValueError(f"{path} and {rows[row]} are both row {row}")
        rows[row] = str(path)
    if not rows:
        raise ValueError(f"no images ({', '.join(IMAGE_SUFFIXES)}) in {directory}")
    missing = sorted(set(range(len(rows))) - set(rows))
    if missing:
        raise ValueError(f"{directory}: rows must be 0..{len(rows) - 1}; missing {missing[:10]}")
    return [rows[row] for row in range(len(rows))]


def read_prompts(path: str | Path) -> list[str]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        prompts = []
        for number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            case = json.loads(line)
            prompt = case.get("score", case.get("prompt")) if isinstance(case, dict) else None
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"{path}:{number}: needs a nonempty 'score' or 'prompt' string")
            prompts.append(prompt)
        return prompts
    return [line.strip() for line in text.splitlines() if line.strip()]


class LazyImages:
    """A sequence of RGB images, each opened when accessed."""

    def __init__(self, paths: list[str]):
        self.paths = list(paths)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        from PIL import Image

        with Image.open(self.paths[index]) as image:
            return image.convert("RGB")

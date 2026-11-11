"""An evaluation's cases, read from its ``configs/eval`` file, and a checkpoint's inference record.

* **Cases**: image ``i`` is rendered from render prompt ``i`` (e.g. a prompt-enhancer rewrite) with seed ``seed + i``
  and scored against score prompt ``i`` (always the raw request); with ``geometry: aspect`` it is rendered at its
  render file's ``wh_ratio`` in the adapter's area tier.
* **Inference record** (``weights/inference.json``, written by the training loop with every checkpoint): adapter, the
  method's sampler, the exact float32 grid, resolution, noise dtype and adapter options. Sampling and evaluation read
  the sampler and grid from it, never from current defaults.

Standard library only at module level (the scorer environments import ``edmd.evaluation``).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

REPO = Path(__file__).resolve().parents[2]

#: every metric, in the order tables list them (``eval/scoring``)
METRICS = ("hpsv3", "pickscore", "aesthetic", "hpsv2", "hpsv3pp", "imagereward", "clip")
INFERENCE_FORMAT = "edmd_inference_v1"
INFERENCE_RECORD = "inference.json"
ROLES = ("student", "student_ema")
#: an image directory of ``edmd.cli.evaluate``: images/, these files, scores/<metric>.json
PROMPTS_FILE = "prompts.jsonl"
INFO_FILE = "info.json"
RESULT_FILE = "result.json"


def write_json(path: Path, value: Any) -> None:
    """Atomic JSON write (temporary file + rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve_path(value: str | Path, *, root: Path = REPO) -> Path:
    """A path from an eval config: ``~`` and ``$VAR`` expanded, a relative path taken from the repository root."""
    text = os.path.expanduser(os.path.expandvars(str(value)))
    if "$" in text:
        raise ValueError(f"{value!r}: environment variable not set (expanded to {text!r})")
    path = Path(text)
    return path if path.is_absolute() else (root / path)


# ============================================================================ cases
def read_prompt_file(path: Path) -> tuple[list[str], list[int | None]]:
    """``.txt``: one prompt per line, stripped, blank lines skipped (as CDM's ``TextPromptDataset``); ``.jsonl``: one
    object per line with ``prompt`` (verbatim) and an optional integer ``seed``. Returns the prompts and each line's
    seed (None when absent)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"evaluation prompt file not found: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        prompts: list[str] = []
        seeds: list[int | None] = []
        for number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            case = json.loads(line)
            prompt = case.get("prompt") if isinstance(case, dict) else None
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"{path}:{number}: every case needs a nonempty 'prompt' string")
            seed = case.get("seed")
            if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
                raise ValueError(f"{path}:{number}: 'seed' must be an integer")
            prompts.append(prompt)
            seeds.append(seed)
        return prompts, seeds
    if path.suffix not in (".txt", ""):
        raise ValueError(f"prompt files are .txt or .jsonl, got {path}")
    prompts = [line.strip() for line in text.splitlines() if line.strip()]
    return prompts, [None] * len(prompts)


def read_prompt_ratios(path: Path) -> list[str]:
    """Each case's ``wh_ratio`` from a ``.jsonl`` render file (blank or absent: 1:1, as in training)."""
    path = Path(path)
    if path.suffix != ".jsonl":
        raise ValueError(f"geometry=aspect reads each case's wh_ratio from a .jsonl render file, got {path}")
    ratios = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line).get("wh_ratio")
        ratios.append(value.strip() if isinstance(value, str) and value.strip() else "1:1")
    return ratios


@dataclass(frozen=True)
class Protocol:
    """An eval config's case lists."""

    name: str
    render_prompts: tuple[str, ...]
    score_prompts: tuple[str, ...]
    seed_base: int
    metrics: tuple[str, ...]
    extra_metrics: tuple[str, ...] = ()
    ratios: tuple[str, ...] | None = None      # geometry=aspect: each case's wh_ratio
    tier: str | None = None                    # geometry=aspect: the adapter's area tier

    @property
    def scored_metrics(self) -> tuple[str, ...]:
        """Every metric scored and reported: ``metrics``, then ``extra_metrics``."""
        return self.metrics + self.extra_metrics

    @property
    def count(self) -> int:
        return len(self.render_prompts)

    def seed(self, index: int) -> int:
        """Image ``index`` uses ``seed + index`` for its initial noise and for every fresh noise of a stochastic
        sampler (one generator per image)."""
        return self.seed_base + int(index)

    def cases(self) -> list[dict[str, Any]]:
        cases = [{"index": index, "seed": self.seed(index), "render": render, "score": score}
                 for index, (render, score) in enumerate(zip(self.render_prompts, self.score_prompts))]
        for case, ratio in zip(cases, self.ratios or ()):
            case["wh_ratio"] = ratio
        return cases


def load_protocol(spec: Any, *, root: Path = REPO) -> Protocol:
    """Read and check an ``EvalSpec``'s prompt lists."""
    metrics, extra = tuple(spec.metrics), tuple(spec.extra_metrics)
    if not metrics or len(set(metrics)) != len(metrics) or any(metric not in METRICS for metric in metrics):
        raise ValueError(f"eval.metrics must be distinct members of {METRICS}, got {list(metrics)}")
    if len(set(extra)) != len(extra) or any(metric not in METRICS for metric in extra) or set(extra) & set(metrics):
        raise ValueError(f"eval.extra_metrics must be distinct members of {METRICS} not in eval.metrics, "
                         f"got {list(extra)}")
    if spec.seed < 0:
        raise ValueError("eval.seed must be a non-negative integer")
    render_path = resolve_path(spec.prompt_file, root=root)
    render, render_seeds = read_prompt_file(render_path)
    if spec.score_prompt_file:
        score, score_seeds = read_prompt_file(resolve_path(spec.score_prompt_file, root=root))
    else:
        score, score_seeds = list(render), list(render_seeds)
    if not render:
        raise ValueError(f"{render_path} holds no prompts")
    if len(render) != len(score):
        raise ValueError(f"render ({len(render)}) and score ({len(score)}) prompt lists differ in length")
    for label, seeds in (("render", render_seeds), ("score", score_seeds)):
        for index, seed in enumerate(seeds):
            if seed is not None and seed != spec.seed + index:
                raise ValueError(f"{label} case {index} carries seed {seed}, but image i uses eval.seed + i = "
                                 f"{spec.seed + index}")
    ratios = tier = None
    if spec.geometry == "aspect":
        ratios, tier = tuple(read_prompt_ratios(render_path)), str(spec.geometry_tier)
        if len(ratios) != len(render):
            raise ValueError(f"{render_path}: {len(ratios)} ratios for {len(render)} prompts")
    return Protocol(name=str(spec.name), render_prompts=tuple(render), score_prompts=tuple(score),
                    seed_base=int(spec.seed), metrics=metrics, extra_metrics=extra, ratios=ratios, tier=tier)


def image_name(index: int) -> str:
    return f"images/prompt_{int(index):05d}.png"


def read_cases(directory: Path) -> list[dict[str, Any]]:
    """An image directory's ``prompts.jsonl``: one ``{"index", "seed", "render", "score", "height", "width"}`` object
    per image, in order."""
    path = Path(directory) / PROMPTS_FILE
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if [case.get("index") for case in cases] != list(range(len(cases))):
        raise ValueError(f"{path}: cases must be indexed 0..n-1 in order")
    return cases


# ============================================================================ inference record
def float32_hex(values: Iterable[float]) -> list[str]:
    """Exact float32 values as ``float.hex`` strings."""
    import struct

    return [float.hex(struct.unpack("f", struct.pack("f", float(value)))[0]) for value in values]


def build_inference_record(config: Any, adapter: Any, *, roles: Mapping[str, str]) -> dict[str, Any]:
    """What ``weights/inference.json`` holds. ``roles`` maps each renderable role (``student``, and ``student_ema``
    with EMA) to its train type (``lora`` / ``full``)."""
    method, model = config.method, config.model
    options = dict(getattr(adapter, "options", None) or model.options)   # resolved: the adapter's defaults included
    grid = adapter.student_grid(int(method.num_steps), str(method.schedule))
    values = [float(value) for value in grid.tolist()]
    geometry = adapter.geometry()
    unknown = sorted(set(roles) - set(ROLES))
    if unknown or not roles:
        raise ValueError(f"renderable roles are {ROLES}, got {sorted(roles)}")
    return {
        "format": INFERENCE_FORMAT,
        "adapter": str(model.name),
        "variant": str(model.variant),
        "method": str(method.name),
        "sampler": str(method.rollout.sampler),
        "num_steps": int(method.num_steps),
        "schedule": str(method.schedule),
        "shift_factor": float(adapter.shift_factor(int(method.num_steps), str(method.schedule))),
        "grid": values,
        "grid_float32_hex": float32_hex(values),
        "resolution": int(model.resolution),
        "height": int(geometry.height),
        "width": int(geometry.width),
        "cfg": 1.0,
        "noise_dtype": str(options.get("noise_dtype", "float32")),
        "sampling_state_dtype": str(adapter.sampling_state_dtype).replace("torch.", ""),
        "roles": {str(role): str(train) for role, train in roles.items()},
        "adapter_options": json.loads(json.dumps(options, sort_keys=True)),
    }


def validate_inference_record(record: Any) -> dict[str, Any]:
    """Check a record and return it normalized: ``grid`` from the exact hex values, defaults for the optional keys.
    Keys of older records (``base``) are ignored."""
    if not isinstance(record, Mapping) or record.get("format") != INFERENCE_FORMAT:
        raise ValueError(f"not an {INFERENCE_FORMAT} record")
    required = ("adapter", "variant", "sampler", "num_steps", "schedule", "grid_float32_hex", "height", "width",
                "roles")
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError(f"inference record lacks {missing}")
    if record["sampler"] not in ("euler", "x0_renoise"):
        raise ValueError(f"unknown sampler {record['sampler']!r}")
    grid = [float.fromhex(value) for value in record["grid_float32_hex"]]
    if (len(grid) != int(record["num_steps"]) + 1 or grid[-1] != 0.0 or grid[0] > 1.0
            or any(a <= b for a, b in zip(grid, grid[1:]))):
        raise ValueError(f"inference grid must descend strictly from <= 1 to 0 over num_steps: {grid}")
    if "grid" in record and [float(value) for value in record["grid"]] != grid:
        raise ValueError("inference record: grid and grid_float32_hex disagree")
    if float(record.get("cfg", 1.0)) != 1.0:
        raise ValueError("students sample without guidance (cfg 1)")
    if not isinstance(record["roles"], Mapping) or not record["roles"]:
        raise ValueError("inference record: roles must map role -> train type")
    options = record.get("adapter_options", {})
    if not isinstance(options, Mapping):
        raise ValueError("inference record: adapter_options must be a mapping")
    normalized = dict(record)
    normalized.update(grid=grid, cfg=1.0, noise_dtype=str(record.get("noise_dtype", "float32")),
                      resolution=None if record.get("resolution") is None else int(record["resolution"]),
                      sampling_state_dtype=str(record.get("sampling_state_dtype", "float32")),
                      adapter_options=dict(options))
    return normalized


def weights_dir(checkpoint: Path) -> Path:
    """Where a checkpoint's weights and ``inference.json`` live: ``<checkpoint>/weights``, or ``<checkpoint>`` itself
    for released weights (``<role>.safetensors``, ``<role>.json``, ``inference.json``)."""
    checkpoint = Path(checkpoint)
    flat = (checkpoint / INFERENCE_RECORD).is_file() and not (checkpoint / "weights" / INFERENCE_RECORD).is_file()
    return checkpoint if flat else checkpoint / "weights"


def safetensors_metadata(path: Path) -> dict[str, str]:
    """The ``__metadata__`` of a safetensors file, read from its JSON header (no tensor is loaded)."""
    import struct

    with open(path, "rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        return dict(json.loads(handle.read(length)).get("__metadata__") or {})


def release_info(checkpoint: Path) -> dict[str, Any] | None:
    """A released file (``edmd_<model>_lora.safetensors``, written by ``scripts/convert_lora.py``): a diffusers LoRA
    whose metadata ``edmd`` holds the role, its weights record and the inference record. None for a checkpoint
    directory."""
    path = Path(checkpoint)
    if not (path.is_file() and path.suffix == ".safetensors"):
        return None
    info = safetensors_metadata(path).get("edmd")
    if info is None:
        raise ValueError(f"{path} is a LoRA without eDMD's sampling record; write it with scripts/convert_lora.py")
    return json.loads(info)


def read_inference_record(checkpoint: Path) -> dict[str, Any]:
    release = release_info(checkpoint)
    if release is not None:
        return validate_inference_record(release["inference"])
    path = weights_dir(checkpoint) / INFERENCE_RECORD
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing: the training loop writes it next to the weights at every "
                                "checkpoint; sampling never guesses the sampler or grid")
    return validate_inference_record(read_json(path))


def weights_file(checkpoint: Path, role: str) -> Path:
    """The weights of ``role``: a released file itself, else ``<role>.safetensors`` of a checkpoint (or of an older
    released directory)."""
    if release_info(checkpoint) is not None:
        return Path(checkpoint)
    return weights_dir(checkpoint) / f"{role}.safetensors"


def default_role(record: Mapping[str, Any]) -> str:
    """``student`` (the raw weights) when the record has it, else its only role (a release of the EMA weights)."""
    roles = list(record["roles"])
    return "student" if "student" in roles else roles[0]


__all__ = ["INFERENCE_FORMAT", "INFERENCE_RECORD", "INFO_FILE", "METRICS", "PROMPTS_FILE",
           "Protocol", "REPO", "RESULT_FILE", "ROLES", "build_inference_record", "default_role", "float32_hex",
           "image_name", "load_protocol", "read_cases", "read_inference_record", "read_json", "read_prompt_file",
           "read_prompt_ratios", "release_info", "resolve_path", "safetensors_metadata", "validate_inference_record",
           "weights_dir", "weights_file", "write_json"]

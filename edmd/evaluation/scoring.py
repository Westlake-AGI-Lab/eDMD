"""Score the image directories of ``edmd.evaluation.generate`` in this interpreter, one process per device.

Per directory: ``scores/<metric>.json`` (every image's value and the aggregate) and ``result.json`` (``label``,
``step``, ``num_images`` and every scored metric's aggregate under ``results``, rewritten after each metric). A metric
already scored is kept unless ``force``. The scorers are ``eval/scoring`` (``eval/README.md``), with their fixed
batch sizes; several devices receive whole batches round-robin and the values are merged in image order, so no
number depends on the device count. HPSv3++ needs its own environment: score it in a separate call there.

Standard library only at module level (the scorer environments import this module).
"""

from __future__ import annotations

import math
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence

from .parallel import batch_shard_indices, plan_devices, run_on_devices
from .protocol import (INFO_FILE, METRICS, PROMPTS_FILE, REPO, RESULT_FILE, image_name, read_cases, read_json,
                       write_json)

EVAL_ROOT = REPO / "eval"
OFFLINE = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}


def scorer_package() -> tuple[Any, Any]:
    """``(scoring.data, scoring.run)`` of ``eval/scoring``, with the Hugging Face offline switches set (every weight
    comes from the manifest)."""
    if str(EVAL_ROOT) not in sys.path:
        sys.path.insert(0, str(EVAL_ROOT))
    os.environ.update(OFFLINE)
    from scoring import data, run

    return data, run


def natural_key(text: str) -> list:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", str(text))]


def image_dirs(roots: Iterable[Path]) -> list[Path]:
    """Every image directory (one with ``prompts.jsonl``) at or below ``roots``."""
    found: list[Path] = []
    for root in map(Path, roots):
        if (root / PROMPTS_FILE).is_file():
            found.append(root)
        else:
            found += sorted((path.parent for path in root.rglob(PROMPTS_FILE)), key=lambda path: natural_key(path))
    return found


def recorded_metrics(directory: Path) -> list[str]:
    """The metrics of the eval config ``generate`` recorded in ``info.json`` (``metrics``, then ``extra_metrics``)."""
    info = read_json(Path(directory) / INFO_FILE) if (Path(directory) / INFO_FILE).is_file() else {}
    config = info.get("eval") or {}
    return list(config.get("metrics", [])) + list(config.get("extra_metrics", []))


def check(metrics: Sequence[str]) -> None:
    """Fail now, with one line, when a metric cannot be scored in this interpreter (weights manifest and imports; no
    model is loaded)."""
    _, run = scorer_package()
    for metric in metrics:
        try:
            run.check(metric)
        except Exception as error:  # noqa: BLE001 - whatever fails, the metric cannot be scored here
            raise SystemExit(f"{metric} cannot be scored in {sys.executable}: "
                             f"{type(error).__name__}: {error}") from None


def _score_shard(device: str, metric: str, paths: Sequence[str], prompts: Sequence[str], out_file: str) -> None:
    _, run = scorer_package()
    write_json(Path(out_file), run.score_paths(metric, paths, prompts, device))


def score_metric(metric: str, paths: Sequence[str], prompts: Sequence[str], devices: Sequence[str]) -> list:
    """Every image's value, in order (shards of whole batches when there are several devices)."""
    _, run = scorer_package()
    batch = run.BATCH_SIZES[metric]
    devices = list(devices)[:math.ceil(len(paths) / batch)]
    if len(devices) == 1:
        return run.score_paths(metric, paths, prompts, devices[0])
    shards = [batch_shard_indices(len(paths), batch, rank, len(devices)) for rank in range(len(devices))]
    values: list = [None] * len(paths)
    with tempfile.TemporaryDirectory(prefix=f"edmd_{metric}_") as scratch:
        outputs = [str(Path(scratch) / f"shard_{rank}.json") for rank in range(len(devices))]
        run_on_devices(_score_shard, devices, [(metric, [paths[i] for i in shard], [prompts[i] for i in shard], out)
                                               for shard, out in zip(shards, outputs)])
        for shard, out in zip(shards, outputs):
            for index, value in zip(shard, read_json(Path(out))):
                values[index] = value
    return values


def write_result(directory: Path) -> dict[str, Any]:
    """``result.json`` from the directory's ``scores/``."""
    directory = Path(directory)
    info = read_json(directory / INFO_FILE) if (directory / INFO_FILE).is_file() else {}
    results = {metric: read_json(directory / "scores" / f"{metric}.json")["aggregate"] for metric in METRICS
               if (directory / "scores" / f"{metric}.json").is_file()}
    result = {"label": info.get("label", directory.name), "step": info.get("step"),
              "num_images": len(read_cases(directory)), "results": results}
    write_json(directory / RESULT_FILE, result)
    return result


def score_directory(directory: Path, metrics: Sequence[str], *, devices: Sequence[str] | None = None,
                    force: bool = False) -> dict[str, Any]:
    """Score one image directory with ``metrics``; returns its ``result.json``."""
    _, run = scorer_package()
    directory = Path(directory)
    cases = read_cases(directory)
    paths = [str(directory / image_name(case["index"])) for case in cases]
    missing = [path for path in paths if not Path(path).is_file()]
    if missing:
        raise SystemExit(f"{directory}: {len(missing)} of {len(paths)} images are missing (e.g. {missing[0]}); "
                         "render them with `edmd.cli.evaluate generate` first")
    prompts = [case["score"] for case in cases]          # the raw request, never the rewrite
    devices = plan_devices(devices)
    for metric in metrics:
        path = directory / "scores" / f"{metric}.json"
        if path.is_file() and not force:
            print(f"{directory}: {metric} already scored", flush=True)
            continue
        values = score_metric(metric, paths, prompts, devices)
        scalars = [value["mu"] if isinstance(value, dict) else value for value in values]
        if len(values) != len(paths) or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in scalars):
            raise RuntimeError(f"{directory}: {metric} returned missing or non-finite scores")
        write_json(path, {"metric": metric, "batch_size": run.BATCH_SIZES[metric],
                          "aggregate": run.aggregate(metric, values), "per_image": values})
        print(f"{directory}: {metric} {run.aggregate(metric, values)['mean']:.6f}", flush=True)
        write_result(directory)
    return write_result(directory)


def missing_metrics(directory: Path, metrics: Sequence[str]) -> list[str]:
    """The ``metrics`` without a score of every image in ``directory/scores/`` (all of them before ``generate``)."""
    directory = Path(directory)
    if not (directory / PROMPTS_FILE).is_file():
        return list(metrics)
    count = len(read_cases(directory))
    paths = {metric: directory / "scores" / f"{metric}.json" for metric in metrics}
    return [metric for metric, path in paths.items()
            if not path.is_file() or read_json(path).get("aggregate", {}).get("n") != count]


# ============================================================================ summary
def find_results(roots: Sequence[Path]) -> list[tuple[str, dict[str, Any]]]:
    """``(label, result)`` of every ``result.json`` at or below ``roots``: the directory's path relative to its root
    (prefixed with the root's name when there are several roots)."""
    rows = []
    for root in map(Path, roots):
        files = [root / RESULT_FILE] if (root / RESULT_FILE).is_file() else list(root.rglob(RESULT_FILE))
        for path in sorted(files, key=lambda item: natural_key(item.parent.relative_to(root).as_posix())):
            relative = path.parent.relative_to(root).as_posix()
            label = root.name if relative == "." else relative
            if len(roots) > 1 and relative != ".":
                label = f"{root.name}/{relative}"
            rows.append((label, read_json(path)))
    return rows


def summary_table(rows: Sequence[tuple[str, dict[str, Any]]]) -> str:
    """A markdown table: one row per result, one column per metric mean (``METRICS`` order)."""
    metrics = [metric for metric in METRICS if any(metric in (result.get("results") or {}) for _, result in rows)]
    lines = ["| " + " | ".join(["label", "images", *metrics]) + " |",
             "|" + "---|" * (len(metrics) + 2)]
    for label, result in rows:
        means = [(result.get("results") or {}).get(metric, {}).get("mean") for metric in metrics]
        cells = ["–" if mean is None else f"{mean:.4f}" for mean in means]
        lines.append("| " + " | ".join([label, str(result.get("num_images", "")), *cells]) + " |")
    return "\n".join(lines)


__all__ = ["EVAL_ROOT", "check", "find_results", "image_dirs", "missing_metrics", "natural_key", "recorded_metrics",
           "score_directory", "score_metric", "scorer_package", "summary_table", "write_result"]

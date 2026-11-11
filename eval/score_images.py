#!/usr/bin/env python3
"""Score a directory of images with the eDMD paper's preference metrics.

    python eval/score_images.py --images <dir> --prompts <prompts.txt | prompts.jsonl> [--metrics ...] [--out F]

Image ``i`` is scored against prompt row ``i``: an image's row is the last number in its file name
(``prompt_00012.png`` -> row 12, the layout of ``python -m edmd.cli.evaluate generate``), and the directory must hold
rows ``0..n-1`` for the ``n`` prompts. ``.txt`` prompt files hold one prompt per non-blank line; ``.jsonl`` files one
object per line with ``score`` (the ``prompts.jsonl`` of ``edmd.cli.evaluate generate``) or ``prompt``.

Weights come from the manifest written by ``eval/download_weights.py`` (``$EDMD_SCORER_MANIFEST``, else
``eval/weights/manifest.json``); nothing is fetched from the Hugging Face Hub. Every metric runs in this interpreter:
by default all but HPSv3++, which needs its own environment -- run the script again there with ``--metrics hpsv3pp``
and the same ``--out``.

Prints each metric's mean; ``--out`` also writes every image's scores (for HPSv3 / HPSv3++ the score is ``mu``, with
``<metric>_sigma`` next to it). The file is rewritten after every metric and keeps the metrics an earlier run wrote
there for the same images and prompts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from scoring import data  # noqa: E402  (standard library only)

OFFLINE = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}


def default_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda:0"
    try:
        import torch_npu  # noqa: F401

        if torch.npu.is_available():
            return "npu:0"
    except ImportError:
        pass
    return "cpu"


def earlier_scores(args: argparse.Namespace, paths: list[str], prompts: list[str]) -> dict[str, list]:
    """The per-image values an earlier run wrote to ``--out`` for the same images and prompts."""
    if not args.out or not args.out.is_file():
        return {}
    earlier = json.loads(args.out.read_text(encoding="utf-8"))
    rows = earlier.get("per_image", [])
    if ([row.get("image") for row in rows] != [os.path.basename(path) for path in paths]
            or [row.get("prompt") for row in rows] != prompts):
        sys.exit(f"{args.out} holds the scores of other images or prompts; choose another --out")
    values = {}
    for metric in earlier.get("metrics", {}):
        if metric in data.METRICS:
            values[metric] = [{"mu": row[metric], "sigma": row[f"{metric}_sigma"]} if f"{metric}_sigma" in row
                              else row[metric] for row in rows]
    return values


def write_scores(args: argparse.Namespace, paths: list[str], prompts: list[str], device: str,
                 values: dict[str, list]) -> None:
    """``--out``: the means and every image's scores of the metrics scored so far, in ``data.METRICS`` order."""
    from scoring.run import aggregate

    done = [metric for metric in data.METRICS if metric in values]
    rows = []
    for index, (path, prompt) in enumerate(zip(paths, prompts)):
        row = {"index": index, "image": os.path.basename(path), "prompt": prompt}
        for metric in done:
            value = values[metric][index]
            if isinstance(value, dict):
                row[metric], row[f"{metric}_sigma"] = value["mu"], value["sigma"]
            else:
                row[metric] = value
        rows.append(row)
    output = {"images": str(args.images.resolve()), "prompts": str(args.prompts.resolve()), "device": device,
              "metrics": {metric: aggregate(metric, values[metric]) for metric in done}, "per_image": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(args.out.name + ".tmp")
    temporary.write_text(json.dumps(output, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    temporary.replace(args.out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", type=Path, required=True, help="image directory (row = last number in a name)")
    parser.add_argument("--prompts", type=Path, required=True, help=".txt (one per line) or .jsonl (score / prompt)")
    parser.add_argument("--metrics", nargs="+", choices=data.METRICS,
                        default=[metric for metric in data.METRICS if metric != "hpsv3pp"],
                        help="default: every metric but hpsv3pp (score it in its own environment)")
    parser.add_argument("--out", type=Path, help="write per-image scores and means here (JSON)")
    parser.add_argument("--device", help="cuda:0, npu:0 or cpu (default: the first accelerator)")
    args = parser.parse_args()
    os.environ.update(OFFLINE)
    metrics = list(dict.fromkeys(args.metrics))
    if {"hpsv3", "hpsv3pp"} <= set(metrics):
        sys.exit("HPSv3 and HPSv3++ both import as `hpsv3` and need different environments: score hpsv3pp in a "
                 "separate run, in its own environment (eval/README.md)")

    from scoring import run

    paths, prompts = data.list_images(args.images), data.read_prompts(args.prompts)
    if len(paths) != len(prompts):
        sys.exit(f"{len(paths)} images in {args.images} but {len(prompts)} prompts in {args.prompts}: image i is "
                 "scored against prompt row i")
    for metric in metrics:      # fail before the first model loads
        try:
            run.check(metric)
        except Exception as error:  # noqa: BLE001 - whatever fails, the metric cannot be scored here
            sys.exit(f"{metric} cannot be scored in {sys.executable}: {type(error).__name__}: {error}")
    device = args.device or default_device()
    values = earlier_scores(args, paths, prompts)
    for metric in metrics:
        values[metric] = run.score_paths(metric, paths, prompts, device)
        if args.out:
            write_scores(args, paths, prompts, device, values)
    for metric in metrics:
        summary = run.aggregate(metric, values[metric])
        print(f"{metric:<12} mean {summary['mean']:.6f}  (n={summary['n']})")
    if args.out:
        print(f"per-image scores -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

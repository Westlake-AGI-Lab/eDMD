"""``python -m edmd.cli.evaluate`` -- render images, score them, tabulate, publish.

Two steps, each in its own environment:

    # 1. images, in the model's environment (one process per device)
    python -m edmd.cli.evaluate generate --checkpoint outputs/<run>/checkpoint-1000 [more | --run outputs/<run>] \\
        --eval drawbench1k --out outputs/eval/<run> [--role student_ema] [--devices cuda:0 cuda:1 ...]
    python -m edmd.cli.evaluate generate --teacher --config configs/runs/<model>/<method>.yaml --steps 50 --cfg 4 \\
        --eval drawbench1k --out outputs/eval/teacher
    # 2. scores, in the reward models' environment (HPSv3++: its own environment, --metrics hpsv3pp)
    python -m edmd.cli.evaluate score --dir outputs/eval/<run> [--metrics ...] [--devices ...]
    python -m edmd.cli.evaluate summary --dir outputs/eval/<run> [outputs/eval/teacher ...]

``generate`` writes ``<out>/<label>/`` per checkpoint (label: the checkpoint's directory name, ``-ema`` added for
``--role student_ema``; ``--label`` for a single source): ``images/``, ``prompts.jsonl`` and ``info.json``. It keeps
the images already there (a rerun renders the missing ones) unless ``--force``. ``--checkpoint`` also takes released
files (``edmd_<model>_lora.safetensors``, from ``scripts/convert_lora.py``; labelled by their name). ``--teacher`` renders the
base model (or ``--weights``) with ``--steps``-step Euler at CFG ``--cfg`` on the pipeline's grid; ``--sigmas`` or
``--scheduler-config`` render a released checkpoint with its own sampler at guidance 1, e.g. CDM's SD3 turbo weights
(``--weights <dir> --sigmas 1,0.75,0.5,0.25``) or TDM's SD3 LoRA (``--scheduler-config <scheduler_config.json>
--scheduler-set flow_shift=6 --lora <file> --lora-weight 0.125``).

``score`` scores every image directory at or below ``--dir`` in this interpreter: ``scores/<metric>.json`` and
``result.json``; metrics already scored are kept unless ``--force``. Without ``--metrics`` it scores the metrics of
the eval config ``generate`` recorded, except HPSv3++, which runs only when named (in its environment).
``summary`` prints a markdown table of the ``result.json`` files below ``--dir``.

``publish --run <dir>`` syncs a run's periodic-evaluation results (``<dir>/eval/<eval name>/``, the layout of
``edmd.cli.coordinate``) to its W&B run; ``register-preset --entity E`` registers the W&B image gallery's chart preset.
This module imports nothing heavy at import time.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence


def _eval_spec(ref: str):
    from ..config.core import from_mapping, load_yaml
    from ..config.load import component_path
    from ..config.schema import EvalSpec

    return from_mapping(EvalSpec, load_yaml(component_path("eval", ref)), f"eval {ref}")


def official_recipe_from_args(args: argparse.Namespace) -> dict[str, Any] | None:
    """The released-checkpoint sampler of ``--teacher`` (``edmd.evaluation.official``), or None without its options."""
    given = (args.sigmas is not None, args.scheduler_config is not None, bool(args.scheduler_set),
             args.lora is not None, args.lora_weight is not None)
    if not any(given):
        return None
    from ..evaluation import official

    if args.scheduler_set and args.scheduler_config is None:
        raise SystemExit("--scheduler-set overrides keys of --scheduler-config")
    if (args.lora is None) != (args.lora_weight is None):
        raise SystemExit("--lora and --lora-weight go together (load_lora_weights + set_adapters([name], [weight]))")
    scheduler = (official.load_scheduler_config(args.scheduler_config, official.parse_overrides(args.scheduler_set))
                 if args.scheduler_config is not None else None)
    lora = official.lora_spec(args.lora, args.lora_weight) if args.lora is not None else None
    return official.official_recipe(steps=args.steps, sigmas=args.sigmas, scheduler=scheduler, lora=lora)


def checkpoint_sources(args: argparse.Namespace) -> list[Path]:
    from ..runtime.checkpoint import complete_checkpoints

    if args.run is not None:
        checkpoints = [path.resolve() for path in complete_checkpoints(args.run).values()]
        if not checkpoints:
            raise SystemExit(f"{args.run} holds no complete checkpoint")
        return checkpoints
    return [Path(path).resolve() for path in args.checkpoint]


def checkpoint_job(checkpoint: Path, args: argparse.Namespace) -> tuple[dict, dict, dict]:
    """``(job, source, info)`` of one checkpoint (``edmd.evaluation.generate.generate``)."""
    from ..config.core import to_mapping
    from ..config.load import checkpoint_model
    from ..evaluation.protocol import default_role, read_inference_record, read_json, release_info, weights_file
    from ..runtime.checkpoint import COMPLETE_MARKER

    record = read_inference_record(checkpoint)
    role = args.role or default_role(record)
    if role not in record["roles"]:
        raise SystemExit(f"{checkpoint} has no {role!r} weights (roles: {sorted(record['roles'])})")
    weights = weights_file(checkpoint, role)
    if not weights.is_file():
        raise SystemExit(f"{weights} is missing")
    run_dir = checkpoint.parent if (checkpoint.parent / "config.yaml").is_file() else None
    model = checkpoint_model(record, config=args.config, run_dir=run_dir, model_path=args.model_path)
    marker, sidecar = checkpoint / COMPLETE_MARKER, weights.with_suffix(".json")
    release = release_info(checkpoint)
    step = (int(marker.read_text().strip()) if marker.is_file() else release["weights"].get("step") if release
            else read_json(sidecar).get("step") if sidecar.is_file() else None)
    grid = {"kind": "recorded", "float32_hex": list(record["grid_float32_hex"]), "num_steps": int(record["num_steps"]),
            "schedule": record["schedule"]}
    job = {"model": to_mapping(model), "weights": str(weights), "role_train": record["roles"][role],
           "sampler": record["sampler"], "grid": grid, "cfg": 1.0,
           "size": [int(record["height"]), int(record["width"])]}
    info = {"step": step, "sampler": record["sampler"], "num_steps": int(record["num_steps"]), "cfg": 1.0,
            "grid": record["grid"], "model": f"{model.name}/{model.variant}"}
    return job, {"checkpoint": str(checkpoint), "role": role}, info


def teacher_job(args: argparse.Namespace, protocol: Any) -> tuple[dict, dict, dict]:
    """``(job, source, info)`` of a teacher row (``--teacher``)."""
    from ..config.core import to_mapping
    from ..config.load import compose

    if args.config is None or args.steps is None or args.cfg is None:
        raise SystemExit("--teacher needs --config (a run file or a run's config.yaml), --steps and --cfg")
    if args.steps < 1 or args.cfg < 1.0:
        raise SystemExit("--steps must be >= 1 and --cfg >= 1")
    model = compose(args.config)[0].model
    if args.model_path is not None:
        from ..config.core import from_mapping
        from ..config.schema import ModelSpec

        model = from_mapping(ModelSpec, dict(to_mapping(model), path=str(Path(args.model_path).absolute())), "model")
    weights = str(Path(args.weights).absolute()) if args.weights is not None else None
    recipe = official_recipe_from_args(args)
    if recipe is not None:
        if args.cfg != 1.0:
            raise SystemExit("released checkpoints sample guidance-free (the TDM / CDM recipes use guidance 1): "
                             "--cfg 1")
        if protocol.tier is not None:
            raise SystemExit("released checkpoints render square evaluations only")
    job = {"model": to_mapping(model), "weights": weights, "role_train": "full", "sampler": "euler",
           "grid": {"kind": "pipeline", "num_steps": int(args.steps)}, "cfg": float(args.cfg), "official": recipe}
    source = {"teacher": f"{model.name}/{model.variant}", "steps": int(args.steps), "cfg": float(args.cfg),
              "weights": weights, "official": recipe}
    info = {"step": None, "sampler": recipe["sampler"] if recipe else "euler", "num_steps": int(args.steps),
            "cfg": float(args.cfg), "model": f"{model.name}/{model.variant}"}
    return job, source, info


def cmd_generate(args: argparse.Namespace) -> int:
    from ..config.core import to_mapping
    from ..evaluation.generate import generate
    from ..evaluation.protocol import load_protocol

    spec = _eval_spec(args.eval)
    protocol = load_protocol(spec)
    if args.teacher:
        job, source, info = teacher_job(args, protocol)
        jobs = [(args.label or f"teacher-{args.steps}step-cfg{args.cfg:g}", job, source, info)]
    else:
        checkpoints = checkpoint_sources(args)
        if args.label and len(checkpoints) > 1:
            raise SystemExit("--label names one source; leave it out for several checkpoints")
        jobs = []
        for checkpoint in checkpoints:
            job, source, info = checkpoint_job(checkpoint, args)
            suffix = "-ema" if source["role"] == "student_ema" else ""
            name = checkpoint.stem if checkpoint.is_file() else checkpoint.name
            jobs.append((args.label or f"{name}{suffix}", job, source, info))
    for label, job, source, info in jobs:
        directory = generate(Path(args.out) / label, job=job, source=source, protocol=protocol,
                             info={"label": label, **info, "eval": to_mapping(spec)}, devices=args.devices,
                             force=args.force)
        print(directory, flush=True)
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    from ..evaluation import scoring

    if args.metrics and {"hpsv3", "hpsv3pp"} <= set(args.metrics):
        raise SystemExit("hpsv3 and hpsv3pp both import as `hpsv3` and need different environments: score hpsv3pp "
                         "in a separate call, in its own environment (eval/README.md)")
    if args.check:
        if not args.metrics:
            raise SystemExit("--check needs --metrics")
        scoring.check(args.metrics)
        print(f"{' '.join(args.metrics)}: can be scored in {sys.executable}")
        return 0
    if not args.dir:
        raise SystemExit("score needs --dir")
    directories = scoring.image_dirs(args.dir)
    if not directories:
        raise SystemExit(f"no image directory (with prompts.jsonl) at or below {', '.join(map(str, args.dir))}")
    plans = []
    for directory in directories:
        recorded = scoring.recorded_metrics(directory)
        metrics = list(args.metrics or [metric for metric in recorded if metric != "hpsv3pp"])
        if not metrics:
            raise SystemExit(f"{directory}: nothing to score by default (info.json records {recorded}; hpsv3pp is "
                             "scored only when named): pass --metrics")
        plans.append((directory, metrics))
    scoring.check(sorted({metric for _, metrics in plans for metric in metrics}))   # before the first model loads
    for directory, metrics in plans:
        result = scoring.score_directory(directory, metrics, devices=args.devices, force=args.force)
        print(json.dumps({"dir": str(directory), **{metric: value["mean"] for metric, value in
                                                    result["results"].items()}}), flush=True)
    return 0


def cmd_summary(args: argparse.Namespace) -> int:
    from ..evaluation.scoring import find_results, summary_table

    rows = find_results([Path(path) for path in args.dir])
    if not rows:
        raise SystemExit(f"no result.json at or below {', '.join(args.dir)}: score the images first")
    print(summary_table(rows))
    return 0


def cmd_publish(args: argparse.Namespace) -> int:
    from ..tracking.wandb import publish_evaluations

    print(json.dumps(publish_evaluations(Path(args.run).resolve()), indent=2))
    return 0


def cmd_register(args: argparse.Namespace) -> int:
    import os

    from ..tracking.media import register_preset

    if args.base_url:
        os.environ["WANDB_BASE_URL"] = args.base_url
    print(register_preset(args.entity, storage_host=args.storage_host))
    return 0


def build_parser() -> argparse.ArgumentParser:
    from ..evaluation.protocol import METRICS

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)

    generate = sub.add_parser("generate", help="render an evaluation's images (the model's environment)")
    source = generate.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path, nargs="+", help="checkpoint-<N> directories or released files")
    source.add_argument("--run", type=Path, help="every complete checkpoint of this run directory")
    source.add_argument("--teacher", action="store_true", help="the base model (with --config, --steps, --cfg)")
    generate.add_argument("--eval", required=True, help="configs/eval/<name>.yaml: a name or a path")
    generate.add_argument("--out", type=Path, required=True, help="images go to <out>/<label>/")
    generate.add_argument("--role", choices=("student", "student_ema"),
                          help="checkpoints: the raw or the EMA weights (default: student, or the only role the "
                               "weights have)")
    generate.add_argument("--label", help="the directory name of a single source (default: the checkpoint's name, "
                                          "-ema added for student_ema; teacher-<steps>step-cfg<cfg>)")
    generate.add_argument("--config", help="the model config: a run file or a run's config.yaml (checkpoints: "
                                           "default config.yaml next to them, else configs/models/)")
    generate.add_argument("--model-path", type=Path, help="the base model directory (default: model.path)")
    generate.add_argument("--devices", nargs="+", help="npu:N / cuda:N / cpu, one process each (default: every "
                                                       "visible card)")
    generate.add_argument("--force", action="store_true", help="render every image again")
    teacher = generate.add_argument_group("teacher rows (--teacher)")
    teacher.add_argument("--steps", type=int, help="Euler steps on the pipeline's grid")
    teacher.add_argument("--cfg", type=float, help="guidance scale")
    teacher.add_argument("--weights", type=Path, help="render these weights instead of the base")
    teacher.add_argument("--sigmas", help="a released checkpoint's raw sigmas for the model's scheduler, Euler on the "
                                          "resulting grid, e.g. 1.0,0.75,0.5,0.25")
    teacher.add_argument("--scheduler-config", type=Path,
                         help="a diffusers scheduler config; sampled with that scheduler's step() like the pipeline")
    teacher.add_argument("--scheduler-set", action="append", metavar="KEY=VALUE",
                         help="override a --scheduler-config key (JSON value), e.g. flow_shift=6; repeatable")
    teacher.add_argument("--lora", type=Path, help="a diffusers / PEFT transformer LoRA file (load_lora_weights)")
    teacher.add_argument("--lora-weight", type=float, help="its adapter weight (set_adapters([name], [weight]))")

    score = sub.add_parser("score", help="score image directories (the reward models' environment)")
    score.add_argument("--dir", type=Path, nargs="+", help="image directories, or directories above them")
    score.add_argument("--metrics", nargs="+", choices=METRICS, metavar="METRIC",
                       help=f"of {', '.join(METRICS)} (default: the eval config's metrics recorded by generate, "
                            "except hpsv3pp)")
    score.add_argument("--devices", nargs="+", help="one process each (default: every visible card)")
    score.add_argument("--force", action="store_true", help="score metrics that are already scored again")
    score.add_argument("--check", action="store_true",
                       help="only check that --metrics can be scored here (weights manifest, imports)")

    summary = sub.add_parser("summary", help="a markdown table of the scored directories")
    summary.add_argument("--dir", nargs="+", required=True, help="directories holding result.json files")

    publish = sub.add_parser("publish", help="sync a run's periodic-evaluation results to its W&B run")
    publish.add_argument("--run", type=Path, required=True, help="the run directory")

    register = sub.add_parser("register-preset", help="register the W&B image gallery's chart preset for an entity")
    register.add_argument("--entity", required=True)
    register.add_argument("--base-url", default="", help="a W&B server other than the default (WANDB_BASE_URL)")
    register.add_argument("--storage-host", default="",
                          help="for a W&B server whose signed file URLs point at an object store the browser cannot "
                               "reach: that host (its files are then fetched through the W&B origin)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    sys.dont_write_bytecode = True
    args = build_parser().parse_args(argv)
    return {"generate": cmd_generate, "score": cmd_score, "summary": cmd_summary, "publish": cmd_publish,
            "register-preset": cmd_register}[args.action](args)


if __name__ == "__main__":
    raise SystemExit(main())

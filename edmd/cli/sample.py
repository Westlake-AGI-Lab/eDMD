"""``python -m edmd.cli.sample`` -- render prompts with a trained checkpoint on one device.

    python -m edmd.cli.sample --checkpoint outputs/<run>/checkpoint-<N> [--role student_ema]
        --prompt "a red fox in fresh snow" [--prompt ...] [--prompts-file prompts.txt]
        [--seed 42] [--out samples] [--device cuda:0] [--model-path DIR]
        [--tier 2k --aspect 16:9 | --width 1536 --height 1024]

Image ``i`` is rendered from seed ``seed + i`` exactly as ``edmd.cli.evaluate generate`` renders it: the
checkpoint's recorded sampler and grid (``weights/inference.json``), no guidance, PNG. ``--checkpoint`` is a run's
``checkpoint-<N>`` or a released file (``edmd_<model>_lora.safetensors``, from ``scripts/convert_lora.py``).
The model config is ``--config``, else the run's ``config.yaml`` next to the checkpoint, else ``configs/models/``.

Images have the checkpoint's training size unless ``--tier`` / ``--aspect`` pick a size from the adapter's table
(tiers ``1k`` / ``2k``; ratios 1:1, 4:3, 3:4, 3:2, 2:3, 16:9, 9:16) or ``--width`` / ``--height`` give one (multiples
of 16 pixels; 32 for Qwen-Image-2.1). The grid then follows that size, as in evaluation. A student trained at one
size may do worse at another. Run it in the model's environment (Qwen-Image-2.1: ``requirements/qwen.txt``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


def read_prompts(prompts: Sequence[str], files: Sequence[Path]) -> list[str]:
    """``--prompt`` values first, then those of every ``--prompts-file``: its non-empty lines, or for a ``.jsonl`` file
    each row's ``prompt`` (else ``positive_prompt``; ``parse_ok: false`` rows are skipped), as training reads them."""
    collected = [prompt for prompt in prompts if prompt.strip()]
    for path in files:
        if Path(path).suffix != ".jsonl":
            collected.extend(line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines()
                             if line.strip())
            continue
        with open(path, encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise SystemExit(f"{path}:{number} is not a JSON object ({error})") from None
                if isinstance(row, dict) and row.get("parse_ok") is False:   # a prompt-enhancer failure
                    continue
                prompt = row.get("prompt", row.get("positive_prompt")) if isinstance(row, dict) else None
                if not isinstance(prompt, str):
                    raise SystemExit(f"{path}:{number} has no string 'prompt' field")
                if prompt.strip():
                    collected.append(prompt)
    if not collected:
        raise SystemExit("give at least one --prompt or --prompts-file")
    return collected


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="a run's checkpoint-<N> directory, or a released .safetensors file")
    parser.add_argument("--role", choices=("student", "student_ema"),
                        help="raw or EMA weights (default: student, or the only role the weights have)")
    parser.add_argument("--config", type=Path, help="a run file or a run's config.yaml (default: config.yaml next to "
                                                    "the checkpoint, else configs/models/)")
    parser.add_argument("--model-path", type=Path, help="the base model directory (default: model.path)")
    parser.add_argument("--prompt", action="append", default=[], help="a prompt (repeatable)")
    parser.add_argument("--prompts-file", type=Path, action="append", default=[],
                        help="one prompt per line; .jsonl: one JSON object per line with a 'prompt' field")
    parser.add_argument("--seed", type=int, default=42, help="image i uses seed + i (default: 42, as the evaluation)")
    parser.add_argument("--out", type=Path, default=Path("samples"), help="output directory (default: samples)")
    parser.add_argument("--device", help="npu:N | cuda:N | cpu (default: the platform's first device)")
    parser.add_argument("--tier", help="area tier of the adapter's geometry table, e.g. 2k (with --aspect)")
    parser.add_argument("--aspect", help="aspect ratio w:h within --tier, e.g. 3:2")
    parser.add_argument("--width", type=int, help="image width in pixels (with --height)")
    parser.add_argument("--height", type=int, help="image height in pixels (with --width)")
    args = parser.parse_args(argv)
    sys.dont_write_bytecode = True
    if (args.tier is None) != (args.aspect is None):
        parser.error("--tier and --aspect go together")
    if (args.width is None) != (args.height is None):
        parser.error("--width and --height go together")
    if args.tier is not None and args.width is not None:
        parser.error("give either --tier/--aspect or --width/--height")
    prompts = read_prompts(args.prompt, args.prompts_file)

    import torch

    from ..config.load import checkpoint_model
    from ..evaluation.generate import png_bytes, render_case, resolve_grid
    from ..evaluation.protocol import default_role, read_inference_record, weights_file
    from ..models.base import Geometry, build_adapter
    from ..platform import current
    from ..runtime import device as device_lib

    checkpoint = args.checkpoint.resolve()
    record = read_inference_record(checkpoint)
    model_spec = checkpoint_model(record, config=args.config, run_dir=checkpoint.parent, model_path=args.model_path)

    args.role = args.role or default_role(record)
    if args.role not in record["roles"]:
        raise SystemExit(f"{checkpoint} has no {args.role!r} weights (roles: {sorted(record['roles'])}; see --role)")
    weights = weights_file(checkpoint, args.role)
    device = torch.device(args.device or current().device(0))
    device_lib.set_device(device)

    adapter = build_adapter(model_spec)
    grid = resolve_grid(adapter, {"kind": "recorded", "float32_hex": record["grid_float32_hex"],
                                  "num_steps": record["num_steps"], "schedule": record["schedule"]})
    view, height, width = adapter, int(record["height"]), int(record["width"])
    geometry = None
    if args.tier is not None:
        table = adapter.geometry_table()
        if args.tier not in table or args.aspect not in table[args.tier]:
            raise SystemExit(f"{model_spec.name} geometries: "
                             + "; ".join(f"{tier}: {sorted(ratios)}" for tier, ratios in table.items()))
        geometry = table[args.tier][args.aspect]
    elif args.width is not None:
        geometry = Geometry(height=args.height, width=args.width)
    if geometry is not None:
        try:
            view = adapter.at(geometry)
        except (ValueError, NotImplementedError) as error:
            raise SystemExit(str(error)) from None
        height, width = int(geometry.height), int(geometry.width)
        if view is not adapter:
            grid = view.student_grid(int(record["num_steps"]), str(record["schedule"])).detach().float().cpu()
            print(f"rendering at {width}x{height}; the checkpoint was trained at {record['width']}x"
                  f"{record['height']}", flush=True)

    adapter.load_conditioning(device)
    adapter.load_decoder(device)
    model = adapter.load_inference_denoiser(weights, device, role_train=record["roles"][args.role])
    model.eval()
    model.requires_grad_(False)
    args.out.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for index, prompt in enumerate(prompts):
            case = {"render": prompt, "seed": args.seed + index}
            array = render_case(view, model, case, grid=grid, sampler=record["sampler"], cfg=1.0, device=device,
                                height=height, width=width)
            path = args.out / f"{index:05d}.png"
            path.write_bytes(png_bytes(array))
            print(f"{path}  seed {case['seed']}  {prompt[:80]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

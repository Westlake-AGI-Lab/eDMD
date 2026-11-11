"""Write an eDMD student LoRA as one diffusers LoRA file, the format of the released weights.

    python scripts/convert_lora.py --weights outputs/<run>/checkpoint-<N> [--out <dir or file>] [--role student]

Writes ``edmd_<model>_lora.safetensors`` (into ``--out``, a directory, default the current one; or ``--out`` itself
when it ends in ``.safetensors``). It loads with ``pipe.load_lora_weights(<file>)`` (needs ``peft``): the keys get the
``transformer.`` prefix and ``alpha / rank`` is folded into ``lora_B``, so the default scale is right. The file also
carries the role's weights record and its inference record (sampler, grid, size) as metadata, so
``scripts/infer.py``, ``edmd.cli.sample`` and ``edmd.cli.evaluate`` take the file as they take the checkpoint. Then it
prints how to sample with diffusers: the student's steps on its grid, without guidance.
"""

import argparse
import json
from pathlib import Path

from infer import BASES, locate_weights, sampling_arguments

PIPELINES = {"sd3_medium": "StableDiffusion3Pipeline", "flux2_klein": "Flux2KleinPipeline",
             "qwen_image_21": "QwenImage21Pipeline"}


def model_name(record: dict) -> str:
    """``sd3_medium``, ``flux2_klein_4b``, ``flux2_klein_9b``, ``qwen_image_21`` (the names of ``configs/runs/``)."""
    return f"{record['adapter']}_{record['variant']}" if record["adapter"] == "flux2_klein" else record["adapter"]


def usage(record: dict, file: Path) -> str:
    adapter = record["adapter"]
    repo = BASES[f"{adapter}/{record['variant']}" if adapter == "flux2_klein" else adapter]
    call = ", ".join(f"{key}={value!r}" for key, value in sampling_arguments(adapter, int(record["num_steps"])).items())
    lines = ["import torch",
             f"from diffusers import {PIPELINES[adapter]}",
             "",
             f"pipe = {PIPELINES[adapter]}.from_pretrained({repo!r}, torch_dtype=torch.bfloat16)"]
    if adapter == "qwen_image_21":    # the student's grid is the pipeline's shifted grid without the terminal stretch
        lines.append("pipe.scheduler = pipe.scheduler.__class__.from_config(pipe.scheduler.config, shift_terminal=None)")
    lines += [f"pipe.load_lora_weights({str(file)!r})",
              'pipe.to("cuda")',
              f'image = pipe("a red fox in fresh snow, morning light", height={record["height"]}, '
              f'width={record["width"]}, {call}).images[0]']
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True, help="a run's checkpoint-<N> (or an older released directory)")
    parser.add_argument("--out", type=Path, default=Path("."),
                        help="a directory for edmd_<model>_lora.safetensors, or the file name (default: .)")
    parser.add_argument("--role", choices=("student", "student_ema"),
                        help="raw or EMA weights (default: student, or student_ema where only it exists)")
    args = parser.parse_args()
    import torch
    from safetensors.torch import load_file, save_file

    weights = locate_weights(args.weights)
    if weights.is_file():
        raise SystemExit(f"{weights} is already a released file: pipe.load_lora_weights({str(weights)!r})")
    record = json.loads((weights / "inference.json").read_text())
    if record["sampler"] != "euler" or record["schedule"] != "native":
        raise SystemExit(f"{record['sampler']} / {record['schedule']} students do not sample with the diffusers "
                         "pipeline; use python -m edmd.cli.sample")
    role = args.role or ("student" if (weights / "student.safetensors").is_file() else "student_ema")
    role_record = json.loads((weights / f"{role}.json").read_text())
    lora = role_record.get("lora")
    if not lora:
        raise SystemExit(f"{role} is fully fine-tuned, not a LoRA: nothing to convert")
    scale = float(lora["alpha"]) / int(lora["rank"])
    converted = {}
    for key, value in load_file(str(weights / f"{role}.safetensors")).items():
        if not key.endswith((".lora_A.weight", ".lora_B.weight")):
            raise SystemExit(f"unexpected tensor {key} in {role}.safetensors")
        value = value.to(torch.float32)
        converted[f"transformer.{key}"] = (value * scale if key.endswith(".lora_B.weight") else value).contiguous()
    # what eDMD reads back (edmd.runtime.checkpoint.read_weights, edmd.evaluation.protocol.release_info): the role,
    # its weights record (LoRA rank, alpha, targets) and the inference record, with this role only
    record = {key: value for key, value in record.items() if key != "base"}
    record["roles"] = {role: "lora"}
    role_record = {key: value for key, value in role_record.items() if key != "base"}
    edmd = json.dumps({"role": role, "weights": role_record, "inference": record}, sort_keys=True)
    out = args.out if args.out.suffix == ".safetensors" else args.out / f"edmd_{model_name(record)}_lora.safetensors"
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(converted, str(out), metadata={"format": "pt", "edmd": edmd})
    print(f"{out}: {len(converted) // 2} LoRA layers of {record['adapter']} {record['variant']} ({role}, "
          f"rank {lora['rank']}, alpha {lora['alpha']} folded in)")
    if record["adapter"] in PIPELINES:
        print(f"\n{usage(record, out)}")


if __name__ == "__main__":
    main()

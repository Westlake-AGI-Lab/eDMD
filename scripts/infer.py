"""Generate images with an eDMD student through the plain diffusers pipeline.

    python scripts/infer.py --weights zttzzt/eDMD/flux2_klein_4b/edmd_flux2_klein_4b_lora.safetensors \
        --prompt "a red fox in fresh snow" [--prompt ...] [--base <dir or HF id>] [--width 1024 --height 1024]
        [--seed 42] [--out samples]

``--weights`` is a released file (``edmd_<model>_lora.safetensors``, written by ``scripts/convert_lora.py``): a local
path, or ``<org>/<repo>/<path in the repo>`` on Hugging Face, downloaded on first use; or a run's ``checkpoint-<N>``.
The model and its sampler come with the weights; the student LoRA is merged into the base transformer, and the pipeline
samples with the student's steps and no guidance. The base model is ``--base``, else ``$EDMD_MODEL_ROOT/<org>/<repo>``
when it exists, else the Hugging Face repository.

Only the packages of the model's own environment are needed (diffusers, transformers, safetensors).
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

BASES = {"sd3_medium": "stabilityai/stable-diffusion-3-medium-diffusers",
         "flux2_klein/4b": "black-forest-labs/FLUX.2-klein-base-4B",
         "flux2_klein/9b": "black-forest-labs/FLUX.2-klein-base-9B",
         "qwen_image_21": "Qwen/Qwen-Image-2.1"}


def load_pipeline(adapter: str, base: str):
    import diffusers

    if adapter == "sd3_medium":
        return diffusers.StableDiffusion3Pipeline.from_pretrained(base, torch_dtype=torch.bfloat16)
    if adapter == "flux2_klein":
        return diffusers.Flux2KleinPipeline.from_pretrained(base, torch_dtype=torch.bfloat16)
    if not hasattr(diffusers, "QwenImage21Pipeline"):
        raise SystemExit("Qwen-Image-2.1 needs the diffusers build of requirements/qwen.txt")
    pipe = diffusers.QwenImage21Pipeline.from_pretrained(base, torch_dtype=torch.bfloat16)
    # the student's grid is the pipeline's shifted grid without the terminal stretch
    pipe.scheduler = pipe.scheduler.__class__.from_config(pipe.scheduler.config, shift_terminal=None)
    return pipe


def sampling_arguments(adapter: str, steps: int) -> dict:
    """The pipeline call that runs the student's own sampler: Euler on its grid, no guidance."""
    if adapter == "sd3_medium":     # static shift 3 of linspace(1, 1/K, K); the pipeline's default grid differs
        return {"num_inference_steps": steps, "sigmas": np.linspace(1.0, 1.0 / steps, steps).tolist(),
                "guidance_scale": 1.0}
    if adapter == "flux2_klein":    # the pipeline's default grid for this step count and image size
        return {"num_inference_steps": steps, "guidance_scale": 1.0}
    return {"num_inference_steps": steps, "true_cfg_scale": 1.0}


def locate_weights(value: str) -> Path:
    """A released file, or the weights directory of a checkpoint; ``<org>/<repo>/<path>.safetensors`` that is not a
    local path is downloaded from Hugging Face."""
    path = Path(value).expanduser()
    if not path.exists():
        parts = value.strip("/").split("/")
        if len(parts) < 3 or not value.endswith(".safetensors"):
            raise SystemExit(f"--weights {value}: no such file or directory (a Hugging Face file is "
                             "<org>/<repo>/<path>.safetensors)")
        from huggingface_hub import hf_hub_download

        path = Path(hf_hub_download("/".join(parts[:2]), "/".join(parts[2:])))
    if path.is_file():
        return path
    weights = path / "weights" if (path / "weights" / "inference.json").is_file() else path
    if not (weights / "inference.json").is_file():
        raise SystemExit(f"{path} holds no inference.json: give a released .safetensors file or a run's checkpoint-<N>")
    return weights


def load_student(weights: Path, role: str | None) -> tuple[dict, str, dict, float]:
    """``(inference record, role, LoRA tensors, merge scale)`` of a released file or of a checkpoint's role."""
    from safetensors import safe_open
    from safetensors.torch import load_file

    if weights.is_file():
        with safe_open(str(weights), framework="pt", device="cpu") as handle:
            info = json.loads((handle.metadata() or {}).get("edmd", "null"))
            if info is None:
                raise SystemExit(f"{weights} carries no eDMD sampling record: write it with scripts/convert_lora.py")
            tensors = {key.removeprefix("transformer."): handle.get_tensor(key) for key in handle.keys()}
        return info["inference"], info["role"], tensors, 1.0         # alpha / rank is folded into lora_B
    record = json.loads((weights / "inference.json").read_text())
    role = role or ("student" if (weights / "student.safetensors").is_file() else "student_ema")
    lora = json.loads((weights / f"{role}.json").read_text())["lora"]
    if not lora:
        raise SystemExit(f"{role} is fully fine-tuned, not a LoRA: sample it with python -m edmd.cli.sample")
    return record, role, load_file(str(weights / f"{role}.safetensors")), float(lora["alpha"]) / int(lora["rank"])


@torch.no_grad()
def merge_lora(transformer: torch.nn.Module, tensors: dict, scale: float) -> int:
    """``W += scale * B @ A`` for every LoRA pair, computed in float32."""
    merged = 0
    for key in tensors:
        if not key.endswith(".lora_A.weight"):
            continue
        name = key[: -len(".lora_A.weight")]
        layer = transformer.get_submodule(name)
        down, up = tensors[key].float(), tensors[f"{name}.lora_B.weight"].float()
        layer.weight.copy_((layer.weight.float() + scale * (up @ down)).to(layer.weight.dtype))
        merged += 1
    return merged


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    try:
        import torch_npu  # noqa: F401

        return "npu"
    except ImportError:
        return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True,
                        help="a released .safetensors file (local, or <org>/<repo>/<file> on Hugging Face), or a "
                             "run's checkpoint-<N>")
    parser.add_argument("--prompt", action="append", required=True, help="a prompt (repeatable)")
    parser.add_argument("--role", choices=("student", "student_ema"),
                        help="a checkpoint's raw or EMA weights (default: student; a released file holds one)")
    parser.add_argument("--base", help="base model directory or Hugging Face id")
    parser.add_argument("--width", type=int, help="default: the training size")
    parser.add_argument("--height", type=int, help="default: the training size")
    parser.add_argument("--seed", type=int, default=42, help="image i uses seed + i")
    parser.add_argument("--out", type=Path, default=Path("samples"))
    parser.add_argument("--device", default=default_device())
    args = parser.parse_args()

    record, role, tensors, scale = load_student(locate_weights(args.weights), args.role)
    adapter = record["adapter"]
    if record["sampler"] != "euler" or record["schedule"] != "native":
        raise SystemExit(f"{record['sampler']} / {record['schedule']} students need python -m edmd.cli.sample")

    repo = BASES[f"{adapter}/{record['variant']}" if adapter == "flux2_klein" else adapter]
    local = Path(os.environ.get("EDMD_MODEL_ROOT", "")) / repo
    base = args.base or (str(local) if os.environ.get("EDMD_MODEL_ROOT") and local.is_dir() else repo)
    if args.device.startswith("npu"):
        import torch_npu  # noqa: F401

    pipe = load_pipeline(adapter, base)
    merged = merge_lora(pipe.transformer, tensors, scale)
    pipe.to(args.device)
    print(f"{adapter} {record['variant']}: {base} + {role} ({merged} LoRA layers merged)")

    height, width = args.height or int(record["height"]), args.width or int(record["width"])
    args.out.mkdir(parents=True, exist_ok=True)
    for index, prompt in enumerate(args.prompt):
        # the evaluation's generator device: for SD3 and FLUX.2 a seed starts from edmd.cli.sample's noise
        generator = torch.Generator("cpu" if adapter == "qwen_image_21" else args.device).manual_seed(args.seed + index)
        image = pipe(prompt=prompt, height=height, width=width, generator=generator,
                     **sampling_arguments(adapter, int(record["num_steps"]))).images[0]
        path = args.out / f"{index:05d}.png"
        image.save(path)
        print(f"{path}  seed {args.seed + index}  {prompt[:80]}")


if __name__ == "__main__":
    main()

"""Building and loading the Qwen-Image-2.1 components (meta transformer, base weights, derived tensors, encoders).

diffusers / transformers are imported lazily.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from torch import nn

from . import npu_compat

TRANSFORMER_SUBDIR = "transformer"
TEXT_ENCODER_SUBDIR = "text_encoder"
PROCESSOR_SUBDIR = "processor"
VAE_SUBDIR = "vae"
TRANSFORMER_CLASS = "QwenImage21Transformer2DModel"
BLOCK_CLASS = "QwenImage21TransformerBlock"

# Every projection of the single-stream blocks (attention q/k/v/out and the SwiGLU MLP); the shared modulation, the
# timestep embedder, txt_in, img_in and proj_out stay frozen.
DEFAULT_LORA_TARGETS: tuple[str, ...] = (
    "transformer_blocks.*.attn.to_q",
    "transformer_blocks.*.attn.to_k",
    "transformer_blocks.*.attn.to_v",
    "transformer_blocks.*.attn.to_out.0",
    "transformer_blocks.*.img_mlp.proj",
    "transformer_blocks.*.img_mlp.gate_layer",
    "transformer_blocks.*.img_mlp.out",
)


def require_diffusers():
    """Import the QwenImage21 module, installing the NPU patches; a clear error when diffusers lacks it."""
    try:
        from diffusers.models.transformers import transformer_qwenimage21  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ImportError(
            "Qwen-Image-2.1 needs a diffusers build with the QwenImage21 classes (diffusers main, 0.41.0.dev0 or "
            "newer: QwenImage21Transformer2DModel, AutoencoderKLQwenImage21) and transformers with "
            "Qwen3VLForConditionalGeneration (tested with 5.17): run Qwen-Image-2.1 in the environment of "
            "requirements/qwen.txt.") from exc
    npu_compat.install()
    return transformer_qwenimage21


def load_transformer_config(model_path: str | Path) -> dict[str, Any]:
    path = Path(model_path) / TRANSFORMER_SUBDIR / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; point model.path at a full Qwen-Image-2.1 checkpoint.")
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("_class_name") not in (None, TRANSFORMER_CLASS):
        raise ValueError(f"{path} describes {config['_class_name']}, not {TRANSFORMER_CLASS}.")
    return config


def _constructor_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if not str(key).startswith("_")}


def build_transformer(config: Mapping[str, Any], *, device: str | torch.device = "meta") -> nn.Module:
    """``QwenImage21Transformer2DModel`` from a checkpoint config dict, constructed under ``device``."""
    module = require_diffusers()
    with torch.device(device):
        return module.QwenImage21Transformer2DModel(**_constructor_kwargs(config))


def transformer_blocks(model: nn.Module) -> list[nn.Module]:
    """Every ``QwenImage21TransformerBlock`` (the FSDP units), checked against a full module scan."""
    blocks = list(getattr(model, "transformer_blocks", []))
    found = [module for module in model.modules() if type(module).__name__ == BLOCK_CLASS]
    if not blocks or len(found) != len(blocks) or any(a is not b for a, b in zip(blocks, found)):
        raise RuntimeError(f"Expected transformer_blocks to hold every {BLOCK_CLASS} (found {len(found)}, "
                           f"listed {len(blocks)}); the upstream model changed.")
    return blocks


def weight_files(model_path: str | Path) -> list[Path]:
    directory = Path(model_path) / TRANSFORMER_SUBDIR
    index_path = directory / "diffusion_pytorch_model.safetensors.index.json"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        files = [directory / name for name in sorted(set(weight_map.values()))]
    else:
        files = sorted(directory.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No safetensors weights under {directory}.")
    return files


def read_transformer_state(model_path: str | Path, expected: Iterable[str], *,
                           dtype: torch.dtype = torch.bfloat16) -> dict[str, torch.Tensor]:
    """CPU tensors of exactly the ``expected`` keys; missing or surplus checkpoint keys raise."""
    from safetensors import safe_open

    wanted = set(expected)
    state: dict[str, torch.Tensor] = {}
    surplus: list[str] = []
    for path in weight_files(model_path):
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in wanted:
                    state[key] = handle.get_tensor(key).to(dtype)
                else:
                    surplus.append(key)
    missing = sorted(wanted - set(state))
    if missing:
        raise RuntimeError(f"The checkpoint lacks {len(missing)} transformer tensors, e.g. {missing[:3]}")
    if surplus:
        raise RuntimeError(f"The checkpoint holds {len(surplus)} tensors the model has no home for, e.g. {surplus[:3]}")
    return state


def reset_derived_tensors(model: nn.Module, device: torch.device | str) -> None:
    """Rebuild what ``to_empty`` destroys and the checkpoint does not carry, with the upstream formulas: the rope
    tables (a plain list; on the CPU, where the NPU patch gathers) and the sinusoid frequencies (on ``device``)."""
    rope = getattr(model, "pos_embed", None)
    if rope is None or not hasattr(rope, "axes_dim") or not hasattr(rope, "rope_params"):
        raise RuntimeError("The transformer has no pos_embed with axes_dim/rope_params; the upstream model changed.")
    pos_index = torch.arange(8192)
    neg_index = torch.arange(1024).flip(0) * -1 - 1
    rope.freqs = [torch.cat([rope.rope_params(pos_index, dim, rope.theta), rope.rope_params(neg_index, dim, rope.theta)],
                            dim=0) for dim in rope.axes_dim]
    npu_compat.drop_rope_cache(rope)

    time_proj = model.time_text_embed.time_proj
    half = time_proj.timestep_dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half, dtype=torch.float32) / half)
    time_proj.freqs = freqs.to(device)

    remaining = [name for name, tensor in list(model.named_parameters()) + list(model.named_buffers())
                 if tensor.is_meta]
    remaining += [f"pos_embed.freqs[{index}]" for index, table in enumerate(rope.freqs) if table.is_meta]
    if remaining:
        raise RuntimeError(f"{len(remaining)} tensors are still on meta after materialization, e.g. {remaining[:3]}")


def load_pretrained_transformer(model_path: str | Path, *, dtype: torch.dtype, device: torch.device | str) -> nn.Module:
    """One unsharded frozen transformer (inference / export); training builds on meta instead."""
    module = require_diffusers()
    model = module.QwenImage21Transformer2DModel.from_pretrained(
        str(Path(model_path) / TRANSFORMER_SUBDIR), dtype=dtype, local_files_only=True)
    model.requires_grad_(False)
    model.eval()
    return model.to(device)


def load_text_encoder(model_path: str | Path, *, dtype: torch.dtype, device: torch.device | str):
    """The frozen Qwen3-VL encoder, the class the pipeline registers."""
    from transformers import Qwen3VLForConditionalGeneration

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(Path(model_path) / TEXT_ENCODER_SUBDIR), dtype=dtype, local_files_only=True)
    model.requires_grad_(False)
    model.eval()
    return model.to(device)


def load_processor(model_path: str | Path):
    from transformers import Qwen3VLProcessor

    return Qwen3VLProcessor.from_pretrained(str(Path(model_path) / PROCESSOR_SUBDIR), local_files_only=True)


def text_encoder_units(text_encoder: nn.Module) -> list[nn.Module]:
    """FSDP units of the encoder: one per decoder layer of the language model (plus the root, added by the caller)."""
    from .text import text_decoder

    layers = getattr(text_decoder(text_encoder), "layers", None)
    if layers is None or len(layers) == 0:
        raise RuntimeError("The text encoder has no language_model.layers to shard; the upstream model changed.")
    return list(layers)


def shard_text_encoder(text_encoder: nn.Module) -> nn.Module:
    """``fully_shard`` each decoder layer, then the root, in place (needs an initialized process group).

    FSDP's param dtype is the encoder's own: forcing bf16 would break the adapter's float32 mode.
    """
    import torch.distributed as dist

    from ...runtime.distributed import shard_module

    if not dist.is_initialized():
        raise RuntimeError("text_encoder_shard needs torch.distributed to be initialized (torchrun).")
    dtype = next(text_encoder.parameters()).dtype
    return shard_module(text_encoder, text_encoder_units(text_encoder), reshard_after_forward=True,
                        param_dtype=dtype)


def load_vae(model_path: str | Path, *, dtype: torch.dtype, device: torch.device | str):
    from diffusers import AutoencoderKLQwenImage21

    vae = AutoencoderKLQwenImage21.from_pretrained(str(Path(model_path) / VAE_SUBDIR), dtype=dtype,
                                                   local_files_only=True)
    vae.requires_grad_(False)
    vae.eval()
    return vae.to(device)


__all__ = [
    "BLOCK_CLASS", "DEFAULT_LORA_TARGETS", "build_transformer", "load_pretrained_transformer",
    "load_processor", "load_text_encoder", "load_transformer_config", "load_vae", "read_transformer_state",
    "require_diffusers", "reset_derived_tensors", "shard_text_encoder", "text_encoder_units", "transformer_blocks",
    "weight_files",
]

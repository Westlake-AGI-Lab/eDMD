"""FLUX.2 klein checkpoint components: configs, variants, meta construction, base weights, FSDP units, LoRA targets.

Everything is read from one diffusers checkpoint directory, the layout ``Flux2KleinPipeline.from_pretrained`` loads.
4B and 9B share the architecture class but differ in size and text encoder (Qwen3-4B / Qwen3-8B); :data:`VARIANTS`
pins each signature so a variant/path mix-up fails at load time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from torch import Tensor, nn

TRANSFORMER_SUBDIR = "transformer"
TEXT_ENCODER_SUBDIR = "text_encoder"
TOKENIZER_SUBDIR = "tokenizer"
VAE_SUBDIR = "vae"
SCHEDULER_SUBDIR = "scheduler"
TRANSFORMER_WEIGHTS = "diffusion_pytorch_model.safetensors"
TRANSFORMER_INDEX = "diffusion_pytorch_model.safetensors.index.json"

#: packed latent channels: 32 VAE channels x a 2x2 patch
LATENT_CHANNELS = 128
#: pixels per latent token: VAE 8x downsampling times the 2x2 patch
PIXELS_PER_TOKEN = 16

#: Architecture signature of each supported checkpoint (transformer/config.json and text_encoder/config.json).
VARIANTS: dict[str, dict[str, int]] = {
    "4b": {"num_layers": 5, "num_single_layers": 20, "num_attention_heads": 24, "attention_head_dim": 128,
           "joint_attention_dim": 7680, "text_hidden_size": 2560, "text_num_hidden_layers": 36},
    "9b": {"num_layers": 8, "num_single_layers": 24, "num_attention_heads": 32, "attention_head_dim": 128,
           "joint_attention_dim": 12288, "text_hidden_size": 4096, "text_num_hidden_layers": 36},
}

DEFAULT_LORA_TARGETS: tuple[str, ...] = (
    "transformer_blocks.*.attn.to_q",
    "transformer_blocks.*.attn.to_k",
    "transformer_blocks.*.attn.to_v",
    "transformer_blocks.*.attn.to_out.0",
    "transformer_blocks.*.attn.add_q_proj",
    "transformer_blocks.*.attn.add_k_proj",
    "transformer_blocks.*.attn.add_v_proj",
    "transformer_blocks.*.attn.to_add_out",
    "single_transformer_blocks.*.attn.to_qkv_mlp_proj",
    "single_transformer_blocks.*.attn.to_out",
)

_EXPECTED_COMPONENTS = {
    "transformer": ["diffusers", "Flux2Transformer2DModel"],
    "text_encoder": ["transformers", "Qwen3ForCausalLM"],
    "tokenizer": ["transformers", "Qwen2TokenizerFast"],
    "vae": ["diffusers", "AutoencoderKLFlux2"],
    "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
}


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; model.path must be a full FLUX.2 klein diffusers checkpoint.")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object.")
    return value


def check_model_index(index: Mapping[str, Any]) -> None:
    """A klein *base* checkpoint, not step-distilled (``is_distilled`` changes the pipeline's CFG and guidance input).
    Another pipeline or component class name than the Klein pipeline's only warns (the configs and the strict weight
    load check the architecture)."""
    if index.get("_class_name") != "Flux2KleinPipeline":
        print(f"warning: model_index.json describes {index.get('_class_name')!r}, not Flux2KleinPipeline", flush=True)
    for name, expected in _EXPECTED_COMPONENTS.items():
        if list(index.get(name) or []) != expected:
            print(f"warning: model_index.json component {name}={index.get(name)!r}, expected {expected}", flush=True)
    if bool(index.get("is_distilled", False)):
        raise ValueError("This is a step-distilled klein checkpoint; the adapter supports the undistilled base models.")


def check_transformer_config(config: Mapping[str, Any], variant: str) -> None:
    if config.get("_class_name") not in (None, "Flux2Transformer2DModel"):
        print(f"warning: transformer/config.json describes {config.get('_class_name')!r}, not "
              "Flux2Transformer2DModel", flush=True)
    if bool(config.get("guidance_embeds", True)):
        raise ValueError("guidance_embeds must be false: klein base takes no guidance embedding (guidance=None).")
    if int(config.get("in_channels", 0)) != LATENT_CHANNELS or config.get("out_channels") not in (None, LATENT_CHANNELS):
        raise ValueError(f"Expected {LATENT_CHANNELS} packed latent channels, got in={config.get('in_channels')} "
                         f"out={config.get('out_channels')}.")
    if int(config.get("patch_size", 1)) != 1:
        raise ValueError("Expected patch_size 1 (the 2x2 patch is folded into the 128 latent channels).")
    signature = variant_signature(variant)
    wrong = {key: config.get(key) for key in ("num_layers", "num_single_layers", "num_attention_heads",
                                              "attention_head_dim", "joint_attention_dim")
             if config.get(key) != signature[key]}
    if wrong:
        raise ValueError(f"transformer/config.json does not match variant {variant!r} {signature}: {wrong}")


def check_text_encoder_config(config: Mapping[str, Any], variant: str, layers: Iterable[int]) -> None:
    """Qwen3 with the variant's width; the selected layers exist and their concatenation feeds the transformer."""
    if config.get("model_type") != "qwen3":
        raise ValueError(f"text_encoder/config.json is model_type {config.get('model_type')!r}, expected qwen3.")
    if list(config.get("architectures") or []) != ["Qwen3ForCausalLM"]:
        print(f"warning: text_encoder/config.json architectures {config.get('architectures')}, expected "
              "['Qwen3ForCausalLM']", flush=True)
    signature = variant_signature(variant)
    hidden = int(config.get("hidden_size", -1))
    depth = int(config.get("num_hidden_layers", -1))
    if hidden != signature["text_hidden_size"] or depth != signature["text_num_hidden_layers"]:
        raise ValueError(f"The {variant} text encoder should be hidden={signature['text_hidden_size']}, "
                         f"layers={signature['text_num_hidden_layers']}; got hidden={hidden}, layers={depth}.")
    layers = list(layers)
    if not layers or any(not 0 <= int(layer) <= depth for layer in layers):
        raise ValueError(f"text_encoder_out_layers {layers} must index hidden_states[0..{depth}].")
    if len(layers) * hidden != signature["joint_attention_dim"]:
        raise ValueError(f"{len(layers)} layers x hidden {hidden} != joint_attention_dim {signature['joint_attention_dim']}.")


def variant_signature(variant: str) -> dict[str, int]:
    if variant not in VARIANTS:
        raise ValueError(f"FLUX.2 klein variant must be one of {sorted(VARIANTS)}, got {variant!r}.")
    return VARIANTS[variant]


# ----------------------------------------------------------------------------------- transformer
def build_transformer_on_meta(config: Mapping[str, Any], *, dtype: torch.dtype) -> nn.Module:
    """``Flux2Transformer2DModel`` on the meta device in ``dtype``."""
    from diffusers import Flux2Transformer2DModel

    kwargs = {key: value for key, value in config.items() if not str(key).startswith("_")}
    with torch.device("meta"):
        model = Flux2Transformer2DModel(**kwargs)
    # nn.Module.to, not ModelMixin.to: the latter warns about _keep_in_fp32_modules (empty for FLUX.2) on every build
    return nn.Module.to(model, dtype=dtype)


def transformer_blocks(model: nn.Module) -> tuple[list[nn.Module], list[nn.Module]]:
    """The double-stream and single-stream blocks, checked against the classes and the config."""
    from diffusers.models.transformers.transformer_flux2 import Flux2SingleTransformerBlock, Flux2TransformerBlock

    double = list(getattr(model, "transformer_blocks", []))
    single = list(getattr(model, "single_transformer_blocks", []))
    if not all(isinstance(block, Flux2TransformerBlock) for block in double):
        raise RuntimeError("model.transformer_blocks holds something other than Flux2TransformerBlock.")
    if not all(isinstance(block, Flux2SingleTransformerBlock) for block in single):
        raise RuntimeError("model.single_transformer_blocks holds something other than Flux2SingleTransformerBlock.")
    config = getattr(model, "config", None)
    if config is not None and (len(double) != int(config.num_layers) or len(single) != int(config.num_single_layers)):
        raise RuntimeError(f"{len(double)}+{len(single)} blocks, config says {config.num_layers}+{config.num_single_layers}.")
    every = [module for module in model.modules() if isinstance(module, (Flux2TransformerBlock, Flux2SingleTransformerBlock))]
    if len(every) != len(double) + len(single):
        raise RuntimeError(f"{len(every)} transformer blocks in the model but only {len(double) + len(single)} "
                           "are in transformer_blocks / single_transformer_blocks; the upstream model changed.")
    return double, single


def fsdp_units(model: nn.Module) -> list[nn.Module]:
    double, single = transformer_blocks(model)
    if not double or not single:
        raise RuntimeError("A FLUX.2 transformer has both double- and single-stream blocks.")
    return double + single


def check_no_buffers(model: nn.Module) -> None:
    """FLUX.2 has no buffers, hence nothing for ``to_empty`` to destroy that the state dict does not carry."""
    buffers = [name for name, _ in model.named_buffers()]
    if buffers:
        raise RuntimeError(f"Flux2Transformer2DModel now has buffers {buffers[:5]}; reset_derived must rebuild "
                           "the non-persistent ones after to_empty (see edmd/models/flux2/components.py).")


def transformer_weight_files(model_path: str | Path) -> list[Path]:
    directory = Path(model_path) / TRANSFORMER_SUBDIR
    index = directory / TRANSFORMER_INDEX
    if index.is_file():
        weight_map = load_json(index).get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"{index} has no weight_map.")
        return [directory / name for name in sorted(set(weight_map.values()))]
    single = directory / TRANSFORMER_WEIGHTS
    if not single.is_file():
        raise FileNotFoundError(f"Neither {index} nor {single} exists.")
    return [single]


def read_transformer_state(model_path: str | Path, expected: Iterable[str], *, model_keys: Iterable[str],
                           dtype: torch.dtype) -> dict[str, Tensor]:
    """CPU tensors for exactly ``expected``, from the safetensors shards. Strict both ways: a checkpoint key the
    architecture lacks (e.g. a guidance embedder) means config and weights disagree."""
    from safetensors import safe_open

    wanted = set(expected)
    known = set(model_keys)
    stray = sorted(wanted - known)
    if stray:
        raise ValueError(f"Requested keys that the transformer does not have, e.g. {stray[:3]}")
    state: dict[str, Tensor] = {}
    seen: set[str] = set()
    for path in transformer_weight_files(model_path):
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in seen:
                    raise ValueError(f"{key} appears in more than one shard.")
                seen.add(key)
                if key in wanted:
                    state[key] = handle.get_tensor(key).to(dtype)
    unexpected = sorted(seen - known)
    missing = sorted(wanted - seen)
    if unexpected or missing:
        raise RuntimeError(f"{Path(model_path) / TRANSFORMER_SUBDIR} does not match the architecture: "
                           f"missing={missing[:5]} ({len(missing)}), unexpected={unexpected[:5]} ({len(unexpected)})")
    return state


def load_state_into_meta(model: nn.Module, state: Mapping[str, Tensor], *, device: torch.device | str) -> nn.Module:
    """Unsharded load (inference / export): assign CPU tensors to a meta model, strictly, then move it."""
    result = model.load_state_dict(dict(state), strict=True, assign=True)
    if result.missing_keys or result.unexpected_keys:  # strict=True raises first; kept for clarity
        raise RuntimeError(f"State mismatch: {result}")
    leftover = [name for name, tensor in list(model.named_parameters()) + list(model.named_buffers()) if tensor.is_meta]
    if leftover:
        raise RuntimeError(f"Tensors left on meta: {leftover[:5]}")
    return model.to(device)


# ----------------------------------------------------------------------------------- frozen components
def load_text_encoder(model_path: str | Path, *, dtype: torch.dtype, device: torch.device | str) -> nn.Module:
    """The frozen Qwen3 causal LM, loaded as the pipeline loads it (``from_pretrained``)."""
    from transformers import Qwen3ForCausalLM

    model = Qwen3ForCausalLM.from_pretrained(str(Path(model_path) / TEXT_ENCODER_SUBDIR), torch_dtype=dtype,
                                             local_files_only=True)
    model.requires_grad_(False)
    model.eval()
    return model.to(device)


def load_tokenizer(model_path: str | Path):
    from transformers import Qwen2TokenizerFast

    return Qwen2TokenizerFast.from_pretrained(str(Path(model_path) / TOKENIZER_SUBDIR), local_files_only=True)


def text_encoder_layers(model: nn.Module) -> list[nn.Module]:
    """The Qwen3 decoder layers: the FSDP units when the frozen encoder is sharded."""
    from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

    layers = list(getattr(getattr(model, "model", None), "layers", []) or [])
    if not layers or not all(isinstance(layer, Qwen3DecoderLayer) for layer in layers):
        raise RuntimeError("The text encoder has no model.layers of Qwen3DecoderLayer; the upstream model changed.")
    return layers


def load_vae(model_path: str | Path, *, dtype: torch.dtype, device: torch.device | str) -> nn.Module:
    from diffusers import AutoencoderKLFlux2

    vae = AutoencoderKLFlux2.from_pretrained(str(Path(model_path) / VAE_SUBDIR), torch_dtype=dtype, local_files_only=True)
    vae.requires_grad_(False)
    vae.eval()
    return vae.to(device)


__all__ = [
    "DEFAULT_LORA_TARGETS", "LATENT_CHANNELS", "PIXELS_PER_TOKEN", "VARIANTS", "build_transformer_on_meta",
    "check_model_index", "check_no_buffers", "check_text_encoder_config", "check_transformer_config", "fsdp_units",
    "load_json", "load_state_into_meta", "load_text_encoder", "load_tokenizer", "load_vae", "read_transformer_state",
    "text_encoder_layers", "transformer_blocks", "transformer_weight_files", "variant_signature",
]

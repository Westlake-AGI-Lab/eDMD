"""Loading the SD3-Medium components: the transformer (meta build + strict weight read), text encoders, VAE.

The checkpoint is a diffusers SD3 directory (``stabilityai/stable-diffusion-3-medium-diffusers``). Like
``DiffusionPipeline.from_pretrained`` without ``variant=``, the plain weight files are read and the ``fp16`` variant
only when no plain file exists.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import nn

TRANSFORMER_SUBDIR = "transformer"
VAE_SUBDIR = "vae"
TEXT_ENCODER_SUBDIRS = ("text_encoder", "text_encoder_2", "text_encoder_3")
TOKENIZER_SUBDIRS = ("tokenizer", "tokenizer_2", "tokenizer_3")
WEIGHTS_NAME = "diffusion_pytorch_model"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; point model.path at a full SD3 diffusers checkpoint.")
    return json.loads(path.read_text(encoding="utf-8"))


def load_transformer_config(model_path: str | Path) -> dict[str, Any]:
    """The transformer config, restricted to SD3-Medium's architecture (no QK-norm, no dual attention)."""
    config = _read_json(Path(model_path) / TRANSFORMER_SUBDIR / "config.json")
    if config.get("_class_name") not in (None, "SD3Transformer2DModel"):
        raise ValueError(f"{model_path}/transformer describes {config['_class_name']}, not SD3Transformer2DModel.")
    if config.get("qk_norm") is not None or tuple(config.get("dual_attention_layers") or ()):
        # SD3.5's QK-norm and attn2 blocks would need their own LoRA targets and numerics checks
        raise NotImplementedError("The sd3_medium adapter supports SD3-Medium only (qk_norm=None, no dual attention).")
    return config


def load_vae_config(model_path: str | Path) -> dict[str, Any]:
    config = _read_json(Path(model_path) / VAE_SUBDIR / "config.json")
    if config.get("_class_name") not in (None, "AutoencoderKL"):
        raise ValueError(f"{model_path}/vae describes {config['_class_name']}, not AutoencoderKL.")
    return config


def vae_scale_factor(vae_config: Mapping[str, Any]) -> int:
    """``2 ** (len(block_out_channels) - 1)``, the pipeline's ``self.vae_scale_factor`` (8 for SD3)."""
    return 2 ** (len(vae_config["block_out_channels"]) - 1)


def _model_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if not str(key).startswith("_")}


def build_transformer_on_meta(config: Mapping[str, Any]) -> nn.Module:
    """A float32 ``SD3Transformer2DModel`` on the meta device, from the checkpoint config dict."""
    from diffusers import SD3Transformer2DModel

    with torch.device("meta"):
        model = SD3Transformer2DModel(**_model_kwargs(config))
    return model


def transformer_blocks(model: nn.Module) -> list[nn.Module]:
    """Every ``JointTransformerBlock`` (the FSDP units), checked against the config's ``num_layers``."""
    from diffusers.models.attention import JointTransformerBlock

    blocks = list(getattr(model, "transformer_blocks", []))
    expected = int(getattr(getattr(model, "config", None), "num_layers", len(blocks)))
    if len(blocks) != expected or not blocks:
        raise RuntimeError(f"Expected {expected} transformer blocks, found {len(blocks)}.")
    wrong = [type(block).__name__ for block in blocks if not isinstance(block, JointTransformerBlock)]
    if wrong:
        raise RuntimeError(f"Unexpected SD3 block types {sorted(set(wrong))}; the upstream model changed.")
    return blocks


def non_persistent_buffers(model: nn.Module) -> list[str]:
    """Buffers ``to_empty`` destroys that no state dict restores."""
    names = []
    for prefix, module in model.named_modules():
        for name in getattr(module, "_non_persistent_buffers_set", ()):
            names.append(f"{prefix}.{name}" if prefix else name)
    return sorted(names)


def transformer_weight_files(model_path: str | Path) -> tuple[list[Path], str | None]:
    """The transformer safetensors to read: the plain file(s) if present, else the ``fp16`` variant."""
    return weight_files(Path(model_path) / TRANSFORMER_SUBDIR)


def weight_files(directory: str | Path) -> tuple[list[Path], str | None]:
    """Safetensors of one diffusers model directory: the plain file(s) if present, else the ``fp16`` variant."""
    directory = Path(directory)
    for variant in (None, "fp16"):
        index = directory / (f"{WEIGHTS_NAME}.safetensors.index.json" if variant is None
                             else f"{WEIGHTS_NAME}.safetensors.index.{variant}.json")
        single = directory / (f"{WEIGHTS_NAME}.safetensors" if variant is None
                              else f"{WEIGHTS_NAME}.{variant}.safetensors")
        if index.is_file():
            weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
            return [directory / name for name in sorted(set(weight_map.values()))], variant
        if single.is_file():
            return [single], variant
    raise FileNotFoundError(f"No transformer safetensors under {directory}.")


def read_transformer_state(model_path: str | Path, expected: Iterable[str], *,
                           dtype: torch.dtype) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Strictly the ``expected`` tensors as CPU ``dtype`` tensors, plus provenance (files, variant, file dtypes)."""
    files, variant = transformer_weight_files(model_path)
    return read_state(files, expected, dtype=dtype, where=str(model_path), variant=variant)


def read_state(files: Sequence[Path], expected: Iterable[str], *, dtype: torch.dtype, where: str,
               variant: str | None = None, optional: Iterable[str] = ()) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """The ``expected`` tensors of ``files`` as CPU ``dtype`` tensors; keys in ``optional`` may be absent."""
    from safetensors import safe_open

    wanted, maybe = set(expected), set(optional)
    state: dict[str, torch.Tensor] = {}
    unexpected: list[str] = []
    file_dtypes: set[str] = set()
    for path in files:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key not in wanted:
                    unexpected.append(key)
                    continue
                tensor = handle.get_tensor(key)
                file_dtypes.add(str(tensor.dtype).replace("torch.", ""))
                state[key] = tensor.to(dtype)
    missing = sorted(wanted - set(state) - maybe)
    if missing or unexpected:
        raise RuntimeError(f"Transformer weights under {where} do not match the model: "
                           f"missing={missing[:5]} ({len(missing)}), unexpected={sorted(unexpected)[:5]} ({len(unexpected)})")
    provenance = {"files": [Path(path).name for path in files], "variant": variant, "file_dtypes": sorted(file_dtypes)}
    return state, provenance


def diffusers_transformer_dir(path: str | Path) -> Path | None:
    """``path/transformer`` (a pipeline root) or ``path`` when it is a diffusers ``SD3Transformer2DModel`` directory,
    else None."""
    root = Path(path)
    for candidate in (root / TRANSFORMER_SUBDIR, root):
        config = candidate / "config.json"
        if candidate.is_dir() and config.is_file():
            if json.loads(config.read_text(encoding="utf-8")).get("_class_name") == "SD3Transformer2DModel":
                return candidate
    return None


def normalized_init_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """The ``SD3Transformer2DModel.__init__`` kwargs a config builds, with the class defaults filled in (tuples as
    lists; other keys dropped), so a config that spells out a default (CDM's turbo) equals one that omits it."""
    import inspect

    from diffusers import SD3Transformer2DModel

    values = {}
    for name, parameter in inspect.signature(SD3Transformer2DModel.__init__).parameters.items():
        if name == "self" or parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue
        value = config.get(name, parameter.default)
        values[name] = list(value) if isinstance(value, tuple) else value
    return values


def read_diffusers_transformer(directory: str | Path, *, base_config: Mapping[str, Any], parameters: Iterable[str],
                               buffers: Iterable[str], dtype: torch.dtype) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Full transformer weights of a diffusers directory whose architecture equals ``base_config``: every parameter,
    and the buffers when present (the caller takes missing ones from the base)."""
    directory = Path(directory)
    config = _read_json(directory / "config.json")
    if normalized_init_config(config) != normalized_init_config(base_config):
        raise ValueError(f"{directory} is not the base architecture: {normalized_init_config(config)} != "
                         f"{normalized_init_config(base_config)}")
    files, variant = weight_files(directory)
    buffers = set(buffers)
    state, provenance = read_state(files, set(parameters) | buffers, dtype=dtype, where=str(directory),
                                   variant=variant, optional=buffers)
    return state, {**provenance, "source": str(directory), "diffusers_version": config.get("_diffusers_version")}


def _transformers_dtype_kwargs(dtype: torch.dtype) -> dict[str, Any]:
    """``dtype=`` for transformers >= 4.56, else ``torch_dtype=`` (as diffusers' ``load_sub_model``)."""
    import transformers
    from packaging import version

    current = version.parse(version.parse(transformers.__version__).base_version)
    return {"dtype": dtype} if current >= version.parse("4.56.0") else {"torch_dtype": dtype}


def load_tokenizers(model_path: str | Path) -> tuple[Any, Any, Any]:
    from transformers import CLIPTokenizer, T5TokenizerFast

    root = Path(model_path)
    return (CLIPTokenizer.from_pretrained(str(root / TOKENIZER_SUBDIRS[0]), local_files_only=True),
            CLIPTokenizer.from_pretrained(str(root / TOKENIZER_SUBDIRS[1]), local_files_only=True),
            T5TokenizerFast.from_pretrained(str(root / TOKENIZER_SUBDIRS[2]), local_files_only=True))


def load_text_encoders(model_path: str | Path, *, dtype: torch.dtype,
                       device: torch.device | str | None = None) -> tuple[nn.Module, nn.Module, nn.Module]:
    """The three frozen encoders, loaded with the kwargs ``StableDiffusion3Pipeline.from_pretrained`` uses."""
    from transformers import CLIPTextModelWithProjection, T5EncoderModel

    root = Path(model_path)
    kwargs = dict(_transformers_dtype_kwargs(dtype), local_files_only=True, low_cpu_mem_usage=True)
    encoders = (CLIPTextModelWithProjection.from_pretrained(str(root / TEXT_ENCODER_SUBDIRS[0]), **kwargs),
                CLIPTextModelWithProjection.from_pretrained(str(root / TEXT_ENCODER_SUBDIRS[1]), **kwargs),
                T5EncoderModel.from_pretrained(str(root / TEXT_ENCODER_SUBDIRS[2]), **kwargs))
    for encoder in encoders:
        encoder.requires_grad_(False)
        encoder.eval()
        if device is not None:
            encoder.to(device)
    return encoders


def shard_t5(model: nn.Module, *, reshard_after_forward: bool = True) -> nn.Module:
    """FSDP2-shard the frozen T5 encoder in place: one unit per ``T5Block`` plus the root (shared embedding), in the
    encoder's own dtype, so sharding does not change what it computes."""
    from edmd.runtime.distributed import shard_module

    blocks = list(getattr(getattr(model, "encoder", None), "block", []) or [])
    if not blocks:
        raise RuntimeError("The T5 encoder has no encoder.block to shard; the upstream model changed.")
    dtype = next(model.parameters()).dtype
    return shard_module(model, blocks, reshard_after_forward=reshard_after_forward, param_dtype=dtype)


def load_vae(model_path: str | Path, *, dtype: torch.dtype, load_dtype: torch.dtype | None = None,
             device: torch.device | str | None = None) -> nn.Module:
    """``from_pretrained(torch_dtype=load_dtype)`` then ``.to(dtype)``: CDM loads its pipeline in bf16 and moves the
    VAE to fp32, so the fp32 weights hold bf16-rounded values."""
    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(str(Path(model_path) / VAE_SUBDIR), torch_dtype=load_dtype or dtype,
                                        local_files_only=True)
    if (load_dtype or dtype) != dtype:
        vae.to(dtype=dtype)
    vae.requires_grad_(False)
    vae.eval()
    return vae.to(device) if device is not None else vae


def assign_state(model: nn.Module, state: Mapping[str, torch.Tensor]) -> nn.Module:
    """Materialize a meta model from a full CPU state dict (strict, ``assign=True``)."""
    result = model.load_state_dict(dict(state), strict=True, assign=True)
    if result.missing_keys or result.unexpected_keys:  # pragma: no cover - strict=True raises first
        raise RuntimeError(f"State mismatch: {result}")
    leftover = [name for name, tensor in list(model.named_parameters()) + list(model.named_buffers()) if tensor.is_meta]
    if leftover:
        raise RuntimeError(f"Tensors left on the meta device: {leftover[:5]}")
    return model


__all__ = [
    "TEXT_ENCODER_SUBDIRS", "TOKENIZER_SUBDIRS", "TRANSFORMER_SUBDIR", "VAE_SUBDIR", "assign_state",
    "build_transformer_on_meta", "load_text_encoders", "load_tokenizers", "load_transformer_config", "load_vae",
    "load_vae_config", "non_persistent_buffers", "normalized_init_config", "read_diffusers_transformer",
    "read_state", "read_transformer_state", "shard_t5", "transformer_blocks", "transformer_weight_files", "vae_scale_factor",
    "weight_files", "diffusers_transformer_dir",
]

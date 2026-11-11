"""FLUX.2 klein base 4B / 9B adapter (``model.name: flux2_klein``, ``model.variant: 4b | 9b``).

``Flux2KleinPipeline`` (diffusers) is the specification of the schedule, prompt encoding, packing, position ids,
velocity call, noise and decode.

Options (``model.options``, all optional):

* ``max_sequence_length`` (512): prompt tokens after the chat template (padded / truncated);
* ``text_encoder_out_layers`` ([9, 18, 27]): Qwen3 hidden-state indices concatenated into the prompt embedding;
* ``dtype`` (``bfloat16``): compute dtype of the transformer (``compute_dtype``), storage dtype of the frozen base and
  load dtype of the text encoder / VAE / inference denoiser; ``float32`` is for CPU checks;
* ``noise_dtype`` (``bfloat16``, as the pipeline): the dtype ``sample_noise`` draws in;
* ``sampling_dtype`` (``bfloat16``, as the pipeline): ``sampling_state_dtype`` of evaluation rollouts;
* ``inference_lora`` (``unmerged_compute``): ``unmerged_compute``, ``peft_fp32`` or ``merged``, see
  ``ModelAdapter.load_inference_denoiser``.
"""

from __future__ import annotations

import copy
from functools import cached_property
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn

from ...config.core import expand_path
from ..base import (Geometry, ModelAdapter, area_geometry_table, check_geometry, check_inference_lora,
                    parse_schedule_family, register)
from . import components, latents, schedule, text
from .denoiser import predict_velocity as _predict_velocity
from .inference_lora import INFERENCE_LORA_MODES, use_peft_order
from .text import Flux2Conditioning

DTYPES: dict[str, torch.dtype] = {"bfloat16": torch.bfloat16, "float32": torch.float32}
DEFAULT_OPTIONS: dict[str, Any] = {"max_sequence_length": 512, "text_encoder_out_layers": (9, 18, 27),
                                   "dtype": "bfloat16", "noise_dtype": "bfloat16", "sampling_dtype": "bfloat16",
                                   "inference_lora": "unmerged_compute"}
DESCRIBE_STEPS = (1, 2, 4, 8)


def parse_options(options: Mapping[str, Any] | None) -> dict[str, Any]:
    options = dict(options or {})
    unknown = sorted(set(options) - set(DEFAULT_OPTIONS))
    if unknown:
        raise ValueError(f"Unknown flux2_klein option(s) {unknown}; allowed: {sorted(DEFAULT_OPTIONS)}")
    merged = {**DEFAULT_OPTIONS, **options}
    length = merged["max_sequence_length"]
    if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
        raise ValueError(f"options.max_sequence_length must be a positive integer, got {length!r}")
    layers = merged["text_encoder_out_layers"]
    if (not isinstance(layers, (list, tuple)) or not layers
            or any(isinstance(layer, bool) or not isinstance(layer, int) for layer in layers)):
        raise ValueError(f"options.text_encoder_out_layers must be a non-empty list of integers, got {layers!r}")
    for key in ("dtype", "noise_dtype", "sampling_dtype"):
        if merged[key] not in DTYPES:
            raise ValueError(f"options.{key} must be one of {sorted(DTYPES)}, got {merged[key]!r}")
    check_inference_lora(merged["inference_lora"], INFERENCE_LORA_MODES)
    return {"max_sequence_length": int(length), "text_encoder_out_layers": tuple(int(layer) for layer in layers),
            "dtype": str(merged["dtype"]), "noise_dtype": str(merged["noise_dtype"]),
            "sampling_dtype": str(merged["sampling_dtype"]), "inference_lora": str(merged["inference_lora"])}


@register
class Flux2KleinAdapter(ModelAdapter):
    name = "flux2_klein"
    default_lora_targets = components.DEFAULT_LORA_TARGETS

    def __init__(self, spec: Any):
        super().__init__(spec)
        if spec.name != self.name:
            raise ValueError(f"spec.name is {spec.name!r}, not {self.name!r}.")
        components.variant_signature(spec.variant)
        self.options = parse_options(getattr(spec, "options", None))
        self.dtype = DTYPES[self.options["dtype"]]
        self.compute_dtype = self.dtype   # edmd.runtime.roles: FSDP2 param_dtype; training loop: base state dtype
        self.noise_dtype = DTYPES[self.options["noise_dtype"]]
        self.sampling_state_dtype = DTYPES[self.options["sampling_dtype"]]   # evaluation.generate's rollout state
        self.inference_lora = self.options["inference_lora"]
        self.max_sequence_length = self.options["max_sequence_length"]
        self.text_layers = self.options["text_encoder_out_layers"]
        self.path = expand_path(spec.path)
        side = int(spec.resolution) // components.PIXELS_PER_TOKEN * components.PIXELS_PER_TOKEN
        if side <= 0:
            raise ValueError("resolution must be at least 16 pixels.")
        # views (``at``) have their own geometry and grids, and use the root's encoder, negative and VAE
        self._root = self
        self._set_geometry(Geometry(height=side, width=side))
        self._text_encoder: nn.Module | None = None
        self._tokenizer: Any = None
        self._encoder_device: torch.device | None = None
        self._negative: Flux2Conditioning | None = None
        self._vae: nn.Module | None = None

    def _set_geometry(self, geometry: Geometry) -> None:
        self._geometry = geometry
        self.token_height = geometry.height // components.PIXELS_PER_TOKEN
        self.token_width = geometry.width // components.PIXELS_PER_TOKEN
        self.image_seq_len = self.token_height * self.token_width
        self._grids: dict[int, Tensor] = {}

    # ------------------------------------------------------------------ checkpoint configs (read lazily, validated)
    @cached_property
    def model_index(self) -> dict[str, Any]:
        index = components.load_json(self.path / "model_index.json")
        components.check_model_index(index)
        return index

    @cached_property
    def transformer_config(self) -> dict[str, Any]:
        self.model_index
        config = components.load_json(self.path / components.TRANSFORMER_SUBDIR / "config.json")
        components.check_transformer_config(config, self.spec.variant)
        return config

    @cached_property
    def scheduler_config(self) -> dict[str, Any]:
        config = components.load_json(self.path / components.SCHEDULER_SUBDIR / "scheduler_config.json")
        schedule.check_scheduler_config(config)
        return config

    @cached_property
    def text_encoder_config(self) -> dict[str, Any]:
        config = components.load_json(self.path / components.TEXT_ENCODER_SUBDIR / "config.json")
        components.check_text_encoder_config(config, self.spec.variant, self.text_layers)
        return config

    @cached_property
    def base_keys(self) -> frozenset[str]:
        """Every parameter / persistent buffer name of the bare transformer (the checkpoint's key set)."""
        return frozenset(self.build_on_meta().state_dict().keys())

    # ------------------------------------------------------------------ static facts / schedule
    def geometry(self) -> Geometry:
        """This view's size; the adapter's own is ``resolution`` square (rounded down to a multiple of 16)."""
        return self._geometry

    def geometry_table(self) -> dict[str, dict[str, Geometry]]:
        """The generic ``1k`` / ``2k`` area tiers (``base.area_geometry_table``)."""
        return area_geometry_table()

    def at(self, geometry: Geometry) -> "Flux2KleinAdapter":
        """A view at ``geometry`` (multiples of 16 pixels): the same weights, encoder, VAE and options, with its own
        token grid, shift (it depends on the token count), grids, noise shape and decode size."""
        geometry = check_geometry(geometry, components.PIXELS_PER_TOKEN, model=self.name)
        if geometry == self._geometry:
            return self
        root = self._root
        if geometry == root._geometry:
            return root
        view = copy.copy(root)
        view._set_geometry(geometry)
        return view

    def native_shift_factor(self, num_steps: int) -> float:
        """``exp(mu(image_seq_len, num_steps))`` -- FLUX's shift depends on the sampler's step count."""
        self.scheduler_config
        return schedule.native_shift_factor(self.image_seq_len, num_steps)

    def student_grid(self, num_steps: int, family: str = "native", *, device: torch.device | None = None) -> Tensor:
        """Native: the real scheduler's grid, bit for bit the pipeline's; ``shift<F>`` / ``lightning``: base warp."""
        if parse_schedule_family(family) is not None:
            return super().student_grid(num_steps, family, device=device)
        if isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps < 1:
            raise ValueError("num_steps must be a positive integer.")
        if num_steps not in self._grids:
            self._grids[num_steps] = schedule.native_grid(self.scheduler_config, image_seq_len=self.image_seq_len,
                                                          num_steps=num_steps)
        grid = self._grids[num_steps].clone()
        return grid.to(device=device) if device is not None else grid

    def describe(self) -> dict[str, Any]:
        config = self.transformer_config
        encoder = self.text_encoder_config
        timestep = ("bf16(bf16(sigma*1000)/1000), as Flux2KleinPipeline" if self.dtype == torch.bfloat16
                    else "fp32(sigma*1000)/1000 (parity-test dtype)")
        return {
            "adapter": self.name,
            "variant": self.spec.variant,
            "model_path": str(self.path),
            "geometry": {"height": self.geometry().height, "width": self.geometry().width,
                         "latent_tokens": [self.token_height, self.token_width], "image_seq_len": self.image_seq_len},
            "schedule": {
                "native": "dynamic exponential shift F = exp(empirical_mu(image_seq_len, K)); grid through the "
                          "checkpoint FlowMatchEulerDiscreteScheduler with np.linspace(1, 1/K, K)",
                "empirical_mu": {k: schedule.compute_empirical_mu(self.image_seq_len, k) for k in DESCRIBE_STEPS},
                "native_shift_factor": {k: self.native_shift_factor(k) for k in DESCRIBE_STEPS},
                "native_grid": {k: self.student_grid(k).tolist() for k in DESCRIBE_STEPS},
            },
            "compute_dtype": self.options["dtype"],
            "noise": f"{self.options['noise_dtype']} randn [B, 128, h, w] (diffusers randn_tensor), packed, returned fp32",
            "sampling_state_dtype": self.options["sampling_dtype"],
            "inference_lora": self.inference_lora,
            "timestep_precision": timestep,
            "guidance": "no guidance embedding (klein base); CFG = uncond + w (cond - uncond), two forwards",
            "negative_prompt": self.spec.negative_prompt,
            "text_encoder": {"class": "Qwen3ForCausalLM", "hidden_size": encoder["hidden_size"],
                             "num_hidden_layers": encoder["num_hidden_layers"],
                             "tie_word_embeddings": bool(encoder.get("tie_word_embeddings", False)),
                             "out_layers": list(self.text_layers), "max_sequence_length": self.max_sequence_length},
            "transformer": {key: config[key] for key in ("num_layers", "num_single_layers", "num_attention_heads",
                                                         "attention_head_dim", "joint_attention_dim")},
            "fsdp_units": {"Flux2TransformerBlock": config["num_layers"],
                           "Flux2SingleTransformerBlock": config["num_single_layers"]},
            "default_lora_targets": list(self.default_lora_targets),
        }

    # ------------------------------------------------------------------ roles
    def build_on_meta(self) -> nn.Module:
        return components.build_transformer_on_meta(self.transformer_config, dtype=self.dtype)

    def fsdp_units(self, model: nn.Module) -> list[nn.Module]:
        return components.fsdp_units(model)

    def read_base_state(self, expected: set[str], *, dtype: torch.dtype) -> dict[str, Tensor]:
        return components.read_transformer_state(self.path, expected, model_keys=self.base_keys, dtype=dtype)

    def reset_derived(self, model: nn.Module, device: torch.device) -> None:
        """Nothing to rebuild: FLUX.2 computes its rope table and timestep sinusoid in every forward."""
        components.check_no_buffers(model)

    # ------------------------------------------------------------------ conditioning
    def load_conditioning(self, device: torch.device, *, shard: bool = False) -> None:
        """Load the frozen Qwen3 encoder (optionally FSDP2-sharded per decoder layer) and encode the negative now, not
        lazily: with a sharded encoder every forward is a collective."""
        self.text_encoder_config
        root = self._root
        device = torch.device(device)
        encoder = components.load_text_encoder(self.path, dtype=self.dtype, device=device)
        if shard:
            from ...runtime.distributed import shard_module

            shard_module(encoder, components.text_encoder_layers(encoder),
                         reshard_after_forward=bool(self.spec.reshard_after_forward), param_dtype=self.dtype)
        root._text_encoder = encoder
        root._tokenizer = components.load_tokenizer(self.path)
        root._encoder_device = device
        root._negative = self._encode([self.spec.negative_prompt])

    def _encode(self, prompts: Sequence[str]) -> Flux2Conditioning:
        root = self._root
        if root._text_encoder is None:
            raise RuntimeError("Call load_conditioning() before encoding prompts.")
        return text.encode_prompts(root._text_encoder, root._tokenizer, prompts,
                                   max_sequence_length=self.max_sequence_length, layers=self.text_layers,
                                   device=root._encoder_device, dtype=self.dtype)

    def encode(self, prompts: Sequence[str]) -> Flux2Conditioning:
        if isinstance(prompts, str):
            raise TypeError("encode takes a sequence of prompts, not a single string.")
        return self._encode(list(prompts)).at_grid(self.token_height, self.token_width)

    def encode_negative(self, batch_size: int) -> Flux2Conditioning:
        negative = self._root._negative
        if negative is None:
            raise RuntimeError("Call load_conditioning() before encode_negative().")
        return negative.expand(batch_size).at_grid(self.token_height, self.token_width)

    # ------------------------------------------------------------------ denoiser
    def predict_velocity(self, model: nn.Module, x: Tensor, sigma: Tensor, cond: Flux2Conditioning) -> Tensor:
        """At the token grid ``cond`` was encoded for, so the root adapter runs a view's batch too (training calls the
        root's ``predict_velocity`` for every geometry)."""
        grid = getattr(cond, "grid", None)
        height, width = grid if grid is not None else (self.token_height, self.token_width)
        return _predict_velocity(model, x, sigma, cond, height=height, width=width, dtype=self.dtype)

    # ------------------------------------------------------------------ latents
    def sample_noise(self, batch_size: int, *, device: torch.device,
                     generator: torch.Generator | Sequence[torch.Generator] | None = None) -> Tensor:
        """Packed ``[B, h*w, 128]`` float32 latents, drawn as ``[B, 128, h, w]`` in ``options.noise_dtype``."""
        return latents.sample_noise(batch_size, self.token_height, self.token_width, device=device,
                                    generator=generator, draw_dtype=self.noise_dtype)

    def load_decoder(self, device: torch.device) -> None:
        self._root._vae = components.load_vae(self.path, dtype=self.dtype, device=device)

    def decode(self, latents_: Tensor) -> Tensor:
        vae = self._root._vae
        if vae is None:
            raise RuntimeError("Call load_decoder() before decode().")
        return latents.decode(vae, latents_, self.token_height, self.token_width)

    # ------------------------------------------------------------------ inference / export
    def load_inference_denoiser(self, weights: Path | None, device: torch.device, *, role_train: str = "lora") -> nn.Module:
        """One unsharded eval-mode transformer on ``device`` (see ``ModelAdapter.load_inference_denoiser``); key sets
        are checked strictly against the architecture and the LoRA targets."""
        from ...runtime.checkpoint import read_weights
        from ...runtime.lora import LoraSpec, inject_lora, lora_parameter_names, merge_lora_into_base

        if role_train not in ("lora", "full"):
            raise ValueError(f"role_train must be 'lora' or 'full', got {role_train!r}")
        model = self.build_on_meta()
        if weights is None:
            model = components.load_state_into_meta(model, self.read_base_state(set(self.base_keys), dtype=self.dtype),
                                                    device=device)
        else:
            tensors, metadata = read_weights(weights)
            self._check_weights_metadata(metadata, role_train, weights)
            if role_train == "lora":
                identity = metadata["lora"]
                spec = LoraSpec(rank=int(identity["rank"]), alpha=float(identity["alpha"]),
                                dropout=float(identity["dropout"]), init=str(identity["init"]),
                                target_modules=tuple(identity["target_modules"]))
                mode = self.inference_lora
                adapter_dtype = self.dtype if mode == "unmerged_compute" else torch.float32
                targets = inject_lora(model, spec, dtype=adapter_dtype)
                adapter_keys = set(lora_parameter_names(model))
                if set(tensors) != adapter_keys:
                    raise ValueError(f"{weights}: adapter tensors do not match the LoRA targets "
                                     f"(missing {sorted(adapter_keys - set(tensors))[:3]}, "
                                     f"unexpected {sorted(set(tensors) - adapter_keys)[:3]}).")
                state = self.read_base_state(set(self.base_keys), dtype=self.dtype)
                state.update({key: value.to(adapter_dtype) for key, value in tensors.items()})
                model = components.load_state_into_meta(model, state, device="cpu")
                if mode == "merged" and merge_lora_into_base(model) != len(targets):
                    raise RuntimeError("Not every LoRA module was merged.")
                if mode == "peft_fp32" and use_peft_order(model) != len(targets):
                    raise RuntimeError("Not every LoRA module was switched to the PEFT cast order.")
                model = model.to(device)
            else:
                if set(tensors) != set(self.base_keys):
                    raise ValueError(f"{weights}: full weights do not match the transformer "
                                     f"(missing {sorted(set(self.base_keys) - set(tensors))[:3]}, "
                                     f"unexpected {sorted(set(tensors) - set(self.base_keys))[:3]}).")
                model = components.load_state_into_meta(model, {key: value.to(self.dtype) for key, value in tensors.items()},
                                                        device=device)
        model.requires_grad_(False)
        model.eval()
        return model

    def _check_weights_metadata(self, metadata: Mapping[str, Any], role_train: str, path: Any) -> None:
        train = metadata.get("train")
        if train is not None and train != role_train:
            raise ValueError(f"{path} holds a {train!r} role, loaded as {role_train!r}.")
        if role_train == "lora" and not isinstance(metadata.get("lora"), Mapping):
            raise ValueError(f"{path}: a LoRA weights file needs its 'lora' identity in the JSON sidecar.")
        adapter = metadata.get("adapter")
        if adapter is not None and adapter != self.name:
            raise ValueError(f"{path} was trained with adapter {adapter!r}, not {self.name!r}.")


__all__ = ["DEFAULT_OPTIONS", "DTYPES", "Flux2KleinAdapter", "INFERENCE_LORA_MODES", "parse_options"]

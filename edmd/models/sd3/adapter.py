"""Stable Diffusion 3 Medium adapter (``sd3_medium``): the model CDM's official code and turbo weights use.

``StableDiffusion3Pipeline`` (diffusers) is the specification:

* **Schedule.** Static shift 3.0 in the scheduler's float32 arithmetic. The student grid is the scheduler's sigmas for
  ``sigmas = linspace(1, 1/K, K)`` (CDM's turbo inference, ``[1, .9, .75, .5, 0]`` at K = 4), not the pipeline's
  no-``sigmas`` default, which :meth:`SD3MediumAdapter.pipeline_grid` keeps for teacher references.
* **Timestep.** ``sigma * 1000`` in float32, the pipeline's ``scheduler.timesteps``; the transformer takes the
  sinusoid of the float32 value.
* **Latents.** ``[B, 16, H/8, W/8]`` (the transformer patchifies), cast to bf16 at the call; outputs are float32.
* **Conditioning.** Exactly ``encode_prompt`` (:mod:`edmd.models.sd3.text`); the negative is ``spec.negative_prompt``
  ("" by default, as CDM).
* **CFG.** Linear; :meth:`predict_velocity_pair` evaluates ``[uncond; cond]`` in one batched forward, as the pipeline.
* **Evaluation** follows CDM's evaluation: a bf16 pipeline (fp32 VAE) under ``torch.autocast(bf16)``, bf16 noise from
  a device generator, bf16 latents between steps (options ``sampling_dtype``, ``noise_dtype``, ``sampling_autocast``,
  ``vae_dtype``). Training never autocasts (neither does CDM's FSDP training) and its rollouts stay float32.
"""

from __future__ import annotations

import contextlib
import copy
from functools import cached_property
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from ...config.core import expand_path
from ..base import Geometry, ModelAdapter, area_geometry_table, check_geometry, check_inference_lora, register
from . import components, inference_lora, schedule
from .text import SD3Conditioning, SD3PromptEncoder

#: LoRA default: every attention projection and feed-forward linear of both streams of every joint block (not the
#: AdaLN modulation, the embedders or ``proj_out``, as for FLUX and Qwen). The last block has no ``to_add_out`` /
#: ``ff_context`` (``context_pre_only``): 24 * 12 - 3 = 285 linears.
DEFAULT_LORA_TARGETS: tuple[str, ...] = (
    "transformer_blocks.*.attn.to_q",
    "transformer_blocks.*.attn.to_k",
    "transformer_blocks.*.attn.to_v",
    "transformer_blocks.*.attn.to_out.0",
    "transformer_blocks.*.attn.add_q_proj",
    "transformer_blocks.*.attn.add_k_proj",
    "transformer_blocks.*.attn.add_v_proj",
    "transformer_blocks.*.attn.to_add_out",
    "transformer_blocks.*.ff.net.0.proj",
    "transformer_blocks.*.ff.net.2",
    "transformer_blocks.*.ff_context.net.0.proj",
    "transformer_blocks.*.ff_context.net.2",
)

DTYPES: dict[str, torch.dtype] = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

#: ``model.options`` of this adapter (unknown keys raise); the evaluation defaults follow CDM's evaluation.
OPTION_DEFAULTS: dict[str, Any] = {
    "max_sequence_length": 256,     # T5 tokens; the pipeline's default and CDM's
    "clip_skip": None,              # None = penultimate CLIP layer, as the pipeline / CDM
    "compute_dtype": "bfloat16",    # dtype of latents / embeddings fed to the transformer (training: bf16 only)
    "text_encoder_dtype": "bfloat16",
    "vae_dtype": "float32",         # CDM's evaluation moves the VAE to float32...
    "vae_load_dtype": "bfloat16",   # ...after loading it in bf16: fp32 weights, bf16 values
    "noise_dtype": "bfloat16",      # randn dtype (the bf16 pipeline draws in bf16)
    "sampling_dtype": "bfloat16",   # evaluation latent state (the bf16 pipeline's)
    "sampling_autocast": "bfloat16",  # autocast of evaluation encode / transformer / VAE, null = none
    # LoRA numerics of load_inference_denoiser: unmerged_compute | peft_fp32 | merged
    "inference_lora": "unmerged_compute",
}
AUTOCAST_DTYPES = ("bfloat16", "float16")


def parse_options(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    raw = dict(raw or {})
    unknown = sorted(set(raw) - set(OPTION_DEFAULTS))
    if unknown:
        raise ValueError(f"model.options: unknown key(s) {unknown} for sd3_medium; allowed: {sorted(OPTION_DEFAULTS)}")
    options = {**OPTION_DEFAULTS, **raw}
    length = options["max_sequence_length"]
    if isinstance(length, bool) or not isinstance(length, int) or not 1 <= length <= 512:
        raise ValueError("model.options.max_sequence_length must be an integer in [1, 512].")
    skip = options["clip_skip"]
    if skip is not None and (isinstance(skip, bool) or not isinstance(skip, int) or skip < 0):
        raise ValueError("model.options.clip_skip must be null or a non-negative integer.")
    for key in ("compute_dtype", "text_encoder_dtype", "vae_dtype", "vae_load_dtype", "noise_dtype", "sampling_dtype"):
        if options[key] not in DTYPES:
            raise ValueError(f"model.options.{key} must be one of {sorted(DTYPES)}, got {options[key]!r}.")
    if options["sampling_autocast"] is not None and options["sampling_autocast"] not in AUTOCAST_DTYPES:
        raise ValueError(f"model.options.sampling_autocast must be null or one of {AUTOCAST_DTYPES}, "
                         f"got {options['sampling_autocast']!r}.")
    check_inference_lora(options["inference_lora"], inference_lora.MODES)
    return options


@register
class SD3MediumAdapter(ModelAdapter):
    """SD3-Medium: 2B MMDiT, CLIP-L + CLIP-G + T5-XXL, 16-channel VAE, static shift 3."""

    name = "sd3_medium"
    default_lora_targets = DEFAULT_LORA_TARGETS

    def __init__(self, spec: Any):
        super().__init__(spec)
        if getattr(spec, "name", self.name) != self.name:
            raise ValueError(f"SD3MediumAdapter got a spec for {spec.name!r}.")
        variant = getattr(spec, "variant", "medium")
        if variant != "medium":
            raise ValueError(f"sd3_medium supports variant 'medium' only, got {variant!r}.")
        self.model_path = expand_path(spec.path)
        self.options = parse_options(getattr(spec, "options", None))
        self.compute_dtype = DTYPES[self.options["compute_dtype"]]
        self.text_encoder_dtype = DTYPES[self.options["text_encoder_dtype"]]
        self.vae_dtype = DTYPES[self.options["vae_dtype"]]
        self.vae_load_dtype = DTYPES[self.options["vae_load_dtype"]]
        self.noise_dtype = DTYPES[self.options["noise_dtype"]]
        self.sampling_state_dtype = DTYPES[self.options["sampling_dtype"]]
        autocast = self.options["sampling_autocast"]
        self.sampling_autocast_dtype: torch.dtype | None = None if autocast is None else DTYPES[autocast]
        self.prompt_encoder: SD3PromptEncoder | None = None
        self.text_encoder_sharded = False
        self.vae: nn.Module | None = None
        self.weights_provenance: dict[str, Any] | None = None
        self._negatives: dict[int, SD3Conditioning] = {}
        # views (``at``) have their own geometry and use the root's encoders, negatives and VAE
        self._root = self
        self._geometry: Geometry | None = None

    # ------------------------------------------------------------------ checkpoint configs (read lazily)
    @cached_property
    def scheduler_config(self) -> dict[str, Any]:
        return schedule.load_scheduler_config(self.model_path)

    @cached_property
    def transformer_config(self) -> dict[str, Any]:
        config = components.load_transformer_config(self.model_path)
        vae_channels = int(self.vae_config.get("latent_channels", config["in_channels"]))
        if int(config["in_channels"]) != vae_channels:
            raise ValueError(f"transformer in_channels {config['in_channels']} != VAE latent_channels {vae_channels}.")
        return config

    @cached_property
    def vae_config(self) -> dict[str, Any]:
        return components.load_vae_config(self.model_path)

    @property
    def shift(self) -> float:
        return float(self.scheduler_config.get("shift", 1.0))

    @property
    def num_train_timesteps(self) -> int:
        return int(self.scheduler_config.get("num_train_timesteps", 1000))

    @property
    def vae_scale_factor(self) -> int:
        return components.vae_scale_factor(self.vae_config)

    @property
    def patch_size(self) -> int:
        return int(self.transformer_config["patch_size"])

    @property
    def latent_channels(self) -> int:
        return int(self.transformer_config["in_channels"])

    # ------------------------------------------------------------------ static facts / schedule
    def geometry(self) -> Geometry:
        """This view's size; the adapter's own is ``resolution`` square."""
        if self._geometry is not None:
            return self._geometry
        resolution = int(self.spec.resolution)
        return self._check_size(Geometry(height=resolution, width=resolution))

    def _check_size(self, geometry: Geometry) -> Geometry:
        """Sides divisible by the VAE scale times the patch size (the pipeline raises rather than rounding) and within
        the position embedding (``pos_embed_max_size`` patches per side)."""
        multiple = self.vae_scale_factor * self.patch_size
        geometry = check_geometry(geometry, multiple, model=self.name)
        max_patches = self.transformer_config.get("pos_embed_max_size")
        if max_patches is not None and max(geometry.height, geometry.width) // multiple > int(max_patches):
            raise ValueError(f"{geometry.width}x{geometry.height} exceeds pos_embed_max_size={max_patches} patches "
                             f"({int(max_patches) * multiple} pixels per side).")
        return geometry

    def geometry_table(self) -> dict[str, dict[str, Geometry]]:
        """The generic ``1k`` / ``2k`` area tiers (``base.area_geometry_table``)."""
        return area_geometry_table()

    def at(self, geometry: Geometry) -> "SD3MediumAdapter":
        """A view at ``geometry``: the same weights, encoders, VAE and options, with its own latent shape (noise and
        decode). The static shift keeps the grid."""
        geometry = self._check_size(geometry)
        if geometry == self.geometry():
            return self
        root = self._root
        if geometry == root.geometry():
            return root
        view = copy.copy(root)
        view._geometry = geometry
        return view

    def latent_shape(self) -> tuple[int, int, int]:
        geometry = self.geometry()
        return (self.latent_channels, geometry.height // self.vae_scale_factor, geometry.width // self.vae_scale_factor)

    def native_shift_factor(self, num_steps: int) -> float:
        if isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps < 1:
            raise ValueError("num_steps must be a positive integer.")
        return self.shift

    def warp(self, raw: Tensor, num_steps: int, family: str = "native") -> Tensor:
        """The scheduler's own float32 static-shift expression (bitwise ``set_timesteps(sigmas=raw)``)."""
        return schedule.static_shift_warp(raw, self.shift_factor(num_steps, family))

    def student_grid(self, num_steps: int, family: str = "native", *, device: torch.device | None = None) -> Tensor:
        """Scheduler sigmas for ``sigmas = linspace(1, 1/K, K)`` (CDM's turbo grid), run through diffusers."""
        grid = schedule.custom_sigma_grid(self.scheduler_config, num_steps, shift=self.shift_factor(num_steps, family))
        return grid.to(device=device) if device is not None else grid

    def raw_grid(self, num_steps: int, family: str = "native", *, device: torch.device | None = None) -> Tensor:
        """float32 ``np.linspace(1, 1/K, K)`` (what the scheduler receives) + 0, so ``warp(raw_grid) == student_grid``
        bit for bit (``torch.linspace(1, 0, K+1)`` is one ulp off for some K)."""
        raw = torch.from_numpy(schedule.raw_grid(num_steps).astype(np.float32))
        grid = torch.cat([raw, torch.zeros(1)])
        return grid.to(device=device) if device is not None else grid

    def pipeline_grid(self, num_steps: int, *, device: torch.device | None = None) -> Tensor:
        """``StableDiffusion3Pipeline(num_inference_steps=K)``'s own grid (no ``sigmas``): teacher references only."""
        grid = schedule.pipeline_default_grid(self.scheduler_config, num_steps)
        return grid.to(device=device) if device is not None else grid

    @staticmethod
    def batch_sigma(sigma: Tensor | float, batch_size: int, *, device: torch.device) -> Tensor:
        """``[B]`` float32 per-sample sigmas (a scalar or one-element tensor is broadcast)."""
        if isinstance(sigma, Tensor):
            values = sigma.detach().to(device=device, dtype=torch.float32).reshape(-1)
            if values.numel() == 1:
                values = values.expand(batch_size)
            if values.numel() != batch_size:
                raise ValueError(f"Expected {batch_size} sigmas, got {values.numel()}.")
            return values
        return torch.full((batch_size,), float(sigma), device=device, dtype=torch.float32)

    def timesteps(self, sigma: Tensor | float, batch_size: int, *, device: torch.device) -> Tensor:
        """``[B]`` float32 ``sigma * num_train_timesteps`` -- the pipeline's ``scheduler.timesteps`` values."""
        return self.batch_sigma(sigma, batch_size, device=device) * self.num_train_timesteps

    def describe(self) -> dict[str, Any]:
        geometry = self.geometry()
        c, h, w = self.latent_shape()
        tokens = (h // self.patch_size) * (w // self.patch_size)
        return {
            "adapter": self.name,
            "variant": getattr(self.spec, "variant", "medium"),
            "model_path": str(self.model_path),
            "resolution": [geometry.height, geometry.width],
            "latent_shape": [c, h, w],
            "vae_scale_factor": self.vae_scale_factor,
            "patch_size": self.patch_size,
            "image_tokens": tokens,
            "text_tokens": 77 + int(self.options["max_sequence_length"]),
            "shift_factor": self.shift,
            "scheduler": {key: self.scheduler_config.get(key) for key in ("shift", "num_train_timesteps")},
            "timestep": "float32 sigma * num_train_timesteps (the pipeline's scheduler.timesteps); sinusoid in float32",
            "student_grid_4": [float(v) for v in self.student_grid(4)],
            "pipeline_default_grid_4": [float(v) for v in self.pipeline_grid(4)],
            "compute_dtype": self.options["compute_dtype"],
            "text_encoder_dtype": self.options["text_encoder_dtype"],
            "vae_dtype": self.options["vae_dtype"],
            "vae_load_dtype": self.options["vae_load_dtype"],
            "noise_dtype": self.options["noise_dtype"],
            "sampling_dtype": self.options["sampling_dtype"],
            "sampling_autocast": self.options["sampling_autocast"],
            "inference_lora": self.options["inference_lora"],
            "evaluation": "CDM scripts/eval.py numerics: bf16 noise from a device generator, bf16 latent state, "
                          "encode / transformer / VAE under autocast(sampling_autocast), fp32 VAE weights, CFG 1",
            "max_sequence_length": int(self.options["max_sequence_length"]),
            "clip_skip": self.options["clip_skip"],
            "negative_prompt": getattr(self.spec, "negative_prompt", ""),
            "text_encoder_sharded": self._root.text_encoder_sharded,
            "cfg": "linear u + w (c - u); predict_velocity_pair batches [uncond; cond]",
            "vae": {"scaling_factor": self.vae_config.get("scaling_factor"),
                    "shift_factor": self.vae_config.get("shift_factor")},
            "weights": self._root.weights_provenance,
            "lora_default_targets": list(self.default_lora_targets),
        }

    # ------------------------------------------------------------------ roles
    def build_on_meta(self) -> nn.Module:
        return components.build_transformer_on_meta(self.transformer_config)

    def fsdp_units(self, model: nn.Module) -> list[nn.Module]:
        return components.transformer_blocks(model)

    def read_base_state(self, expected: set[str], *, dtype: torch.dtype) -> dict[str, Tensor]:
        state, provenance = components.read_transformer_state(self.model_path, expected, dtype=dtype)
        self.weights_provenance = {**provenance, "dtype": str(dtype).replace("torch.", "")}
        return state

    def reset_derived(self, model: nn.Module, device: torch.device) -> None:
        """Nothing to rebuild: SD3-M's only buffer (``pos_embed.pos_embed``) is persistent and in the checkpoint."""
        derived = components.non_persistent_buffers(model)
        if derived:
            raise RuntimeError(f"SD3 transformer has non-persistent buffers {derived} that the state dict does not "
                               "restore after to_empty; teach reset_derived to rebuild them.")

    # ------------------------------------------------------------------ conditioning
    def load_conditioning(self, device: torch.device, *, shard: bool = False) -> None:
        tokenizers = components.load_tokenizers(self.model_path)
        encoders = components.load_text_encoders(self.model_path, dtype=self.text_encoder_dtype, device=device)
        if shard:
            import torch.distributed as dist

            if not dist.is_initialized():
                raise RuntimeError("load_conditioning(shard=True) needs an initialized process group.")
            components.shard_t5(encoders[2], reshard_after_forward=bool(getattr(self.spec, "reshard_after_forward", True)))
        root = self._root
        root.prompt_encoder = SD3PromptEncoder(tokenizers, encoders,
                                               max_sequence_length=int(self.options["max_sequence_length"]),
                                               clip_skip=self.options["clip_skip"])
        root.text_encoder_sharded = bool(shard)
        root._negatives.clear()

    def _text(self) -> SD3PromptEncoder:
        encoder = self._root.prompt_encoder
        if encoder is None:
            raise RuntimeError("Call load_conditioning() before encoding prompts.")
        return encoder

    def encode(self, prompts: Sequence[str]) -> SD3Conditioning:
        return self._text().encode(prompts)

    def encode_negative(self, batch_size: int) -> SD3Conditioning:
        """The pipeline's negative encoding of ``[negative_prompt] * batch_size``, cached per batch size. With a sharded
        T5 every rank must request the same batch sizes in the same order (true in training)."""
        negatives = self._root._negatives      # one cache for every view: the conditioning has no geometry
        cached = negatives.get(int(batch_size))
        if cached is None:
            cached = self._text().encode_negative(getattr(self.spec, "negative_prompt", ""), int(batch_size))
            negatives[int(batch_size)] = cached
        return cached

    # ------------------------------------------------------------------ denoiser
    def _check_latents(self, x: Tensor) -> None:
        if x.ndim != 4 or x.shape[1] != self.latent_channels:
            raise ValueError(f"latents must be [B, {self.latent_channels}, H/8, W/8], got {tuple(x.shape)}.")
        if x.shape[-2] % self.patch_size or x.shape[-1] % self.patch_size:
            raise ValueError(f"latent height/width must be divisible by the patch size {self.patch_size}.")

    def predict_velocity(self, model: nn.Module, x: Tensor, sigma: Tensor, cond: SD3Conditioning) -> Tensor:
        """``v = eps - x0`` in float32, feeding the transformer exactly like ``StableDiffusion3Pipeline.__call__``."""
        self._check_latents(x)
        batch = int(x.shape[0])
        if cond.batch_size != batch:
            raise ValueError(f"latents have batch {batch}, the conditioning has {cond.batch_size}.")
        timestep = self.timesteps(sigma, batch, device=x.device)
        output = model(
            hidden_states=x.to(self.compute_dtype),
            timestep=timestep,
            encoder_hidden_states=cond.prompt_embeds.to(device=x.device, dtype=self.compute_dtype),
            pooled_projections=cond.pooled_prompt_embeds.to(device=x.device, dtype=self.compute_dtype),
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]
        return output.float()

    def predict_velocity_pair(self, model: nn.Module, x: Tensor, sigma: Tensor, cond: SD3Conditioning,
                              uncond: SD3Conditioning) -> tuple[Tensor, Tensor]:
        """``(v_cond, v_uncond)`` from one forward on ``[uncond; cond]`` (pipeline / CDM batching, uncond first)."""
        batch = int(x.shape[0])
        if uncond.batch_size != batch:
            raise ValueError(f"latents have batch {batch}, the negative conditioning has {uncond.batch_size}.")
        sigmas = self.batch_sigma(sigma, batch, device=x.device)
        both = self.predict_velocity(model, torch.cat([x, x], dim=0), torch.cat([sigmas, sigmas], dim=0),
                                     SD3Conditioning.cat([uncond, cond]))
        v_uncond, v_cond = both.chunk(2, dim=0)
        return v_cond, v_uncond

    # ------------------------------------------------------------------ evaluation (CDM's evaluation numerics)
    def sampling_context(self, device: torch.device | str | None = None) -> contextlib.AbstractContextManager:
        """``torch.autocast(device, sampling_autocast)``, which CDM's evaluation wraps its whole pipeline call in; a
        null context when ``sampling_autocast`` is null or ``device`` is None."""
        if self.sampling_autocast_dtype is None or device is None:
            return contextlib.nullcontext()
        return torch.autocast(device_type=torch.device(device).type, dtype=self.sampling_autocast_dtype)

    def sampling_velocity(self, model: nn.Module, cond: SD3Conditioning) -> Callable[[Tensor, Tensor], Tensor]:
        """``(x, sigma) -> v`` for an evaluation rollout: ``predict_velocity`` under :meth:`sampling_context`."""
        def velocity(x: Tensor, sigma: Tensor) -> Tensor:
            with self.sampling_context(x.device):
                return self.predict_velocity(model, x, sigma, cond)
        return velocity

    def sampling_velocity_pair(self, model: nn.Module, cond: SD3Conditioning,
                               uncond: SD3Conditioning) -> Callable[[Tensor, Tensor], tuple[Tensor, Tensor]]:
        """``(x, sigma) -> (v_cond, v_uncond)`` for a guided evaluation rollout (teacher references): the pipeline's
        one batched ``[uncond; cond]`` forward, under :meth:`sampling_context`."""
        def velocity(x: Tensor, sigma: Tensor) -> tuple[Tensor, Tensor]:
            with self.sampling_context(x.device):
                return self.predict_velocity_pair(model, x, sigma, cond, uncond)
        return velocity

    # ------------------------------------------------------------------ latents
    def sample_noise(self, batch_size: int, *, device: torch.device,
                     generator: torch.Generator | Sequence[torch.Generator] | None = None) -> Tensor:
        """``prepare_latents``: ``randn_tensor([B, 16, H/8, W/8])``, drawn in ``noise_dtype``, returned as float32."""
        from diffusers.utils.torch_utils import randn_tensor

        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        if isinstance(generator, (list, tuple)):
            generator = list(generator)
            if len(generator) != batch_size:
                raise ValueError(f"Got {len(generator)} generators for a batch of {batch_size}.")
        geometry = self.geometry()
        shape = (batch_size, self.latent_channels, int(geometry.height) // self.vae_scale_factor,
                 int(geometry.width) // self.vae_scale_factor)
        noise = randn_tensor(shape, generator=generator, device=torch.device(device), dtype=self.noise_dtype)
        return noise.to(torch.float32)

    def load_decoder(self, device: torch.device) -> None:
        """The VAE loaded in ``vae_load_dtype`` then cast to ``vae_dtype`` (CDM: bf16 pipeline, VAE moved to fp32)."""
        self._root.vae = components.load_vae(self.model_path, dtype=self.vae_dtype, load_dtype=self.vae_load_dtype,
                                             device=device)

    @torch.no_grad()
    def decode(self, latents: Tensor) -> Tensor:
        """The pipeline's tail, as CDM's evaluation runs it: unscale in the latents' own dtype, VAE decode under
        :meth:`sampling_context`, ``(x * 0.5 + 0.5).clamp(0, 1)``, then float32."""
        vae = self._root.vae
        if vae is None:
            raise RuntimeError("Call load_decoder() before decode().")
        parameter = next(vae.parameters())
        scaled = latents.to(parameter.device) / vae.config.scaling_factor + vae.config.shift_factor
        with self.sampling_context(parameter.device):
            if self.sampling_autocast_dtype is None:
                scaled = scaled.to(dtype=parameter.dtype)      # no autocast: the VAE takes its own dtype
            image = vae.decode(scaled, return_dict=False)[0]
            image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.float()

    # ------------------------------------------------------------------ inference / export
    def load_inference_denoiser(self, weights: Path | None, device: torch.device,
                                *, role_train: str = "lora") -> nn.Module:
        """A single-device ``compute_dtype`` transformer (see ``ModelAdapter.load_inference_denoiser``). ``"full"`` also
        takes a diffusers SD3 transformer directory or pipeline root with the base architecture (e.g. CDM's turbo
        weights); buffers the weights lack come from the base."""
        from ...runtime.checkpoint import read_weights
        from ...runtime.lora import inject_lora, lora_parameter_names, merge_lora_into_base

        if role_train not in ("lora", "full"):
            raise ValueError(f"role_train must be 'lora' or 'full', got {role_train!r}.")
        model = self.build_on_meta()
        expected = set(model.state_dict().keys())
        parameters = {name for name, _ in model.named_parameters()}
        state, provenance = components.read_transformer_state(self.model_path, expected, dtype=self.compute_dtype)
        self.weights_provenance = {**provenance, "dtype": self.options["compute_dtype"]}
        if weights is None:
            components.assign_state(model, state)
        elif Path(weights).is_dir():
            directory = components.diffusers_transformer_dir(weights)
            if directory is None:
                raise ValueError(f"{weights} is neither a diffusers SD3 transformer directory nor a pipeline root with "
                                 "one; for eDMD exports pass weights/<role>.safetensors.")
            if role_train != "full":
                raise ValueError(f"{weights} is a diffusers transformer (full weights); use role_train='full'.")
            tensors, source = components.read_diffusers_transformer(
                directory, base_config=self.transformer_config, parameters=parameters, buffers=expected - parameters,
                dtype=self.compute_dtype)
            components.assign_state(model, {**state, **tensors})
            self.weights_provenance["full_weights"] = source
        else:
            path = Path(weights)
            tensors, metadata = read_weights(path)
            recorded = metadata.get("train")
            if recorded is not None and recorded != role_train:
                raise ValueError(f"{path} holds a {recorded!r} role, loaded as {role_train!r}.")
            if role_train == "lora":
                identity = metadata.get("lora")
                if not isinstance(identity, Mapping):
                    raise ValueError(f"{path} carries no LoRA settings (metadata['lora']).")
                spec = inference_lora.spec_from_metadata(identity)
                mode = self.options["inference_lora"]
                if mode == "merged":
                    components.assign_state(model, {key: value.float() for key, value in state.items()})
                    inject_lora(model, spec, dtype=torch.float32)
                    names = set(lora_parameter_names(model))
                    if set(tensors) != names:
                        raise ValueError(f"{path} does not match the LoRA targets: "
                                         f"missing={sorted(names - set(tensors))[:3]} "
                                         f"unexpected={sorted(set(tensors) - names)[:3]}")
                    named = dict(model.named_parameters())
                    with torch.no_grad():
                        for name in names:
                            named[name].copy_(tensors[name].float())
                    merge_lora_into_base(model)
                    model.to(self.compute_dtype)
                else:
                    components.assign_state(model, state)
                    inference_lora.attach_unmerged(model, spec, tensors, mode=mode, compute_dtype=self.compute_dtype)
                self.weights_provenance["lora"] = {"mode": mode, **dict(identity)}
            else:
                # exports hold the parameters (maybe also the frozen pos_embed buffer); all cast to compute_dtype
                if not parameters <= set(tensors) <= expected:
                    raise ValueError(f"{path} is not a full SD3 transformer export: "
                                     f"missing={sorted(parameters - set(tensors))[:3]} "
                                     f"unexpected={sorted(set(tensors) - expected)[:3]}")
                components.assign_state(model, {**state, **{key: value.to(self.compute_dtype)
                                                            for key, value in tensors.items()}})
        model.requires_grad_(False)
        model.eval()
        return model.to(device)

    def read_role_weights(self, path: str | Path) -> tuple[dict[str, Tensor], dict[str, Any]]:
        """Initial weights of a trainable role: an ``edmd_weights_v1`` file, or a diffusers SD3 transformer directory /
        pipeline root (CDM's turbo weights) with the base architecture, read as a full role in float32."""
        from ...runtime.checkpoint import read_weights

        path = Path(path)
        if not path.is_dir():
            return read_weights(path)
        directory = components.diffusers_transformer_dir(path)
        if directory is None:
            raise ValueError(f"{path} is neither an edmd_weights_v1 file nor a diffusers SD3 transformer directory.")
        model = self.build_on_meta()
        expected = set(model.state_dict().keys())
        parameters = {name for name, _ in model.named_parameters()}
        tensors, source = components.read_diffusers_transformer(
            directory, base_config=self.transformer_config, parameters=parameters, buffers=expected - parameters,
            dtype=torch.float32)
        absent = expected - set(tensors)
        if absent:
            base, _ = components.read_transformer_state(self.model_path, expected, dtype=torch.float32)
            tensors.update({key: base[key] for key in absent})
        for key in expected - parameters:
            # frozen buffers get the base state's precision (compute dtype), so pos_embed matches the other roles'
            tensors[key] = tensors[key].to(self.compute_dtype).float()
        return tensors, {"train": "full", "adapter": self.name, "format": "diffusers", **source}


__all__ = ["DEFAULT_LORA_TARGETS", "DTYPES", "OPTION_DEFAULTS", "SD3MediumAdapter", "parse_options"]

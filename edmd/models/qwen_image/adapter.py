"""Qwen-Image-2.1 adapter (``qwen_image_21``): 7B single-stream DiT, Qwen3-VL 8B text encoder, 64-channel RGBA VAE.

``QwenImage21Pipeline`` (diffusers main, 0.41.0.dev0) is the specification. Needs diffusers with the ``QwenImage21*``
classes and transformers with ``Qwen3VLForConditionalGeneration`` (tested with 5.17), the environment of
``requirements/qwen.txt``; both are imported lazily.

The student schedule is ``F = exp(mu(latent tokens))`` without the pipeline's terminal stretch (``schedule.py``).
``at(geometry)`` is a view sharing the weights, encoder and VAE; ``predict_velocity`` reads the image shape from the
conditioning, so the roles never need a view.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
from torch import Tensor, nn

from ...config.core import expand_path
from ..base import Geometry, ModelAdapter, check_inference_lora, parse_schedule_family, register
from . import components, layout, npu_compat, resolutions, schedule
from .inference_lora import INFERENCE_LORA_MODES, use_adapter_rounding
from .layout import QwenCondition

OPTION_DEFAULTS: dict[str, Any] = {
    "dtype": "bfloat16",       # compute dtype of the transformer inputs / frozen weights (float32: CPU tests)
    "alpha": "drop",           # decode: drop alpha | white (composite onto white)
    "kv_cache": True,          # sampling_velocity uses the pipeline's exact prefix KV cache
    "vae_tiling": "auto",      # auto = tile above UNTILED_DECODE_MAX_PIXELS; true | false
    "noise_dtype": "float32",  # draw dtype: float32 | bfloat16 (the stock pipeline's); returned fp32
    # LoRA numerics of load_inference_denoiser: unmerged_compute (the training forward) | unmerged_fp32 | merged
    "inference_lora": "unmerged_compute",
}
_DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32}
#: ``vae_tiling: auto`` tiles the decode above this many pixels, the largest 1K-tier area (1184 x 896): the 1K tier
#: decodes untiled, the 2K tier tiled. The reason is memory (an untiled 2K decode peaks at 27-28 GiB on Ascend, a tiled
#: one at 0.5 GiB). Tiling changes the pixels: ``weights/inference.json`` records the option with every checkpoint.
UNTILED_DECODE_MAX_PIXELS = 1184 * 896


def _terminal_error(family: str) -> ValueError:
    return ValueError(
        f"schedule {family!r}: the pipeline's terminal-stretched grid is refused for training and student sampling. "
        "At four steps it collapses to [1, .76671, .45561, .02, 0]: the last phase is 0.02 wide, equal to the TDM "
        "query margin (no valid query interval), and the fourth NFE moves the latent by 2% of one velocity. Train "
        "on 'native' (exp(mu) without the stretch); teacher references use QwenImage21Adapter.inference_sigma_grid.")


def _validate_options(options: Any) -> dict[str, Any]:
    if options is None:
        options = {}
    if not isinstance(options, dict):
        raise ValueError("model.options must be a mapping.")
    unknown = sorted(set(options) - set(OPTION_DEFAULTS))
    if unknown:
        raise ValueError(f"Unknown qwen_image_21 option(s) {unknown}; allowed: {sorted(OPTION_DEFAULTS)}")
    resolved = {**OPTION_DEFAULTS, **options}
    if resolved["dtype"] not in _DTYPES:
        raise ValueError(f"options.dtype must be one of {sorted(_DTYPES)}, got {resolved['dtype']!r}.")
    if resolved["noise_dtype"] not in _DTYPES:
        raise ValueError(f"options.noise_dtype must be one of {sorted(_DTYPES)}, got {resolved['noise_dtype']!r}.")
    if resolved["alpha"] not in layout.ALPHA_MODES:
        raise ValueError(f"options.alpha must be one of {layout.ALPHA_MODES}, got {resolved['alpha']!r}.")
    if not isinstance(resolved["kv_cache"], bool):
        raise ValueError("options.kv_cache must be true or false.")
    if resolved["vae_tiling"] not in ("auto", True, False):
        raise ValueError("options.vae_tiling must be 'auto', true or false.")
    check_inference_lora(resolved["inference_lora"], INFERENCE_LORA_MODES)
    return resolved


@register
class QwenImage21Adapter(ModelAdapter):
    name = "qwen_image_21"
    default_lora_targets = components.DEFAULT_LORA_TARGETS
    #: evaluation rollouts: float32 start noise, velocities and Euler states
    sampling_state_dtype = torch.float32
    #: a prompt batch whose prompts differ in token length is refused by ``encode`` (see ``text.MIXED_LENGTHS``)
    padded_prompt_batches = False

    def __init__(self, spec: Any):
        super().__init__(spec)
        if getattr(spec, "name", self.name) != self.name:
            raise ValueError(f"spec.name is {spec.name!r}, expected {self.name!r}.")
        negative = getattr(spec, "negative_prompt", "")
        if not isinstance(negative, str):
            raise ValueError("model.negative_prompt must be a string: '' is the unconditional prompt (encoded as ' ' "
                             "like the pipeline); None is not accepted.")
        resolution = int(spec.resolution)
        if resolution <= 0 or resolution % schedule.PIXEL_MULTIPLE:
            raise ValueError("model.resolution must be a positive multiple of 32.")
        self.options = _validate_options(getattr(spec, "options", None))
        self.dtype = _DTYPES[self.options["dtype"]]
        self.compute_dtype = self.dtype              # read by edmd.runtime.roles for FSDP2's param_dtype
        self.noise_dtype = _DTYPES[self.options["noise_dtype"]]
        self.model_path = expand_path(spec.path)
        # per view: the geometry and the laid-out negative conditioning; the rest lives on the root, shared by its views
        self._geometry = Geometry(height=resolution, width=resolution)
        self._negative: dict[tuple[int, int], QwenCondition] = {}
        self._negative_prompts: dict[tuple[int, int], Any] = {}
        self._root = self
        self._scheduler_config: dict[str, Any] | None = None
        self._scheduler_source = ""
        self._transformer_config: dict[str, Any] | None = None
        self._encoder = None
        self._encoder_version = 0
        self._vae = None
        try:  # install the Ascend patches now when diffusers has QwenImage21; load paths re-check (and raise)
            components.require_diffusers()
        except ImportError:
            pass

    # ------------------------------------------------------------------ checkpoint facts
    @property
    def scheduler_config(self) -> dict[str, Any]:
        """The checkpoint's scheduler config; the embedded copy only when the checkpoint is not mounted at all."""
        root = self._root
        if root._scheduler_config is None:
            if root.model_path.is_dir():
                root._scheduler_config = schedule.load_scheduler_config(root.model_path)
                root._scheduler_source = "checkpoint"
            else:
                root._scheduler_config = dict(schedule.QWEN_SCHEDULER_CONFIG)
                root._scheduler_source = "embedded (checkpoint not mounted)"
        return root._scheduler_config

    @property
    def transformer_config(self) -> dict[str, Any]:
        root = self._root
        if root._transformer_config is None:
            root._transformer_config = components.load_transformer_config(root.model_path)
        return root._transformer_config

    # ------------------------------------------------------------------ geometry (multi-resolution)
    def geometry(self) -> Geometry:
        """This view's generation size; the root adapter's is ``model.resolution`` square."""
        return self._geometry

    def geometry_table(self) -> dict[str, dict[str, Geometry]]:
        """``{"1k": ..., "2k": ...}``: the official 2K sizes and derived 1K sizes (``resolutions.py``)."""
        return resolutions.geometry_table()

    def at(self, geometry: Geometry) -> "QwenImage21Adapter":
        """A view at ``geometry`` (positive multiples of 32) sharing the weights, encoder, VAE and options; everything
        geometry-dependent, including the negative-conditioning layout, is its own."""
        geometry = resolutions.check_geometry(geometry)
        if geometry == self._geometry:
            return self
        root = self._root
        if geometry == root._geometry:
            return root
        view = copy.copy(root)
        view._geometry = geometry
        view._negative = {}
        return view

    @property
    def tokens(self) -> int:
        geometry = self.geometry()
        return schedule.latent_tokens(geometry.height, geometry.width)

    @staticmethod
    def _check_steps(num_steps: int) -> None:
        if isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps < 1:
            raise ValueError("num_steps must be a positive integer.")

    def native_shift_factor(self, num_steps: int) -> float:
        """``exp(mu(tokens))``; the step count does not enter."""
        self._check_steps(num_steps)
        return schedule.native_factor(self.tokens, self.scheduler_config)

    def shift_factor(self, num_steps: int, family: str = "native") -> float:
        if str(family).lower() in schedule.TERMINAL_FAMILIES:
            raise _terminal_error(family)
        fixed = parse_schedule_family(family)
        return self.native_shift_factor(num_steps) if fixed is None else fixed

    def warp(self, raw: Tensor, num_steps: int, family: str = "native") -> Tensor:
        """``F u / (1 + (F - 1) u)`` in float32 -- the grid's own expression, so ``warp(linspace) == grid``."""
        return schedule.sigma_warp(raw, self.shift_factor(num_steps, family))

    def student_grid(self, num_steps: int, family: str = "native", *, device: torch.device | None = None) -> Tensor:
        """The ``[K+1]`` training / student-sampling grid (no terminal stretch)."""
        self._check_steps(num_steps)
        grid = schedule.warped_grid(num_steps, self.shift_factor(num_steps, family))
        return grid.to(device=device) if device is not None else grid

    def inference_sigma_grid(self, num_steps: int = 40, *, device: torch.device | None = None) -> Tensor:
        """The pipeline's grid with the terminal stretch: teacher reference images only (40 steps by default)."""
        self._check_steps(num_steps)
        return schedule.inference_sigma_grid(num_steps, self.tokens, scheduler_config=self.scheduler_config,
                                             device=device)

    def pipeline_grid(self, num_steps: int, *, device: torch.device | None = None) -> Tensor:
        """Evaluation hook (``edmd.evaluation.generate``): the teacher-reference grid = :meth:`inference_sigma_grid`."""
        return self.inference_sigma_grid(num_steps, device=device)

    @staticmethod
    def eval_generator(seed: int, device: torch.device | str | None = None) -> torch.Generator:
        """On the CPU (noise is drawn there and moved), so an image depends on its seed only, whatever the device."""
        return torch.Generator("cpu").manual_seed(int(seed))

    def describe(self) -> dict[str, Any]:
        config = self.scheduler_config
        geometry = self.geometry()
        facts: dict[str, Any] = {
            "adapter": self.name,
            "variant": getattr(self.spec, "variant", ""),
            "model_path": str(self.model_path),
            "resolution": [geometry.height, geometry.width],
            "geometry": {"height": geometry.height, "width": geometry.width},
            "latent_tokens": self.tokens,
            "latent_layout": "[B, h*w, 64] packed from [B, 1, 64, h, w] (unpatched, 16x VAE)",
            "scheduler_config_source": self._root._scheduler_source,
            "mu": schedule.qwen_mu(self.tokens, config),
            "native_shift_factor": self.native_shift_factor(1),
            "shift_depends_on": "latent token count (not the step count)",
            "student_grids": {str(k): schedule.describe_grid(self.student_grid(k)) for k in (1, 2, 4, 8)},
            "terminal_stretch": "refused for training and student sampling; teacher references only",
            "teacher_reference_grid_40": schedule.describe_grid(self.inference_sigma_grid(40)),
            "grid_precision": "float32 F*u/(1+(F-1)u) (<= 1 float32 ULP from the float64 scheduler)",
            "timestep": (f"float32 sigma passed; {'bf16' if self.dtype == torch.bfloat16 else 'fp32'} inside the model "
                         "(the transformer casts it to the hidden-state dtype). The stock pipeline feeds "
                         "bf16(bf16(1000 sigma)/1000): equal on the 1024^2 4-step grid, 12/40 teacher steps differ by "
                         "one bf16 ULP"),
            "compute_dtype": self.options["dtype"],
            "pipeline_cfg_default": 1.0,
            "guidance": "linear u + w (c - u); w == 1 returns cond (no norm rescale)",
            "negative_prompt": self.spec.negative_prompt,
            "negative_prompt_encoded_as": self.spec.negative_prompt or " ",
            "noise": (f"{self.options['noise_dtype']} randn [B, 1, 64, h, w] (diffusers randn_tensor), packed, "
                      "returned float32"),
            "sampling_state_dtype": "float32",
            "decode_alpha": self.options["alpha"],
            "sampling_kv_cache": self.options["kv_cache"],
            "inference_lora": self.options["inference_lora"],
            "padded_prompt_batches": "refused (per-rank prompt batch size 1, or equal token lengths)",
            "vae_tiling": self._vae_tiling(),
            "text_encoder": "Qwen3-VL: raw template, system prefix dropped, final RMSNorm bypassed, no truncation",
            "lora_targets": list(self.default_lora_targets),
            "geometry_table": {
                tier: {"source": resolutions.TIER_SOURCES[tier],
                       "sizes": {ratio: {"width": g.width, "height": g.height,
                                         "tokens": schedule.latent_tokens(g.height, g.width),
                                         "shift_factor": schedule.native_factor(
                                             schedule.latent_tokens(g.height, g.width), config)}
                                 for ratio, g in table.items()}}
                for tier, table in self.geometry_table().items()},
            "vae_tiling_rule": f"auto: tiled above {UNTILED_DECODE_MAX_PIXELS} pixels (the largest 1K-tier area)",
        }
        try:
            import diffusers

            facts["diffusers"] = diffusers.__version__
        except ImportError:  # pragma: no cover
            facts["diffusers"] = None
        return facts

    # ------------------------------------------------------------------ roles
    def build_on_meta(self) -> nn.Module:
        return components.build_transformer(self.transformer_config, device="meta")

    def fsdp_units(self, model: nn.Module) -> list[nn.Module]:
        return components.transformer_blocks(model)

    def read_base_state(self, expected: set[str], *, dtype: torch.dtype) -> dict[str, Tensor]:
        return components.read_transformer_state(self.model_path, expected, dtype=dtype)

    def reset_derived(self, model: nn.Module, device: torch.device) -> None:
        components.reset_derived_tensors(model, device)

    # ------------------------------------------------------------------ conditioning
    def load_conditioning(self, device: torch.device, *, shard: bool = False) -> None:
        components.require_diffusers()
        from .text import QwenPromptEncoder

        device = torch.device(device)
        text_encoder = components.load_text_encoder(self.model_path, dtype=self.dtype, device=device)
        if shard:
            components.shard_text_encoder(text_encoder)
        self.attach_encoder(QwenPromptEncoder(text_encoder, components.load_processor(self.model_path), device=device))

    def attach_encoder(self, encoder: Any) -> None:
        """Use an already-built ``QwenPromptEncoder``, shared by all views (their cached negatives become stale)."""
        root = self._root
        root._encoder = encoder
        root._encoder_version += 1

    @property
    def encoder(self):
        encoder = self._root._encoder
        if encoder is None:
            raise RuntimeError("Call load_conditioning(device) before encoding prompts.")
        return encoder

    def encode(self, prompts: Sequence[str], *, allow_padding: bool = False) -> QwenCondition:
        """Conditioning for ``prompts``; prompts of different token lengths raise unless ``allow_padding``."""
        geometry = self.geometry()
        prompt = self.encoder.encode(prompts, require_equal_lengths=not allow_padding)
        return layout.build_condition(prompt, geometry.height, geometry.width)

    def encode_negative(self, batch_size: int) -> QwenCondition:
        """``[negative_prompt] * batch_size``, encoded once per batch size on the root and laid out per view.

        Encoding on the root keeps a sharded text encoder in step across ranks: with multi-resolution buckets the ranks
        use different views, and per-view encodes would mismatch their collectives and hang the run.
        """
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        root = self._root
        key = (root._encoder_version, batch_size)
        if key not in self._negative:
            prompt = root._negative_prompts.get(key)
            if prompt is None:
                prompt = self.encoder.encode([self.spec.negative_prompt] * batch_size, require_equal_lengths=True)
                root._negative_prompts[key] = prompt
            geometry = self.geometry()
            self._negative[key] = layout.build_condition(prompt, geometry.height, geometry.width)
        return self._negative[key]

    # ------------------------------------------------------------------ denoiser
    def predict_velocity(self, model: nn.Module, x: Tensor, sigma: Tensor, cond: QwenCondition) -> Tensor:
        if not npu_compat.is_installed():
            components.require_diffusers()
        return layout.predict_velocity(model, x, sigma, cond, dtype=self.dtype)

    def predict_velocity_degraded(self, model: nn.Module, x: Tensor, sigma: Tensor, cond: QwenCondition,
                                  blocks: Sequence[int]) -> Tensor:
        """PAG's weak read: ``predict_velocity`` with the identity image map in ``blocks`` (``degraded_attention``)."""
        from .degraded_attention import degraded_attention

        with degraded_attention(model, blocks):
            return self.predict_velocity(model, x, sigma, cond)

    def sampling_velocity(self, model: nn.Module, cond: QwenCondition) -> Callable[[Tensor, Tensor], Tensor]:
        """Velocity for inference rollouts: prefix-KV-cached (``options.kv_cache``, default on, as the pipeline);
        the uncached call is ``predict_velocity``."""
        if not npu_compat.is_installed():
            components.require_diffusers()
        if self.options["kv_cache"]:
            return layout.CachedVelocity(model, cond, dtype=self.dtype)
        return lambda x, sigma: self.predict_velocity(model, x, sigma, cond)

    # ------------------------------------------------------------------ latents
    def sample_noise(self, batch_size: int, *, device: torch.device,
                     generator: torch.Generator | Sequence[torch.Generator] | None = None) -> Tensor:
        """Drawn as ``[B, 1, 64, h, w]`` in ``options.noise_dtype`` (pipeline element order), returned float32."""
        geometry = self.geometry()
        return layout.sample_noise(batch_size, geometry.height, geometry.width, device=device, generator=generator,
                                   draw_dtype=self.noise_dtype)

    def _vae_tiling(self) -> bool:
        """Whether this view's decode is tiled (``options.vae_tiling``; ``auto`` by pixel area, see
        :data:`UNTILED_DECODE_MAX_PIXELS`)."""
        setting = self.options["vae_tiling"]
        if setting != "auto":
            return bool(setting)
        return self._geometry.height * self._geometry.width > UNTILED_DECODE_MAX_PIXELS

    def load_decoder(self, device: torch.device) -> None:
        components.require_diffusers()
        self.attach_decoder(components.load_vae(self.model_path, dtype=self.dtype, device=torch.device(device)))

    def attach_decoder(self, vae: Any) -> None:
        """Use an already-built VAE (tests, or a caller that loaded it itself); shared by the adapter and its views."""
        self._root._vae = vae

    def decode(self, latents: Tensor) -> Tensor:
        """Pixels at this view's geometry. The VAE is shared, so its ``use_tiling`` switch is set for this geometry on
        every call."""
        vae = self._root._vae
        if vae is None:
            raise RuntimeError("Call load_decoder(device) before decoding.")
        if hasattr(vae, "use_tiling"):
            vae.use_tiling = self._vae_tiling()
        geometry = self.geometry()
        return layout.decode_pixels(vae, latents, geometry.height, geometry.width, alpha=self.options["alpha"])

    # ------------------------------------------------------------------ inference / export
    def load_inference_denoiser(self, weights: Path | None, device: torch.device, *,
                                role_train: str = "lora") -> nn.Module:
        """A single-device frozen denoiser: the base (``weights=None``), full exported weights, or the base plus an
        exported adapter with the LoRA numerics of ``options.inference_lora`` (modes: the base class)."""
        from ...runtime import lora as lora_runtime
        from ...runtime.checkpoint import read_weights

        components.require_diffusers()
        device = torch.device(device)
        if role_train not in ("lora", "full"):
            raise ValueError(f"role_train must be 'lora' or 'full', got {role_train!r}.")
        if weights is None:
            return components.load_pretrained_transformer(self.model_path, dtype=self.dtype, device=device)
        tensors, metadata = read_weights(weights)
        recorded = metadata.get("train")
        if recorded is not None and recorded != role_train:
            raise ValueError(f"{weights} holds a {recorded!r} role, but role_train={role_train!r}.")
        if role_train == "full":
            model = self.build_on_meta()
            model.to_empty(device=device)
            nn.Module.to(model, self.dtype)           # before reset_derived, so the sinusoid buffer stays float32
            self.reset_derived(model, device)
            missing, unexpected = model.load_state_dict({key: value.to(self.dtype) for key, value in tensors.items()},
                                                        strict=False)
            if missing or unexpected:
                raise ValueError(f"{weights} does not match the transformer: missing={missing[:3]} "
                                 f"unexpected={unexpected[:3]}")
            model.requires_grad_(False)
            model.eval()
            return model
        settings = metadata.get("lora")
        if not isinstance(settings, dict):
            raise ValueError(f"{weights}: metadata has no 'lora' settings.")
        spec = lora_runtime.LoraSpec(rank=int(settings["rank"]), alpha=float(settings["alpha"]),
                                     dropout=float(settings.get("dropout", 0.0)),
                                     init=str(settings.get("init", "gaussian")),
                                     target_modules=tuple(settings["target_modules"]))
        mode = self.options["inference_lora"]
        lora_dtype = self.dtype if mode == "unmerged_compute" else torch.float32
        home = torch.device("cpu") if mode == "merged" else device
        model = components.load_pretrained_transformer(self.model_path, dtype=self.dtype, device=home)
        names = lora_runtime.inject_lora(model, spec, dtype=lora_dtype)
        expected = {f"{name}.lora_{part}.weight" for name in names for part in ("A", "B")}
        if set(tensors) != expected:
            extra, absent = sorted(set(tensors) - expected), sorted(expected - set(tensors))
            raise ValueError(f"{weights} does not match the LoRA targets: extra={extra[:3]} missing={absent[:3]}")
        model.load_state_dict({key: value.to(device=home, dtype=lora_dtype) for key, value in tensors.items()},
                              strict=False)
        if mode == "merged":
            lora_runtime.merge_lora_into_base(model)
            model = model.to(device)
        elif mode == "unmerged_fp32":
            use_adapter_rounding(model)
        model.requires_grad_(False)
        model.eval()
        return model


__all__ = ["INFERENCE_LORA_MODES", "OPTION_DEFAULTS", "QwenImage21Adapter", "UNTILED_DECODE_MAX_PIXELS"]

"""The model interface every adapter implements, and the facts the algorithm layer may rely on.

The algorithm (``edmd.algo``) and the training step (``edmd.methods.step``) never import a model; they see an
adapter through this class:

* **schedule** (pure, no weights): the exact K-step student grid and the warp from raw time (in which the sampler
  grid is uniform) to sigma; DDPM baselines' uniform integer timestep becomes uniform raw time, then warp;
* **roles**: meta construction, FSDP units, base weights and LoRA defaults, which ``edmd.runtime.roles`` turns into
  sharded teacher / student / fake modules;
* **denoiser call**: ``predict_velocity(model, x, sigma, cond) -> v`` in float32 with ``v = eps - x0``, the timestep
  fed as each adapter documents; ``guide`` is the pipeline's CFG;
* **latents**: Gaussian noise drawn in the pipeline's element order; decode to [0, 1] pixels.

Conditioning objects are opaque to the algorithm; they only need ``batch_size``.
"""

from __future__ import annotations

import abc
import contextlib
import math
import operator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, ClassVar, Sequence

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class Geometry:
    """Pixel size of one generation (rounded by the adapter as its pipeline rounds)."""

    height: int
    width: int


def shift_warp(raw: Tensor, factor: float) -> Tensor:
    """The time shift ``F*u / (1 + (F-1)*u)``, written as ``F / (F + (1/u - 1))`` (the FLUX/diffusers expression)
    with ``u=0`` mapped to exactly 0."""
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("The shift factor must be finite and positive.")
    values = raw.to(torch.promote_types(raw.dtype, torch.float32))   # at least fp32; fp64 kept
    if (values < 0).any() or (values > 1).any():
        raise ValueError("raw times must lie in [0, 1].")
    f = values.new_tensor(float(factor))
    positive = values.clamp_min(1e-6)
    shifted = f / (f + (1.0 / positive - 1.0))
    return torch.where(values <= 0, torch.zeros_like(shifted), shifted).to(dtype=raw.dtype)


def parse_schedule_family(family: str) -> float | None:
    """``native`` -> None (the checkpoint's own schedule); ``shift<F>`` / ``lightning`` (=shift3) -> F."""
    name = str(family).lower()
    if name == "native":
        return None
    if name == "lightning":
        return 3.0
    if name.startswith("shift"):
        try:
            factor = float(name[5:])
        except ValueError:
            factor = float("nan")
        if math.isfinite(factor) and factor > 0:
            return factor
    raise ValueError(f"schedule must be 'native', 'lightning' or 'shift<F>' (F > 0); got {family!r}.")


#: Aspect ratios of the generic area tiers, keyed as the prompt enhancer writes ``wh_ratio`` (``"w:h"``).
ASPECT_RATIOS = ("1:1", "4:3", "3:4", "3:2", "2:3", "16:9", "9:16")
#: Side of the square whose area each generic tier keeps.
AREA_TIERS = {"1k": 1024, "2k": 2048}


def area_geometry_table(multiple: int = 32) -> dict[str, dict[str, Geometry]]:
    """``{tier: {"w:h": Geometry}}``: per tier of :data:`AREA_TIERS` and ratio, the size of the tier's area at that
    ratio (``width = sqrt(area * w / h)``, ``height = width * h / w``), each side rounded to ``multiple`` pixels like
    diffusers' ``calculate_dimensions``. The ``1k`` tier is Qwen-Image-2.1's."""
    table: dict[str, dict[str, Geometry]] = {}
    for tier, side in AREA_TIERS.items():
        sizes = {}
        for ratio in ASPECT_RATIOS:
            w, h = (int(part) for part in ratio.split(":"))
            width = math.sqrt(side * side * w / h)
            sizes[ratio] = Geometry(height=round(width * h / w / multiple) * multiple,
                                    width=round(width / multiple) * multiple)
        table[tier] = sizes
    return table


def check_geometry(geometry: Geometry, multiple: int, *, model: str) -> Geometry:
    """``geometry`` with plain int sides; refused unless both are positive multiples of ``multiple`` pixels."""
    sides = {}
    for name in ("height", "width"):
        value = getattr(geometry, name)
        try:
            integer = None if isinstance(value, bool) else operator.index(value)   # ints, numpy ints; not floats
        except TypeError:
            integer = None
        if integer is None or integer <= 0 or integer % multiple:
            raise ValueError(f"{model}: height and width must be positive multiples of {multiple} pixels, got "
                             f"{geometry.width}x{geometry.height} (width x height)")
        sides[name] = int(integer)
    return Geometry(**sides)


def check_inference_lora(value: Any, modes: Sequence[str]) -> None:
    """Validate ``model.options.inference_lora`` against the adapter's ``modes``."""
    if value not in modes:
        raise ValueError(f"model.options.inference_lora must be one of {tuple(modes)}, got {value!r}.")


class ModelAdapter(abc.ABC):
    """Base class of the FLUX.2 klein, Qwen-Image-2.1, SD3-Medium and toy adapters."""

    #: registry key, e.g. "flux2_klein"
    name: ClassVar[str]
    #: glob patterns of nn.Linear modules that receive LoRA by default
    default_lora_targets: ClassVar[tuple[str, ...]]
    #: dtype of the frozen base weights and of FSDP2's compute (``MixedPrecisionPolicy.param_dtype``)
    compute_dtype: torch.dtype = torch.bfloat16
    #: whether prompts of different token lengths batch exactly (padding changes nothing); if not, training refuses
    #: per-rank batches > 1
    padded_prompt_batches: ClassVar[bool] = True
    #: dtype of an evaluation rollout's latent state (noise, velocities, Euler steps); training rollouts stay float32
    sampling_state_dtype: torch.dtype = torch.float32

    def __init__(self, spec: Any):
        self.spec = spec

    # ------------------------------------------------------------------ static facts / schedule
    @abc.abstractmethod
    def geometry(self) -> Geometry:
        """Training / evaluation generation size."""

    def geometry_table(self) -> dict[str, dict[str, Geometry]]:
        """Training / evaluation geometries: ``{area tier: {"w:h" (as the prompt enhancer's ``wh_ratio``): Geometry}}``;
        every tier has the "1:1" fallback. Default: one tier, the model's own geometry."""
        return {"default": {"1:1": self.geometry()}}

    def at(self, geometry: Geometry) -> "ModelAdapter":
        """This adapter at another geometry: the same weights, encoders and VAE, with everything geometry-dependent
        (grid and shift, noise shape, conditioning layout, decode incl. tiling) following ``geometry``. Default: only
        the model's own geometry."""
        if geometry == self.geometry():
            return self
        raise NotImplementedError(f"{self.name} has no multi-resolution support (its geometry is {self.geometry()})")

    @abc.abstractmethod
    def native_shift_factor(self, num_steps: int) -> float:
        """F of the checkpoint's own time shift for a ``num_steps`` sampler at ``geometry()``."""

    def shift_factor(self, num_steps: int, family: str = "native") -> float:
        fixed = parse_schedule_family(family)
        return self.native_shift_factor(num_steps) if fixed is None else fixed

    def warp(self, raw: Tensor, num_steps: int, family: str = "native") -> Tensor:
        return shift_warp(raw, self.shift_factor(num_steps, family))

    def student_grid(self, num_steps: int, family: str = "native", *, device: torch.device | None = None) -> Tensor:
        """``[K+1]`` float32 sigmas from 1 to 0: warp(linspace(1, 1/K, K)) followed by 0. An adapter whose pipeline
        builds its grid differently overrides this with the pipeline's exact grid."""
        if isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps < 1:
            raise ValueError("num_steps must be a positive integer.")
        raw = torch.linspace(1.0, 1.0 / num_steps, num_steps, dtype=torch.float32)
        grid = torch.cat([self.warp(raw, num_steps, family), torch.zeros(1)])
        return grid.to(device=device) if device is not None else grid

    def raw_grid(self, num_steps: int, family: str = "native", *, device: torch.device | None = None) -> Tensor:
        """``[K+1]`` float32 raw times of the student grid (``warp(raw_grid) ~ student_grid``, to one float32 ulp):
        the coordinate in which ``space: raw`` queries are uniform. Default ``linspace(1, 0, K+1)``."""
        if isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps < 1:
            raise ValueError("num_steps must be a positive integer.")
        return torch.linspace(1.0, 0.0, num_steps + 1, dtype=torch.float32, device=device)

    def pipeline_grid(self, num_steps: int, *, device: torch.device | None = None) -> Tensor:
        """The official pipeline's ``num_steps`` grid (teacher references only). Default: the native student grid."""
        return self.student_grid(num_steps, "native", device=device)

    def describe(self) -> dict[str, Any]:
        """Facts for runtime.json / W&B (grid, shift factor, timestep precision, ...)."""
        return {"adapter": self.name}

    # ------------------------------------------------------------------ roles (see edmd.runtime.roles)
    @abc.abstractmethod
    def build_on_meta(self) -> nn.Module:
        """The denoiser on the meta device, from the checkpoint's config."""

    @abc.abstractmethod
    def fsdp_units(self, model: nn.Module) -> list[nn.Module]:
        """Every block that becomes an FSDP unit (all block types, explicitly)."""

    @abc.abstractmethod
    def read_base_state(self, expected: set[str], *, dtype: torch.dtype) -> dict[str, Tensor]:
        """Rank 0: the denoiser's checkpoint tensors on CPU, strictly the ``expected`` keys."""

    def reset_derived(self, model: nn.Module, device: torch.device) -> None:
        """Rebuild tensors that ``to_empty`` destroyed and the state dict does not carry."""

    def enable_gradient_checkpointing(self, model: nn.Module) -> None:
        if hasattr(model, "enable_gradient_checkpointing"):
            model.enable_gradient_checkpointing()

    # ------------------------------------------------------------------ conditioning
    @abc.abstractmethod
    def load_conditioning(self, device: torch.device, *, shard: bool = False) -> None:
        """Load (and optionally FSDP-shard) the frozen text encoder(s)."""

    @abc.abstractmethod
    def encode(self, prompts: Sequence[str]) -> Any:
        """Conditioning for ``prompts`` (opaque; has ``batch_size``)."""

    @abc.abstractmethod
    def encode_negative(self, batch_size: int) -> Any:
        """Unconditional conditioning for CFG (the pipeline's empty / configured negative prompt)."""

    # ------------------------------------------------------------------ denoiser
    @abc.abstractmethod
    def predict_velocity(self, model: nn.Module, x: Tensor, sigma: Tensor, cond: Any) -> Tensor:
        """``v = eps - x0`` at per-sample ``sigma`` ([B] float32), float32, same shape as ``x``."""

    def predict_velocity_pair(self, model: nn.Module, x: Tensor, sigma: Tensor, cond: Any,
                              uncond: Any) -> tuple[Tensor, Tensor]:
        """``(v_cond, v_uncond)`` for CFG. Default: two ``predict_velocity`` calls; an adapter may batch them like its
        pipeline (SD3) only if padding the mixed batch changes nothing."""
        return self.predict_velocity(model, x, sigma, cond), self.predict_velocity(model, x, sigma, uncond)

    def predict_velocity_degraded(self, model: nn.Module, x: Tensor, sigma: Tensor, cond: Any,
                                  blocks: Sequence[int]) -> Tensor:
        """The conditional read with the image-to-image attention of ``blocks`` replaced by the identity (PAG's weak
        read, ``engine.pag``). No-grad teacher reads only. Adapters without an implementation refuse."""
        raise NotImplementedError(f"{type(self).__name__} has no perturbed-attention (PAG) read")

    def sampling_velocity(self, model: nn.Module, cond: Any) -> Callable[[Tensor, Tensor], Tensor]:
        """``(x, sigma) -> v`` for an inference rollout whose conditioning stays ``cond`` (evaluation). Default:
        ``predict_velocity``; Qwen caches the text prefix KV like its pipeline."""
        return lambda x, sigma: self.predict_velocity(model, x, sigma, cond)

    def sampling_velocity_pair(self, model: nn.Module, cond: Any,
                               uncond: Any) -> Callable[[Tensor, Tensor], tuple[Tensor, Tensor]]:
        """``(x, sigma) -> (v_cond, v_uncond)`` for a guided evaluation rollout (teacher references). Default: two
        ``sampling_velocity`` closures; SD3 runs its pipeline's one batched ``[uncond; cond]`` forward."""
        conditional, unconditional = self.sampling_velocity(model, cond), self.sampling_velocity(model, uncond)
        return lambda x, sigma: (conditional(x, sigma), unconditional(x, sigma))

    def sampling_context(self, device: torch.device | str | None = None) -> contextlib.AbstractContextManager:
        """The context an evaluation renders in (encode, rollout, decode). Default: none; SD3 autocasts to bf16 like
        CDM's evaluation. Training never uses it."""
        return contextlib.nullcontext()

    def guide(self, cond_v: Tensor, uncond_v: Tensor, scale: float) -> Tensor:
        """The pipeline's CFG; linear ``u + w (c - u)`` for all current models."""
        if scale == 1.0:
            return cond_v
        return uncond_v + float(scale) * (cond_v - uncond_v)

    # ------------------------------------------------------------------ latents
    @abc.abstractmethod
    def sample_noise(self, batch_size: int, *, device: torch.device,
                     generator: torch.Generator | Sequence[torch.Generator] | None = None) -> Tensor:
        """Gaussian latents in the layout ``predict_velocity`` takes, drawn in pipeline order and in the adapter's
        noise dtype (a bf16 draw is a different sample from an fp32 draw with the same seed, not a rounding of it),
        returned as float32."""

    def eval_generator(self, seed: int, device: torch.device | str | None = None) -> torch.Generator:
        """The per-image generator of evaluation (initial noise and every fresh draw of a stochastic sampler).
        Default: on ``device``; Qwen keeps it on the CPU, so an image depends on its seed only."""
        return torch.Generator(device=device if device is not None else "cpu").manual_seed(int(seed))

    @abc.abstractmethod
    def load_decoder(self, device: torch.device) -> None:
        """Load the VAE (evaluation only)."""

    @abc.abstractmethod
    def decode(self, latents: Tensor) -> Tensor:
        """Latents -> ``[B, 3, H, W]`` float pixels in [0, 1] (alpha handling, if any, is per adapter)."""

    # ------------------------------------------------------------------ inference / export
    def read_role_weights(self, path: str | Path) -> tuple[dict[str, Tensor], dict[str, Any]]:
        """Initial weights of a trainable role (``model.roles.<role>.load_path``): tensors keyed by parameter name
        plus metadata with ``train`` (and ``lora`` for a LoRA role). Default: an ``edmd_weights_v1`` file; SD3 also
        reads diffusers transformer directories (CDM's turbo weights)."""
        from ..runtime.checkpoint import read_weights

        return read_weights(path)

    @abc.abstractmethod
    def load_inference_denoiser(self, weights: Path | None, device: torch.device,
                                *, role_train: str = "lora") -> nn.Module:
        """A single-device eval denoiser: the base (``weights=None``), the base plus an exported LoRA
        (``role_train="lora"``) or full exported weights (``"full"``).

        LoRA numerics (``model.options.inference_lora``):

        * ``unmerged_compute`` (default): the adapter cast to the compute dtype, exactly the training forward (FSDP2
          casts the fp32 adapter to bf16), so an evaluation measures the function that was trained;
        * ``peft_fp32`` (FLUX.2, SD3): fp32 adapter in peft's cast order, one bf16 rounding at the end; Qwen's
          ``unmerged_fp32``: fp32 adapter, a bf16 rounding after each adapter matmul;
        * ``merged``: folded into the base weights (SD3: ``bf16(W + s B A)``; FLUX.2, Qwen: ``W + bf16(s B A)``).

        With ``B = 0`` (an untrained adapter) every mode equals the base exactly."""


ADAPTERS: dict[str, type[ModelAdapter]] = {}


def register(cls: type[ModelAdapter]) -> type[ModelAdapter]:
    if cls.name in ADAPTERS and ADAPTERS[cls.name] is not cls:
        raise ValueError(f"Adapter {cls.name!r} registered twice.")
    ADAPTERS[cls.name] = cls
    return cls


def build_adapter(spec: Any) -> ModelAdapter:
    """Instantiate the adapter named by ``spec.name`` (lazy imports: only the named model's dependencies load)."""
    name = spec.name
    if name not in ADAPTERS:
        import importlib

        module = {"flux2_klein": "edmd.models.flux2.adapter", "qwen_image_21": "edmd.models.qwen_image.adapter",
                  "sd3_medium": "edmd.models.sd3.adapter", "toy": "edmd.models.toy"}.get(name)
        if module is None:
            raise ValueError(f"Unknown model adapter {name!r}.")
        importlib.import_module(module)
    return ADAPTERS[name](spec)


__all__ = ["ADAPTERS", "AREA_TIERS", "ASPECT_RATIOS", "Geometry", "ModelAdapter", "area_geometry_table",
           "build_adapter", "check_geometry", "check_inference_lora", "parse_schedule_family", "register",
           "shift_warp"]

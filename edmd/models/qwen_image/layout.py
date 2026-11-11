"""The denoiser call and the latent/pixel ends: conditioning layout, packing, noise, velocity, decode.

``QwenImage21Pipeline`` is the specification:

* **Parameterization**: ``v = eps - x0`` (the scheduler's Euler step is ``x + (s' - s) v``).
* **Timestep**: float32 sigma ``[B]``, cast to the hidden-state dtype inside the transformer. The stock pipeline
  passes ``bf16(bf16(1000 s) / 1000)``, one bf16 ULP away on some sigmas.
* **Latents**: unpatched, ``[B, 1, 64, h, w] -> [B, h*w, 64]``.
* **Conditioning layout**: one image slot per 2x2 latent tokens, after the prompt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn

from .schedule import PIXEL_MULTIPLE, VAE_SCALE_FACTOR, latent_tokens
from .text import PromptBatch

IMG_TOKENS_PER_SLOT = 4
LATENT_CHANNELS = 64
ALPHA_MODES = ("drop", "white")


def latent_hw(height: int, width: int) -> tuple[int, int]:
    """Latent grid of an image, after the pipeline's round-down to a multiple of 32 pixels."""
    latent_tokens(height, width)  # validates
    return (int(height) // PIXEL_MULTIPLE * PIXEL_MULTIPLE // VAE_SCALE_FACTOR,
            int(width) // PIXEL_MULTIPLE * PIXEL_MULTIPLE // VAE_SCALE_FACTOR)


@dataclass(frozen=True)
class QwenCondition:
    """Everything a velocity call needs besides latents and time; built once per prompt batch, shared by roles."""

    prompt: PromptBatch
    height: int
    width: int
    img_shapes: list[list[tuple[int, int, int]]]
    img_mask: Tensor

    @property
    def batch_size(self) -> int:
        return self.prompt.batch_size

    @property
    def tokens(self) -> int:
        return latent_tokens(self.height, self.width)

    def to(self, device: torch.device | str) -> "QwenCondition":
        return QwenCondition(prompt=self.prompt.to(device), height=self.height, width=self.width,
                             img_shapes=self.img_shapes, img_mask=self.img_mask.to(device))


def build_condition(prompt: PromptBatch, height: int, width: int) -> QwenCondition:
    """Text-to-image layout: no condition images, so the only image slots are the target's, after the prompt.

    The transformer lays every sample out from ``img_mask[0]``, so in a padded batch each image block starts at the
    padded text length and a shorter prompt's image-text RoPE offsets shift (pipeline behavior, kept; batch size 1
    per rank is exact).
    """
    tokens = latent_tokens(height, width)
    slots = tokens // IMG_TOKENS_PER_SLOT
    if slots * IMG_TOKENS_PER_SLOT != tokens:
        raise ValueError(f"{tokens} latent tokens is not a multiple of {IMG_TOKENS_PER_SLOT}.")
    device = prompt.embeds.device
    img_mask = torch.cat([torch.zeros(prompt.batch_size, prompt.sequence_length, dtype=torch.bool, device=device),
                          torch.ones(prompt.batch_size, slots, dtype=torch.bool, device=device)], dim=1)
    h, w = latent_hw(height, width)
    img_shapes = [[(1, h, w)]] * prompt.batch_size
    return QwenCondition(prompt=prompt, height=int(height), width=int(width), img_shapes=img_shapes, img_mask=img_mask)


def batch_time(sigma: Tensor | float, batch_size: int, *, device: torch.device) -> Tensor:
    """``[B]`` float32 times from a scalar, a one-element tensor or a ``[B]`` tensor."""
    if isinstance(sigma, Tensor):
        values = sigma.to(device=device, dtype=torch.float32).reshape(-1)
        if values.numel() == 1:
            values = values.expand(batch_size)
        if values.numel() != batch_size:
            raise ValueError(f"Expected {batch_size} times, got {values.numel()}.")
        return values.contiguous()
    return torch.full((batch_size,), float(sigma), device=device, dtype=torch.float32)


def _check_latents(latents: Tensor, cond: QwenCondition) -> None:
    if latents.ndim != 3 or latents.shape[-1] != LATENT_CHANNELS:
        raise ValueError(f"latents must be [B, N, {LATENT_CHANNELS}], got {tuple(latents.shape)}.")
    if latents.shape[1] != cond.tokens:
        raise ValueError(f"latents carry {latents.shape[1]} tokens, the condition's geometry says {cond.tokens}.")
    if latents.shape[0] != cond.batch_size:
        raise ValueError(f"latents have batch {latents.shape[0]}, the condition has {cond.batch_size}.")


def predict_velocity(model: nn.Module, latents: Tensor, sigma: Tensor | float, cond: QwenCondition, *,
                     dtype: torch.dtype = torch.bfloat16, kv_cache=None, kv_cache_mode: str | None = None) -> Tensor:
    """``v = eps - x0`` for packed ``[B, N, 64]`` latents, float32; the target image is the output's tail."""
    _check_latents(latents, cond)
    timestep = batch_time(sigma, latents.shape[0], device=latents.device)
    extra = {} if kv_cache is None else {"kv_cache": kv_cache, "kv_cache_mode": kv_cache_mode}
    output = model(
        hidden_states=latents.to(dtype),
        encoder_hidden_states=cond.prompt.embeds,
        encoder_hidden_states_mask=cond.prompt.mask,
        timestep=timestep,
        img_shapes=cond.img_shapes,
        img_mask=cond.img_mask,
        return_dict=False,
        **extra,
    )[0]
    return output[:, -latents.shape[1]:].float()


class CachedVelocity:
    """A velocity function with the pipeline's prefix KV cache; one instance per (model, condition).

    Exact: with ``causal_condition`` the text prefix attends only to itself and is modulated at t = 0, so its keys and
    values do not depend on the latents or the time. The first call prefills, later calls decode the target tokens
    only. The different attention tiling changes the last bits, so the flag is recorded with every checkpoint
    (``adapter_options`` in ``weights/inference.json``).
    """

    def __init__(self, model: nn.Module, cond: QwenCondition, *, dtype: torch.dtype = torch.bfloat16) -> None:
        from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache

        if not bool(getattr(model.config, "causal_condition", False)):
            raise ValueError("The prefix KV cache needs causal_condition=True.")
        self.model, self.cond, self.dtype = model, cond, dtype
        self.cache = QwenImage21KVCache(len(model.transformer_blocks))
        self.calls = 0

    def __call__(self, latents: Tensor, sigma: Tensor | float) -> Tensor:
        mode = "extract" if self.calls == 0 else "cached"
        velocity = predict_velocity(self.model, latents, sigma, self.cond, dtype=self.dtype,
                                    kv_cache=self.cache, kv_cache_mode=mode)
        self.calls += 1
        return velocity


def pack_latents(latents: Tensor) -> Tensor:
    """``[B, 1, C, h, w] -> [B, h*w, C]`` (the pipeline's ``_pack_latents``)."""
    batch, _, channels, height, width = latents.shape
    return latents.reshape(batch, channels, height * width).transpose(1, 2)


def unpack_latents(latents: Tensor, height: int, width: int) -> Tensor:
    """``[B, h*w, C] -> [B, C, 1, h, w]`` (the pipeline's ``_unpack_latents``, the shape the VAE decodes)."""
    batch, _, channels = latents.shape
    h, w = latent_hw(height, width)
    return latents.transpose(1, 2).reshape(batch, channels, 1, h, w)


def sample_noise(batch_size: int, height: int, width: int, *, device: torch.device | str,
                 generator: torch.Generator | Sequence[torch.Generator] | None = None,
                 draw_dtype: torch.dtype = torch.float32) -> Tensor:
    """Packed Gaussian latents, drawn as ``[B, 1, 64, h, w]`` in ``draw_dtype`` with diffusers' ``randn_tensor``
    (pipeline order) and returned as float32."""
    from diffusers.utils.torch_utils import randn_tensor

    h, w = latent_hw(height, width)
    shape = (int(batch_size), 1, LATENT_CHANNELS, h, w)
    generator = list(generator) if isinstance(generator, (list, tuple)) else generator
    noise = randn_tensor(shape, generator=generator, device=torch.device(device), dtype=draw_dtype)
    return pack_latents(noise).float()


@torch.no_grad()
def decode_pixels(vae: nn.Module, latents: Tensor, height: int, width: int, *, alpha: str = "drop") -> Tensor:
    """Packed latents -> ``[B, 3, H, W]`` float32 in [0, 1]: the pipeline's decode and postprocess (in the VAE dtype),
    then the RGBA VAE's alpha is dropped (``"drop"``, as ``Image.convert("RGB")`` does) or composited onto white
    (``"white"``, in float32)."""
    if alpha not in ALPHA_MODES:
        raise ValueError(f"alpha must be one of {ALPHA_MODES}, got {alpha!r}.")
    unpacked = unpack_latents(latents, height, width).to(vae.dtype)
    shape = (1, vae.config.z_dim, 1, 1, 1)
    mean = torch.tensor(vae.config.latents_mean).view(shape).to(unpacked.device, unpacked.dtype)
    std = torch.tensor(vae.config.latents_std).view(shape).to(unpacked.device, unpacked.dtype)
    image = vae.decode(unpacked * std + mean, return_dict=False)[0][:, :, 0]
    image = (image * 0.5 + 0.5).clamp(0, 1).float()
    if image.shape[1] == 3:
        return image
    if image.shape[1] != 4:
        raise ValueError(f"Expected an RGB or RGBA decode, got {image.shape[1]} channels.")
    rgb, a = image[:, :3], image[:, 3:4]
    return rgb if alpha == "drop" else rgb * a + (1.0 - a)


__all__ = [
    "ALPHA_MODES", "CachedVelocity", "IMG_TOKENS_PER_SLOT", "LATENT_CHANNELS", "QwenCondition", "batch_time",
    "build_condition", "decode_pixels", "latent_hw", "pack_latents", "predict_velocity", "sample_noise",
    "unpack_latents",
]

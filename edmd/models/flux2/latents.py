"""FLUX.2 latent layout: packing, image position ids, Gaussian noise and VAE decoding.

As in ``Flux2KleinPipeline``, the VAE latent ``[B, 32, 2h, 2w]`` is patchified 2x2 into ``[B, 128, h, w]`` ("raw"
latents) and packed row-major into ``[B, h*w, 128]`` tokens, ``h = H / 16``; the adapter works on packed tokens.
Noise is drawn in the raw shape and then packed, the pipeline's element order, so a seed gives the pipeline's sample.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn

from .components import LATENT_CHANNELS

PATCH = 2


def pack(raw: Tensor) -> Tensor:
    """``[B, C, h, w] -> [B, h*w, C]``."""
    if raw.ndim != 4:
        raise ValueError(f"Expected raw latents [B, C, h, w], got {tuple(raw.shape)}")
    batch, channels, height, width = raw.shape
    return raw.reshape(batch, channels, height * width).permute(0, 2, 1).contiguous()


def unpack(tokens: Tensor, height: int, width: int) -> Tensor:
    """``[B, h*w, C] -> [B, C, h, w]``."""
    if tokens.ndim != 3 or tokens.shape[1] != int(height) * int(width):
        raise ValueError(f"Expected packed tokens [B, {int(height) * int(width)}, C], got {tuple(tokens.shape)}")
    batch, _, channels = tokens.shape
    return tokens.permute(0, 2, 1).reshape(batch, channels, int(height), int(width)).contiguous()


def image_ids(batch_size: int, height: int, width: int, *, device: torch.device | str) -> Tensor:
    """``(t, h, w, l) = (0, row, col, 0)`` for every token, ``[B, h*w, 4]`` int64 (the pipeline's
    ``_prepare_latent_ids``)."""
    coords = torch.cartesian_prod(
        torch.arange(1, device=device, dtype=torch.long),
        torch.arange(int(height), device=device, dtype=torch.long),
        torch.arange(int(width), device=device, dtype=torch.long),
        torch.arange(1, device=device, dtype=torch.long),
    )
    return coords.unsqueeze(0).expand(int(batch_size), -1, -1)


def sample_noise(
    batch_size: int,
    height: int,
    width: int,
    *,
    device: torch.device | str,
    generator: torch.Generator | Sequence[torch.Generator] | None = None,
    draw_dtype: torch.dtype,
) -> Tensor:
    """Packed ``[B, h*w, 128]`` float32 noise, drawn as ``[B, 128, h, w]`` in ``draw_dtype``."""
    from diffusers.utils.torch_utils import randn_tensor

    if isinstance(batch_size, bool) or int(batch_size) <= 0:
        raise ValueError("batch_size must be positive.")
    if isinstance(generator, (list, tuple)):
        generator = list(generator)
        if len(generator) != int(batch_size):
            raise ValueError("A generator list needs one generator per sample.")
    shape = (int(batch_size), LATENT_CHANNELS, int(height), int(width))
    raw = randn_tensor(shape, generator=generator, device=torch.device(device), dtype=draw_dtype)
    return pack(raw).float()


def unpatchify(raw: Tensor) -> Tensor:
    """``[B, 4c, h, w] -> [B, c, 2h, 2w]`` (pipeline ``_unpatchify_latents``)."""
    batch, channels, height, width = raw.shape
    if channels % (PATCH * PATCH):
        raise ValueError(f"Cannot unpatchify {channels} channels.")
    raw = raw.reshape(batch, channels // (PATCH * PATCH), PATCH, PATCH, height, width)
    raw = raw.permute(0, 1, 4, 2, 5, 3)
    return raw.reshape(batch, channels // (PATCH * PATCH), height * PATCH, width * PATCH)


@torch.no_grad()
def decode(vae: nn.Module, tokens: Tensor, height: int, width: int) -> Tensor:
    """Packed normalized latents -> ``[B, 3, 16h, 16w]`` float32 pixels in [0, 1]."""
    parameter = next(vae.parameters())
    device, dtype = parameter.device, parameter.dtype
    raw = unpack(tokens, height, width).to(device=device, dtype=dtype)
    bn = getattr(vae, "bn", None)
    if bn is None:
        raise RuntimeError("AutoencoderKLFlux2 has no latent batch norm `bn`; the upstream model changed.")
    mean = bn.running_mean.view(1, -1, 1, 1).to(device=device, dtype=raw.dtype)
    std = torch.sqrt(bn.running_var.view(1, -1, 1, 1) + float(vae.config.batch_norm_eps)).to(device=device, dtype=raw.dtype)
    raw = unpatchify(raw * std + mean)
    decoded = vae.decode(raw.to(dtype), return_dict=False)[0]
    return (decoded / 2 + 0.5).clamp(0, 1).float()


__all__ = ["PATCH", "decode", "image_ids", "pack", "sample_noise", "unpack", "unpatchify"]

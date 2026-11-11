"""The FLUX.2 klein velocity call, fed exactly as ``Flux2KleinPipeline`` feeds the transformer.

As in the pipeline, the timestep is ``bf16(bf16(sigma * 1000) / 1000)`` (quantized on the 0..1000 scale), klein base
gets no guidance embedding, and the output is sliced to the image tokens. FSDP2 does not cast forward inputs
(``cast_forward_inputs=False``), so :func:`predict_velocity` does all input casts itself; it returns float32.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .components import LATENT_CHANNELS
from .latents import image_ids
from .text import Flux2Conditioning


def timestep_input(sigma: Tensor, dtype: torch.dtype) -> Tensor:
    """The pipeline's timestep argument for per-sample ``sigma``: ``dtype(dtype(sigma * 1000) / 1000)``."""
    return (sigma.to(torch.float32) * 1000).to(dtype) / 1000


def check_sigma(sigma: Tensor | float, batch_size: int, device: torch.device) -> Tensor:
    """``[B]`` float32 sigmas on ``device`` (a scalar is expanded); values must lie in [0, 1]."""
    if not isinstance(sigma, Tensor):
        sigma = torch.tensor(float(sigma), dtype=torch.float32)
    sigma = sigma.to(device=device, dtype=torch.float32)
    if sigma.ndim == 0:
        sigma = sigma.expand(batch_size)
    if sigma.ndim != 1 or sigma.shape[0] != batch_size:
        raise ValueError(f"sigma must be a scalar or [B={batch_size}], got {tuple(sigma.shape)}")
    if not bool(((sigma >= 0) & (sigma <= 1)).all()):
        raise ValueError("sigma must lie in [0, 1] (FLUX takes sigma, not the 0..1000 scheduler timestep).")
    return sigma


def predict_velocity(model: nn.Module, x: Tensor, sigma: Tensor | float, cond: Flux2Conditioning, *,
                     height: int, width: int, dtype: torch.dtype) -> Tensor:
    """``v = eps - x0`` for packed ``[B, h*w, 128]`` latents at per-sample ``sigma``, float32."""
    if x.ndim != 3 or x.shape[-1] != LATENT_CHANNELS:
        raise ValueError(f"x must be packed [B, N, {LATENT_CHANNELS}] latents, got {tuple(x.shape)}")
    batch, tokens, _ = x.shape
    if tokens != int(height) * int(width):
        raise ValueError(f"x carries {tokens} tokens, the adapter geometry is {height}x{width}={height * width}.")
    if not isinstance(cond, Flux2Conditioning):
        raise TypeError(f"cond must be a Flux2Conditioning, got {type(cond).__name__}")
    if cond.batch_size != batch:
        raise ValueError(f"cond has batch {cond.batch_size}, x has {batch}.")
    sigma = check_sigma(sigma, batch, x.device)
    output = model(
        hidden_states=x.to(dtype),
        encoder_hidden_states=cond.embeds.to(device=x.device, dtype=dtype),
        timestep=timestep_input(sigma, dtype),
        img_ids=image_ids(batch, height, width, device=x.device),
        txt_ids=cond.text_ids.to(x.device),
        guidance=None,
        return_dict=False,
    )[0]
    return output[:, :tokens].float()


__all__ = ["check_sigma", "predict_velocity", "timestep_input"]

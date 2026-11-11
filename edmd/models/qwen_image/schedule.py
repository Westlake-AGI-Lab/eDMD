"""Qwen-Image-2.1 time schedule: the checkpoint's token-dependent shift, and the terminal stretch it is kept from.

The pipeline shifts ``linspace(1, 1/K, K)`` by ``s' = F / (F + (1/s - 1))`` with ``F = exp(mu)``, ``mu`` linear in the
latent token count, stretches the grid so its last entry is ``shift_terminal`` (0.02) and appends 0.

* **student grid** (training and the student's own sampler): the shift **without** the stretch; ``F`` depends on the
  token count, not the step count (1024^2: ``F = 2.000803``, 4 steps ``[1, .85719, .66676, .40010, 0]``).
* **inference grid** (teacher references only): the pipeline's grid. At four steps, ``[1, .76671, .45561, .02, 0]``:
  its last phase equals TDM's query margin, so it is refused for training and student sampling (``TERMINAL_FAMILIES``).

The warp is float32 ``F u / (1 + (F - 1) u)``, within one float32 ULP of the scheduler's float64 form.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor

#: ``scheduler/scheduler_config.json`` of Qwen/Qwen-Image-2.1 (a copy of the checkpoint's file). Used only when the
#: checkpoint is not mounted; the adapter reads the real file when present.
QWEN_SCHEDULER_CONFIG: dict[str, Any] = {
    "base_image_seq_len": 256,
    "base_shift": 0.5,
    "max_image_seq_len": 8192,
    "max_shift": 0.9,
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": 0.02,
    "time_shift_type": "exponential",
    "use_dynamic_shifting": True,
}
#: Names under which the pipeline's terminal-stretched grid might be requested for training; always refused.
TERMINAL_FAMILIES = ("qwen_terminal", "native_terminal", "terminal")
VAE_SCALE_FACTOR = 16
PIXEL_MULTIPLE = 32  # the pipeline rounds requests down to vae_scale_factor * 2


def load_scheduler_config(model_path: str | Path) -> dict[str, Any]:
    """Read and validate ``scheduler/scheduler_config.json``."""
    path = Path(model_path) / "scheduler" / "scheduler_config.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; point model.path at a full Qwen-Image-2.1 checkpoint.")
    config = json.loads(path.read_text(encoding="utf-8"))
    for key in ("base_image_seq_len", "base_shift", "max_image_seq_len", "max_shift", "time_shift_type"):
        if key not in config:
            raise ValueError(f"{path} lacks {key}; the shift ramp cannot be inferred.")
    if not bool(config.get("use_dynamic_shifting", False)):
        raise ValueError("Qwen-Image-2.1 is expected to use dynamic shifting; this checkpoint does not.")
    if str(config["time_shift_type"]) != "exponential":
        raise ValueError(f"Unsupported time_shift_type {config['time_shift_type']!r}; only 'exponential' is ported.")
    return config


def latent_tokens(height: int, width: int) -> int:
    """Latent tokens for an ``height x width`` image: 16x VAE, unpatched, request rounded down to 32 px."""
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive.")
    height = int(height) // PIXEL_MULTIPLE * PIXEL_MULTIPLE
    width = int(width) // PIXEL_MULTIPLE * PIXEL_MULTIPLE
    if height == 0 or width == 0:
        raise ValueError("height and width must be at least 32 pixels.")
    return (height // VAE_SCALE_FACTOR) * (width // VAE_SCALE_FACTOR)


def qwen_mu(image_seq_len: int, scheduler_config: Mapping[str, Any] | None = None) -> float:
    """``mu`` of the dynamic shift for ``image_seq_len`` tokens (the pipeline's ``calculate_shift``)."""
    config = dict(scheduler_config or QWEN_SCHEDULER_CONFIG)
    tokens = int(image_seq_len)
    if tokens <= 0:
        raise ValueError("image_seq_len must be positive.")
    base_seq = float(config.get("base_image_seq_len", 256))
    max_seq = float(config.get("max_image_seq_len", 8192))
    base_shift = float(config.get("base_shift", 0.5))
    max_shift = float(config.get("max_shift", 0.9))
    if base_seq == max_seq:
        raise ValueError("base_image_seq_len and max_image_seq_len must differ.")
    slope = (max_shift - base_shift) / (max_seq - base_seq)
    return slope * tokens + (base_shift - slope * base_seq)


def native_factor(image_seq_len: int, scheduler_config: Mapping[str, Any] | None = None) -> float:
    """``F = exp(mu(tokens))`` -- the checkpoint's own shift factor (no dependence on the step count)."""
    return float(math.exp(qwen_mu(image_seq_len, scheduler_config)))


def _pinned(result: Tensor, argument: Tensor) -> Tensor:
    """0 and 1 are fixed points of the warp that the float32 expression can miss by one ulp: pin both ends, clamp."""
    result = torch.where(argument >= 1, torch.ones_like(result), result)
    result = torch.where(argument <= 0, torch.zeros_like(result), result)
    return result.clamp(0.0, 1.0)


def sigma_warp(raw: Tensor | float, factor: float) -> Tensor:
    """``F u / (1 + (F - 1) u)`` in float32, with 0 and 1 pinned."""
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("The shift factor must be finite and positive.")
    values = raw if isinstance(raw, Tensor) else torch.as_tensor(raw, dtype=torch.float32)
    floats = values.float()
    if (floats < 0).any() or (floats > 1).any():
        raise ValueError("raw times must lie in [0, 1].")
    shifted = _pinned(factor * floats / (1.0 + (factor - 1.0) * floats), floats)
    return shifted.to(device=values.device, dtype=values.dtype)


def stretch_to_terminal(sigmas: Tensor, terminal: float) -> Tensor:
    """The scheduler's ``stretch_shift_to_terminal`` on the sampled sigmas (no trailing 0)."""
    if sigmas.ndim != 1 or sigmas.numel() < 2:
        raise ValueError("stretch_to_terminal expects the 1-D grid of sampled sigmas (at least two).")
    if not 0 <= terminal < 1:
        raise ValueError(f"shift_terminal must lie in [0, 1), got {terminal}.")
    one_minus = 1.0 - sigmas.float()
    scale = one_minus[-1] / (1.0 - terminal)
    if float(scale) == 0.0:
        return sigmas
    return (1.0 - one_minus / scale).to(sigmas.dtype)


def warped_grid(num_steps: int, factor: float, *, terminal: float | None = None) -> Tensor:
    """``[K+1]`` float32 grid: ``sigma_warp(linspace(1, 0, K+1))`` (+ optional stretch), ends pinned to 1 and 0.

    The pipeline's ``linspace(1, 1/K, K)`` + 0 is the same grid mathematically; this spelling's float32 bits are the
    ones training uses.
    """
    if isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps <= 0:
        raise ValueError("num_steps must be a positive integer.")
    raw = torch.linspace(1.0, 0.0, num_steps + 1, dtype=torch.float32)
    grid = sigma_warp(raw, factor)
    if terminal:
        if num_steps < 2:
            raise ValueError("The terminal stretch needs at least two sampled sigmas (the pipeline divides by 0 at K=1).")
        grid = torch.cat([stretch_to_terminal(grid[:-1], float(terminal)), grid.new_zeros(1)])
    grid[0] = 1.0
    grid[-1] = 0.0
    if not bool(torch.all(grid[1:] < grid[:-1])):
        raise ValueError(f"The sigma grid must be strictly decreasing, got {grid.tolist()}.")
    return grid


def inference_sigma_grid(num_steps: int, image_seq_len: int, *,
                         scheduler_config: Mapping[str, Any] | None = None,
                         device: torch.device | str | None = None) -> Tensor:
    """The pipeline's ``[K+1]`` grid (to one float32 ULP), terminal stretch included: teacher references only."""
    config = scheduler_config or QWEN_SCHEDULER_CONFIG
    grid = warped_grid(num_steps, native_factor(image_seq_len, config), terminal=config.get("shift_terminal"))
    return grid.to(device=device) if device is not None else grid


def describe_grid(grid: Tensor) -> list[float]:
    return [round(float(value), 6) for value in grid.tolist()]


__all__ = [
    "PIXEL_MULTIPLE", "QWEN_SCHEDULER_CONFIG", "TERMINAL_FAMILIES", "VAE_SCALE_FACTOR", "describe_grid",
    "inference_sigma_grid", "latent_tokens", "load_scheduler_config", "native_factor", "qwen_mu", "sigma_warp",
    "stretch_to_terminal", "warped_grid",
]

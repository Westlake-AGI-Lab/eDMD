"""SD3-Medium time schedule: a static shift, applied exactly as ``FlowMatchEulerDiscreteScheduler`` applies it.

:func:`static_shift_warp` evaluates the scheduler's float32 ``shift * s / (1 + (shift - 1) * s)`` in its operation
order (the base-class ``shift_warp`` differs in the last ulp for about half of all inputs). The pipeline's default grid
(no ``sigmas``) shifts the already-shifted ``sigma_min`` again and ends in a nearly empty step (K = 4:
``[1, 0.8577, 0.6022, 0.0089, 0]``); CDM's turbo inference and training pass ``sigmas = linspace(1, 1/K, K)`` instead
(K = 4: ``[1, 0.9, 0.75, 0.5, 0]``). :func:`custom_sigma_grid` is that grid (the student grid),
:func:`pipeline_default_grid` the pipeline's (teacher references); both run the real scheduler.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor

SCHEDULER_SUBDIR = "scheduler"

#: Scheduler settings under which ``set_timesteps(sigmas)`` is exactly the static shift of the raw grid.
REQUIRED_SCHEDULER_CONFIG: dict[str, Any] = {
    "use_dynamic_shifting": False,
    "shift_terminal": None,
    "use_karras_sigmas": False,
    "use_exponential_sigmas": False,
    "use_beta_sigmas": False,
    "invert_sigmas": False,
    "stochastic_sampling": False,
}


def load_scheduler_config(model_path: str | Path) -> dict[str, Any]:
    path = Path(model_path) / SCHEDULER_SUBDIR / "scheduler_config.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; point model.path at a full SD3 diffusers checkpoint.")
    config = json.loads(path.read_text(encoding="utf-8"))
    check_scheduler_config(config)
    return config


def check_scheduler_config(config: Mapping[str, Any]) -> None:
    """The checkpoint scheduler must be a static-shift FlowMatchEulerDiscreteScheduler and nothing more."""
    if config.get("_class_name", "FlowMatchEulerDiscreteScheduler") != "FlowMatchEulerDiscreteScheduler":
        raise ValueError(f"Expected a FlowMatchEulerDiscreteScheduler config, got {config.get('_class_name')!r}.")
    wrong = {key: config.get(key, expected) for key, expected in REQUIRED_SCHEDULER_CONFIG.items()
             if config.get(key, expected) != expected}
    shift = config.get("shift", 1.0)
    if isinstance(shift, bool) or not isinstance(shift, (int, float)) or not math.isfinite(shift) or shift <= 0:
        wrong["shift"] = shift
    steps = config.get("num_train_timesteps", 1000)
    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        wrong["num_train_timesteps"] = steps
    if wrong:
        raise ValueError(f"Unsupported SD3 scheduler config (the sd3_medium adapter needs a static shift and "
                         f"{REQUIRED_SCHEDULER_CONFIG}): {wrong}")


def _check_steps(num_steps: int) -> int:
    if isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps < 1:
        raise ValueError("num_steps must be a positive integer.")
    return num_steps


def static_shift_warp(raw: Tensor, shift: float) -> Tensor:
    """``shift * u / (1 + (shift - 1) * u)`` in float32, in the scheduler's operation order; returns ``raw.dtype``."""
    if not math.isfinite(shift) or shift <= 0:
        raise ValueError("The shift factor must be finite and positive.")
    values = raw.to(torch.float32)
    if bool((values < 0).any()) or bool((values > 1).any()):
        raise ValueError("raw times must lie in [0, 1].")
    factor = float(shift)
    numerator = factor * values
    denominator = 1.0 + (factor - 1.0) * values
    return (numerator / denominator).to(dtype=raw.dtype)


def _scheduler(config: Mapping[str, Any], shift: float | None):
    from diffusers import FlowMatchEulerDiscreteScheduler

    check_scheduler_config(config)
    values = dict(config)
    if shift is not None:
        values["shift"] = float(shift)
    return FlowMatchEulerDiscreteScheduler.from_config(values)


def _finish(scheduler, num_steps: int) -> Tensor:
    grid = scheduler.sigmas
    if grid is None or grid.ndim != 1 or int(grid.numel()) != num_steps + 1:
        raise RuntimeError(f"The scheduler returned {None if grid is None else tuple(grid.shape)}, not {num_steps + 1} sigmas.")
    grid = grid.to(dtype=torch.float32, device="cpu").clone()
    if float(grid[0]) != 1.0 or float(grid[-1]) != 0.0 or not bool((grid[:-1] > grid[1:]).all()):
        raise RuntimeError(f"Unexpected SD3 grid {grid.tolist()}.")
    return grid


def raw_grid(num_steps: int) -> np.ndarray:
    """``np.linspace(1, 1/K, K)`` in float64, the ``sigmas`` CDM's SD3 inference passes (a float32 ``torch.linspace``
    differs by one ulp for some K)."""
    k = _check_steps(num_steps)
    return np.linspace(1.0, 1.0 / k, k)


def custom_sigma_grid(config: Mapping[str, Any], num_steps: int, *, shift: float | None = None) -> Tensor:
    """``[K+1]`` float32 CPU sigmas of ``retrieve_timesteps(scheduler, K, sigmas=linspace(1, 1/K, K))`` (+ trailing 0).

    ``shift`` overrides the checkpoint's static shift (``schedule: shift<F>``); None keeps it.
    """
    k = _check_steps(num_steps)
    scheduler = _scheduler(config, shift)
    scheduler.set_timesteps(sigmas=raw_grid(k), device="cpu")
    return _finish(scheduler, k)


def pipeline_default_grid(config: Mapping[str, Any], num_steps: int, *, shift: float | None = None) -> Tensor:
    """``[K+1]`` float32 CPU sigmas of ``StableDiffusion3Pipeline(num_inference_steps=K)`` without ``sigmas``."""
    k = _check_steps(num_steps)
    scheduler = _scheduler(config, shift)
    scheduler.set_timesteps(num_inference_steps=k, device="cpu")
    return _finish(scheduler, k)


__all__ = ["REQUIRED_SCHEDULER_CONFIG", "SCHEDULER_SUBDIR", "check_scheduler_config", "custom_sigma_grid",
           "load_scheduler_config", "pipeline_default_grid", "raw_grid", "static_shift_warp"]

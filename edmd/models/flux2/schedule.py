"""FLUX.2 klein time schedule: the empirical dynamic shift and the pipeline's exact K-step grid.

The pipeline shifts ``np.linspace(1, 1/K, K)`` by ``F = exp(mu)``, where ``mu = compute_empirical_mu(tokens, K)``
depends on the step count (4 steps at 1024^2: F = 9.8866, grid ``[1, .96738, .90814, .76720, 0]``).
:func:`native_grid` runs that through the real diffusers scheduler. Continuous warps (queries in raw time) use the
base-class ``shift_warp`` with ``F = native_shift_factor(K)``, the scheduler's own float32 expression.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor

#: Scheduler config values that make ``set_timesteps(sigmas, mu)`` a pure exponential shift of the raw grid.
REQUIRED_SCHEDULER_CONFIG: dict[str, Any] = {
    "use_dynamic_shifting": True,
    "time_shift_type": "exponential",
    "shift_terminal": None,
    "use_karras_sigmas": False,
    "use_exponential_sigmas": False,
    "use_beta_sigmas": False,
    "invert_sigmas": False,
    "stochastic_sampling": False,
    "use_flow_sigmas": False,
}
#: FlowMatchEulerDiscreteScheduler's own defaults, used for keys a config file omits.
_SCHEDULER_DEFAULTS: dict[str, Any] = {**REQUIRED_SCHEDULER_CONFIG, "use_dynamic_shifting": False}


def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    """The klein pipeline's ``compute_empirical_mu`` (copied, not imported: it is module-private)."""
    image_seq_len = int(image_seq_len)
    num_steps = int(num_steps)
    if image_seq_len <= 0 or num_steps <= 0:
        raise ValueError("image_seq_len and num_steps must be positive.")
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666
    if image_seq_len > 4300:
        return float(a2 * image_seq_len + b2)
    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    return float(a * num_steps + b)


def check_scheduler_config(config: Mapping[str, Any]) -> None:
    """The checkpoint scheduler must be a pure dynamic exponential shift, which :func:`native_shift_factor` and the
    base-class warp assume."""
    if config.get("_class_name", "FlowMatchEulerDiscreteScheduler") != "FlowMatchEulerDiscreteScheduler":
        raise ValueError(f"Expected a FlowMatchEulerDiscreteScheduler config, got {config.get('_class_name')!r}.")
    actual = {key: config.get(key, _SCHEDULER_DEFAULTS[key]) for key in REQUIRED_SCHEDULER_CONFIG}
    wrong = {key: value for key, value in actual.items() if value != REQUIRED_SCHEDULER_CONFIG[key]}
    if int(config.get("num_train_timesteps", 1000)) != 1000:
        # the pipeline feeds sigma * num_train_timesteps / 1000; predict_velocity hard-codes the 1000
        wrong["num_train_timesteps"] = config["num_train_timesteps"]
    if wrong:
        raise ValueError(f"Unsupported FLUX.2 scheduler config (the klein adapter needs {REQUIRED_SCHEDULER_CONFIG}): {wrong}")


def native_shift_factor(image_seq_len: int, num_steps: int) -> float:
    """``F = exp(mu(image_seq_len, num_steps))`` -- the time-shift factor of a ``num_steps`` sampler."""
    if isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps < 1:
        raise ValueError("num_steps must be a positive integer.")
    return float(math.exp(compute_empirical_mu(image_seq_len, num_steps)))


def native_grid(scheduler_config: Mapping[str, Any], *, image_seq_len: int, num_steps: int) -> Tensor:
    """``[K+1]`` float32 CPU sigmas of ``Flux2KleinPipeline(num_inference_steps=K)``, through a fresh real scheduler."""
    from diffusers import FlowMatchEulerDiscreteScheduler

    if isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps < 1:
        raise ValueError("num_steps must be a positive integer.")
    check_scheduler_config(scheduler_config)
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(dict(scheduler_config))
    sigmas = np.linspace(1.0, 1 / num_steps, num_steps)                  # as Flux2KleinPipeline.__call__
    mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=num_steps)
    scheduler.set_timesteps(sigmas=sigmas, device="cpu", mu=mu)         # retrieve_timesteps' sigmas branch
    grid = scheduler.sigmas
    if grid is None or grid.ndim != 1 or int(grid.numel()) != num_steps + 1:
        raise RuntimeError(f"The scheduler returned {None if grid is None else tuple(grid.shape)}, not {num_steps + 1} sigmas.")
    grid = grid.to(dtype=torch.float32, device="cpu")
    if float(grid[0]) != 1.0 or float(grid[-1]) != 0.0 or not bool((grid[:-1] > grid[1:]).all()):
        raise RuntimeError(f"Unexpected FLUX.2 grid {grid.tolist()}.")
    return grid


__all__ = ["REQUIRED_SCHEDULER_CONFIG", "check_scheduler_config", "compute_empirical_mu", "native_grid",
           "native_shift_factor"]

"""Student rollouts: the samplers each method trains through and evaluates with.

Two samplers (``rollout.sampler``):

* ``euler`` -- deterministic ``x <- x + (s' - s) v``: TDM (its DDIM), eDMD, D-DMD, CDM. Stored states are detached.
* ``x0_renoise`` -- DMD2: at every step predict ``x0`` and re-noise it with **fresh** noise to the next level,
  ``(1 - s') x0 + s' z``. The final step returns its x0.

The rollout record keeps, per grid point, the state and (optionally) the one-step x0 prediction and the velocity at that
point -- what the critic sources and CDM's off-trajectory term read.

``backward_simulation`` reproduces DMD2's training draw, including its random-number order: fresh start noise, one fresh
``randn`` per executed step (the last one is discarded by DMD2), and the exit input rebuilt from the round's own noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import torch
from torch import Tensor

from . import fm
from .draws import Draws, TorchDraws

VelocityFn = Callable[[Tensor, Tensor], Tensor]


@dataclass
class Rollout:
    """A detached student trajectory on ``grid`` ([n+1], 1 -> 0).

    ``states[i]`` is x at ``grid[i]`` (``states[0]`` the start noise; for x0_renoise the re-noised state fed to
    step i). ``x0s[i]`` / ``velocities[i]`` are the student's prediction at point i (i < n), when recorded.
    ``final`` is the sampler's output (the last state for euler, the last x0 for x0_renoise).
    """

    grid: Tensor
    states: list[Tensor]
    final: Tensor
    x0s: list[Tensor] = field(default_factory=list)
    velocities: list[Tensor] = field(default_factory=list)
    exit_step: Optional[int] = None           # DMD2: the step whose input the student recomputes with grad
    anchor: Optional[Tensor] = None           # CDM: per-sample anchor index drawn at rollout time
    raw_grid: Optional[Tensor] = None         # CDM: this rollout's own sampled raw grid, before model shift

    @property
    def num_steps(self) -> int:
        return int(self.grid.numel()) - 1


#: How far a random grid may rise between two points (float32 rounding of the warp).
GRID_RISE_TOLERANCE = 1e-6


def _check_grid(grid: Tensor, *, strict: bool = True) -> Tensor:
    """A sampler grid from <= 1 down to exactly 0. ``strict`` (fixed grids): strictly descending. Otherwise (CDM's random
    grids) ties and float32-rounding rises are accepted, as in the official sampler: sorted raw points that the warp
    maps to the same float32 sigma are a zero-length Euler step. Refusing them would stop a CDM run at random."""
    grid = fm.wide(grid.detach()).reshape(-1)
    descending = (grid[:-1] > grid[1:]).all() if strict else (grid[1:] - grid[:-1] <= GRID_RISE_TOLERANCE).all()
    if (grid.numel() < 2 or not bool(torch.isfinite(grid).all()) or not descending or float(grid[0]) > 1.0
            or float(grid[-1]) != 0.0 or float(grid[-2]) <= 0.0):
        kind = "strictly descend" if strict else "descend (ties allowed)"
        raise ValueError(f"grid must {kind} from <= 1 to exactly 0, got {grid.tolist()}")
    return grid


@torch.no_grad()
def euler_rollout(velocity: VelocityFn, noise: Tensor, grid: Tensor, *, record: bool = False,
                  strict_grid: bool = True) -> Rollout:
    """Deterministic Euler through the whole grid (no gradient; states detached). ``strict_grid=False`` accepts the
    ties of a random grid (``_check_grid``)."""
    grid = _check_grid(grid, strict=strict_grid).to(noise.device)
    x = noise
    states = [x.detach().clone()]
    x0s: list[Tensor] = []
    velocities: list[Tensor] = []
    batch = noise.shape[0]
    for i in range(grid.numel() - 1):
        t = grid[i].expand(batch)
        v = velocity(x, t)
        if record:
            # x0 in at least fp32, cast back to the state's dtype
            wide = torch.promote_types(x.dtype, torch.float32)
            velocities.append(v.detach().clone())
            x0s.append(fm.x0_from_v(x.to(wide), v.to(wide), t).to(x.dtype).detach())
        x = fm.euler_step(x, t, grid[i + 1].expand(batch), v)
        states.append(x.detach().clone())
    return Rollout(grid=grid, states=states, final=states[-1], x0s=x0s, velocities=velocities)


@torch.no_grad()
def x0_renoise_rollout(velocity: VelocityFn, noise: Tensor, grid: Tensor, *,
                       draws: Draws | None = None, record: bool = True,
                       stop_before: int | None = None, prefix: str = "rollout") -> Rollout:
    """DMD2's sampler: x0 at every step, fresh noise to the next level; returns the last x0.

    ``stop_before=k`` executes steps ``0..k-1`` only (training's backward simulation); the returned
    ``final`` is then ``x0_{k-1}`` (or the start noise when k = 0, which DMD2 overwrites with the round's noise).
    Re-noised states are stored in the start noise's dtype.
    The fresh noise of step i is the draw ``f"{prefix}.step_noise.{i}"`` (DMD2 draws it even for the last
    executed step of a backward simulation, whose re-noised state is then discarded).
    """
    draws = draws if draws is not None else TorchDraws()
    grid = _check_grid(grid).to(noise.device)
    n = grid.numel() - 1
    last = n if stop_before is None else int(stop_before)
    if not 0 <= last <= n:
        raise ValueError(f"stop_before must lie in [0, {n}]")
    batch = noise.shape[0]
    x = noise
    states = [x.detach().clone()]
    x0s: list[Tensor] = []
    velocities: list[Tensor] = []
    x0 = noise
    for i in range(last):
        t = grid[i].expand(batch)
        v = velocity(x, t)
        # x0 in double as in DMD2 (fp64 latents stay fp64); on a device without float64 (NPU) it runs on the host
        where = fm.float64_device(x.device)
        x0 = fm.x0_from_v(x.to(where, torch.float64), v.to(where, torch.float64), t.to(where)).to(
            device=x.device, dtype=torch.promote_types(x.dtype, torch.float32))
        if record:
            x0s.append(x0.detach().clone())
            velocities.append(v.detach().clone())
        if i + 1 < n or stop_before is not None:
            z = draws.randn(f"{prefix}.step_noise.{i}", x0)
            x = fm.state(x0, z, grid[i + 1].expand(batch)).to(noise.dtype)
            states.append(x.detach().clone())
    return Rollout(grid=grid, states=states, final=x0.detach(), x0s=x0s, velocities=velocities)


@torch.no_grad()
def backward_simulation(velocity: VelocityFn, round_noise: Tensor, grid: Tensor, exit_step: int, *,
                        draws: Draws | None = None, prefix: str = "rollout") -> Rollout:
    """DMD2's training draw for a given (shared) exit step k.

    Start from fresh noise (draw ``f"{prefix}.start"``); run k steps of x0_renoise; the student's input at the exit step
    is the last x0 re-noised to ``grid[k]`` with the **round's** noise, or the round's noise itself when k = 0.
    ``states[k]`` holds that exit input.
    """
    draws = draws if draws is not None else TorchDraws()
    grid = _check_grid(grid).to(round_noise.device)
    start = draws.randn(f"{prefix}.start", round_noise)
    roll = x0_renoise_rollout(velocity, start, grid, draws=draws, record=True, stop_before=exit_step, prefix=prefix)
    if exit_step == 0:
        exit_input = round_noise
    else:
        batch = round_noise.shape[0]
        exit_input = fm.state(roll.final, round_noise, grid[exit_step].expand(batch)).to(round_noise.dtype)
    states = roll.states[:exit_step] + [exit_input.detach().clone()]
    return Rollout(grid=grid, states=states, final=roll.final, x0s=roll.x0s, velocities=roll.velocities,
                   exit_step=int(exit_step))


def sample(velocity: VelocityFn, noise: Tensor, grid: Tensor, sampler: str, *,
           generator: torch.Generator | None = None) -> Tensor:
    """Inference with a method's own sampler (euler or x0_renoise); returns the clean latents.

    For x0_renoise the per-step fresh noise comes from ``generator`` (seeded per image by the evaluation protocol).
    States keep the noise's dtype (Euler: the velocity's), so bf16 noise and a bf16 velocity give a bf16 rollout.
    """
    if sampler == "euler":
        return euler_rollout(velocity, noise, grid).final
    if sampler == "x0_renoise":
        return x0_renoise_rollout(velocity, noise, grid, draws=TorchDraws(generator), record=False).final
    raise ValueError(f"unknown sampler {sampler!r}")


def dynamic_raw_grid(num_steps: int, raw_interior: Sequence[float] | Tensor) -> Tensor:
    """The exact pre-shift grid of this CDM rollout, including its terminal zero."""
    interior = torch.as_tensor(list(raw_interior) if not isinstance(raw_interior, Tensor) else raw_interior,
                               dtype=torch.float32).reshape(-1)
    if interior.numel() != num_steps - 1:
        raise ValueError("a dynamic grid of n steps needs n-1 interior points")
    raw = torch.cat([torch.ones(1), torch.sort(interior, descending=True).values])
    return torch.cat([raw, torch.zeros(1)])


def dynamic_grid(num_steps: int, raw_interior: Sequence[float] | Tensor, warp: Callable[[Tensor, int], Tensor]) -> Tensor:
    """CDM's random raw grid shifted for its sampled step count, then terminal zero."""
    raw = dynamic_raw_grid(num_steps, raw_interior)
    return torch.cat([warp(raw[:-1], num_steps).float(), torch.zeros(1)])


__all__ = ["Rollout", "VelocityFn", "backward_simulation", "dynamic_grid", "dynamic_raw_grid", "euler_rollout", "sample",
           "x0_renoise_rollout"]

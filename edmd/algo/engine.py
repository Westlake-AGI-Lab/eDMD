"""The precise teacher: a guided walk instead of one guided read.

From the query ``(x_t, t)`` the guided teacher is walked ``K`` uniform Euler nodes to 0, ``nodes = t * linspace(1,
1/K, K)`` then 0, and the returned velocity is the step-weighted mean ``v_bar = sum_j (t_j - t_{j+1}) v(x_j, t_j) / t``
-- so ``x_t - t v_bar`` is exactly where the walk lands. ``K = 1`` gives the one-step guided read. Inside the window
(fractions of the last step's input level ``t_{K-1}``, on the last step's reads only) the walk's answer replaces the
one-step read with share ``weight``; outside, the one-step read is kept. Every rank still runs the full walk for every
sample (FSDP needs identical forwards on every rank).
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
from torch import Tensor

from . import fm

VelocityFn = Callable[[Tensor, Tensor], Tensor]


def walk_grid(t: Tensor, nodes: int) -> Tensor:
    """``[nodes + 1, B]`` float32: ``t * linspace(1, 1/K, K)`` then exactly 0 (row 0 is exactly t)."""
    if nodes < 1:
        raise ValueError("nodes must be >= 1")
    t = fm.wide(t.detach()).reshape(-1)
    if (t <= 0).any() or (t > 1).any():
        raise ValueError("precise-teacher query times must lie in (0, 1]")
    endpoint = torch.zeros_like(t)
    scales = torch.linspace(1.0, 1.0 / nodes, steps=nodes, device=t.device, dtype=t.dtype)
    grid_nodes = endpoint.unsqueeze(0) + scales.unsqueeze(1) * (t - endpoint).unsqueeze(0)
    grid = torch.cat([grid_nodes, endpoint.unsqueeze(0)], dim=0)
    grid[0] = t
    grid[-1] = endpoint
    if (grid[1:] >= grid[:-1]).any():
        raise ValueError("precise-teacher nodes must strictly decrease")
    return grid


def walk(guided: VelocityFn, x_t: Tensor, grid: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Euler walk along ``grid``; returns ``(v_bar, v_first, end_state)``.

    Euler updates run in float32 and are cast back to ``x_t``'s dtype before the next query.
    """
    if grid.ndim != 2 or grid.shape[0] < 2 or grid.shape[1] != x_t.shape[0]:
        raise ValueError("grid must have shape [nodes + 1, batch]")
    span = fm.wide(grid[0] - grid[-1])
    shape = (x_t.shape[0],) + (1,) * (x_t.ndim - 1)
    x = x_t
    weighted: Optional[Tensor] = None
    first: Optional[Tensor] = None
    for index in range(grid.shape[0] - 1):
        t_curr, t_next = grid[index], grid[index + 1]
        v = guided(x, t_curr)
        if first is None:
            first = v
        contribution = (fm.wide(t_curr - t_next) / span).reshape(shape) * fm.wide(v)
        weighted = contribution if weighted is None else weighted + contribution
        x = (fm.wide(x) + fm.wide(t_next - t_curr).reshape(shape) * fm.wide(v)).to(dtype=x_t.dtype)
    assert weighted is not None and first is not None
    return weighted.to(dtype=first.dtype), first, x


def window_mask(t: Tensor, index: Tensor, num_steps: int, sigma_grid: Tensor, spec) -> Tensor:
    """Which reads are walked (``PreciseTeacherSpec``): the last sampler step's reads inside the window, whose bounds
    are fractions of ``t_{K-1} = sigma_grid[K-1]``."""
    t = t.detach().float().reshape(-1)
    start = float(sigma_grid[num_steps - 1])
    lo, hi = spec.window.lo * start, spec.window.hi * start
    inside = (t >= lo) & (t <= hi)
    return inside & (index.to(t.device) == num_steps - 1)


def walk_share(t: Tensor, mask: Tensor, spec) -> Tensor:
    """The walk's share per read: ``weight`` inside the window, 0 outside."""
    t = t.detach().float().reshape(-1)
    share = torch.full_like(t, float(spec.weight))
    return torch.where(mask, share, torch.zeros_like(t))


def blend(v_bar: Tensor, v_first: Tensor, share: Tensor) -> Tensor:
    """``v_first + share (v_bar - v_first)`` per sample; rows with share exactly 0 / 1 return that tensor unchanged."""
    shaped = share.reshape((-1,) + (1,) * (v_bar.ndim - 1)).to(v_bar.device)
    mixed = (fm.wide(v_first) + shaped * (fm.wide(v_bar) - fm.wide(v_first))).to(dtype=v_bar.dtype)
    mixed = torch.where(shaped == 1, v_bar, mixed)
    return torch.where(shaped == 0, v_first.to(dtype=v_bar.dtype), mixed)


__all__ = ["blend", "walk", "walk_grid", "walk_share", "window_mask"]

"""Choosing the student's step, drawing query times and building re-noised query points.

Three rules for the query time (``QuerySpec.range``):

* ``interval`` -- TDM: ``t in [max(t_base, lo), upper)``, ``upper = grid[k] - margin`` (separate) or
  ``grid[0] - margin`` (overlap), drawn in the ``space`` grid (sigma, or the raw grid ``linspace(1, 0, K + 1)`` followed
  by the warp -- the flow-matching analogue of TDM's uniform integer timestep). Upper bounds are exclusive after
  rounding (``nextafter``).
* ``uniform`` -- ``t in [lo, hi]`` (truncate: ``lo + u (hi - lo)``) or ``clamp(u, lo, hi)`` (clamp: CDM).
* ``below_step`` -- cleaner than the step's input, D-DMD's CA range ``tau_CA ~ U(t, 1)``: truncate
  ``lo + u (min(hi, c) - lo)`` or CDM's ``clamp(u c, lo, hi)`` with ``c = sigma_k`` (``raw_k`` in raw space).

Query points: ``next_state`` re-noises the step's next on-policy state ``x_base = (1 - s) x0 + s eps`` from
``s = sigma_{k+1}`` with TDM's variance-completing kernel; ``x0`` re-noises the clean prediction with fresh noise
(``t_base = 0``). ``waypoint: calibrated`` (:class:`edmd.config.schema.QuerySpec`) draws every step but the last at
its calibrated law's level (:func:`calibrated_laws`) by the inverse CDF of the same uniform number. TDM's importance
weight uses the effective noise ``(x_t - (1 - t) x0) / t``, which is invariant under the VP -> FM map.

All randomness comes from a named ``Draws`` (``edmd.algo.draws``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import Tensor

from . import fm
from .draws import Draws

Warp = Callable[[Tensor], Tensor]


@dataclass
class Query:
    """A re-noised point: ``x_t`` (detached) at per-sample sigma ``t`` from ``x_base`` at ``t_base``."""

    x_t: Tensor
    t: Tensor
    x_base: Tensor
    t_base: Tensor
    z: Tensor
    x0: Tensor                       # the clean anchor of the effective noise (detached)
    importance: Optional[Tensor] = None
    mixed: Optional[Tensor] = None


def select_steps(mode: str, batch: int, num_steps: int, draws: Draws, *, device: torch.device, name: str,
                 anchor: Optional[Tensor] = None, exit_step: Optional[int] = None) -> Tensor:
    """Per-sample step index k in ``[0, K)``.

    ``per_sample``: uniform per sample (TDM). ``shared``: the rollout's exit step for everyone (DMD2). ``anchor``: the
    rollout-time anchor (CDM). ``final``: always the last step (eDMD's critic source).
    """
    if mode == "per_sample":
        return draws.randint(name, num_steps, (batch,), device=device)
    if mode == "shared":
        if exit_step is None:
            raise ValueError("step_select=shared needs the rollout's exit step")
        return torch.full((batch,), int(exit_step), device=device, dtype=torch.long)
    if mode == "anchor":
        if anchor is None:
            raise ValueError("step_select=anchor needs the rollout's anchor")
        return anchor.to(device=device, dtype=torch.long)
    if mode == "final":
        return torch.full((batch,), num_steps - 1, device=device, dtype=torch.long)
    raise ValueError(f"unknown step selection {mode!r}")


def _nextafter_down(value: Tensor, toward: Tensor) -> Tensor:
    return torch.nextafter(value, toward)


def calibrated_laws(spec, sigma_grid: Tensor):
    """The read law of every step before the last on this grid (:func:`edmd.algo.calibration.step_laws` of
    ``spec.calibration``) on ``[max(sigma_{k+1}, lo), sigma_0 - margin)``, bounds in the time arithmetic's dtype."""
    from .calibration import step_laws

    grid = fm.wide(sigma_grid.detach().to("cpu"))
    upper = float(grid[0] - float(spec.margin))
    return step_laws(spec.calibration, grid, lo=float(spec.lo), upper=upper)


def sample_times(spec, *, index: Tensor, sigma_grid: Tensor, raw_grid: Optional[Tensor], warp: Optional[Warp],
                 draws: Draws, name: str, laws=None) -> tuple[Tensor, Tensor]:
    """Return ``(t_base, t)`` in sigma, both ``[B]`` float32, following ``spec`` (a ``QuerySpec``).

    ``t_base`` is ``sigma_{k+1}`` for ``next_state`` sources and 0 for ``x0``; ``t >= t_base``. ``laws``
    (:func:`calibrated_laws`) is required by ``waypoint: calibrated`` and only read there.
    """
    batch = int(index.shape[0])
    device = index.device
    # time arithmetic in the grid's dtype (float32 in production)
    grid = fm.wide(sigma_grid.to(device=device))
    raw = spec.space == "raw"
    if raw and warp is None:
        raise ValueError("space=raw needs the student grid's warp")
    qgrid = grid
    if raw and spec.range in ("interval", "below_step"):
        if raw_grid is None:
            raise ValueError("space=raw interval/below_step queries need the raw grid")
        qgrid = fm.wide(raw_grid.to(device=device))
    k = index.to(device=device, dtype=torch.long)
    dtype = qgrid.dtype

    def draw_u() -> Tensor:
        return draws.rand(f"{name}.t", (batch,), device=device, dtype=dtype)

    if spec.range == "interval":
        left, right = qgrid[:-1][k], qgrid[1:][k]
        upper = (left if spec.interval == "separate" else qgrid[0].expand_as(left)) - float(spec.margin)
        lower_floor = torch.full_like(right, float(spec.lo))
        base_q = right
        t_base = grid[1:][k] if spec.source == "next_state" else torch.zeros_like(right)
        lower = torch.maximum(base_q, lower_floor)
        if (upper <= lower).any():
            raise ValueError("invalid query interval: target upper must exceed lower (reduce margins / lo)")
        u = draw_u()
        t_q = torch.minimum(lower + u * (upper - lower), _nextafter_down(upper, lower))
        if spec.waypoint == "calibrated":
            # every step but the last at its calibrated law's inverse CDF at the same u (no extra draw)
            if raw or spec.source != "next_state":
                raise ValueError("query.waypoint needs a sigma-space next_state interval query")
            if laws is None:
                raise ValueError("query.waypoint=calibrated needs the per-step laws (queries.calibrated_laws)")
            from .calibration import draw_levels

            last = k == int(grid.shape[0]) - 2
            drawn = draw_levels(laws, k, u, fallback=t_q).to(device=device, dtype=dtype)
            drawn = torch.minimum(torch.maximum(drawn, lower), _nextafter_down(upper, lower))
            t_q = torch.where(last, t_q, drawn)
        t = warp(t_q) if raw else t_q
        if spec.source == "x0":
            t_base = torch.zeros_like(t)
    else:
        u = draw_u()
        if spec.range == "uniform":
            if spec.bound == "truncate":
                t_q = float(spec.lo) + u * (float(spec.hi) - float(spec.lo))
            else:
                t_q = u.clamp(float(spec.lo), float(spec.hi))
        else:  # below_step
            ceiling = qgrid[:-1][k]
            if spec.bound == "truncate":
                top = torch.minimum(ceiling, torch.full_like(ceiling, float(spec.hi)))
                if (top <= float(spec.lo)).any():
                    raise ValueError("below_step: the step's level is not above the lower bound")
                t_q = float(spec.lo) + u * (top - float(spec.lo))
            else:
                t_q = (u * ceiling).clamp(float(spec.lo), float(spec.hi))
        t = warp(t_q) if raw else t_q
        if spec.source == "next_state":
            t_base = grid[1:][k]
            t = torch.maximum(t, t_base)
        else:
            t_base = torch.zeros_like(t)
    return fm.wide(t_base), fm.wide(t)


def importance_weight(mixed: Tensor, reference: Tensor) -> Tensor:
    """TDM's likelihood-ratio weight ``exp(-mean(mix^2)/2) / exp(-mean(z^2)/2)`` per sample (the mean is taken
    *before* exponentiating)."""
    dims = tuple(range(1, mixed.ndim))
    return torch.exp(-0.5 * fm.wide(mixed).square().mean(dim=dims) + 0.5 * fm.wide(reference).square().mean(dim=dims))


def build_query(spec, *, x0: Tensor, eps: Optional[Tensor], index: Tensor, sigma_grid: Tensor,
                raw_grid: Optional[Tensor], warp: Optional[Warp], draws: Draws, name: str,
                compute_importance: bool = False, times: Optional[tuple[Tensor, Tensor]] = None, laws=None) -> Query:
    """Draw the query time and noise and re-noise the step's source point (everything detached).

    ``x0`` / ``eps`` are the step's one-step prediction (``eps`` is only needed for ``next_state``).
    ``times=(t_base, t)`` bypasses the time draw (alignment tests inject a reference's times). ``laws``: the
    per-step read laws of ``waypoint: calibrated`` (:func:`calibrated_laws`).
    """
    x0 = x0.detach()
    if times is None:
        t_base, t = sample_times(spec, index=index, sigma_grid=sigma_grid, raw_grid=raw_grid, warp=warp,
                                 draws=draws, name=name, laws=laws)
    else:
        t_base, t = times
    if spec.source == "next_state":
        if eps is None:
            raise ValueError("next_state queries need the step's eps prediction")
        x_base = fm.state(x0, eps.detach(), t_base)
    else:
        x_base = x0
    z = draws.randn(f"{name}.z", x_base)
    x_t = fm.sde_renoise(x_base, t_base, t, z)
    if spec.source == "next_state" and spec.waypoint == "calibrated":
        # the steps before the last only (the last step's next state is its clean prediction, so its read already is
        # (1 - t) x0 + t z)
        inner = index.to(device=x_t.device) < int(sigma_grid.shape[0]) - 2
        if spec.waypoint_noise == "fresh" and bool(inner.any()):
            # the step's clean prediction re-noised with the fresh noise z drawn above: (1 - t) x0 + t z
            fresh = fm.state(x0, z, t).to(dtype=x_t.dtype)
            x_t = torch.where(fm.bt(inner, x_t), fresh, x_t)
        if spec.waypoint_noise == "own" and bool(inner.any()):
            # the step's own next state carried up to the drawn level (the noise the step hands on is judged too). Run
            # in at least float32 with t >= s: near the rollout point q^2 = t^2 - (A s)^2 is a difference of nearly
            # equal squares, which bf16 turns into spurious noise (and rounding must not put t below s).
            level = torch.where(inner, torch.maximum(t, t_base.to(t.dtype)), t)
            own = fm.sde_renoise(fm.wide(x_base), t_base, level, fm.wide(z)).to(dtype=x_t.dtype)
            x_t = torch.where(fm.bt(inner, x_t), own, x_t)
            t = level
    importance = mixed = None
    if compute_importance:
        mixed = (x_t - (1.0 - fm.bt(t, x_t).to(x_t.dtype)) * x0) / fm.bt(t, x_t).to(x_t.dtype)
        mixed = torch.nan_to_num(mixed)
        importance = importance_weight(mixed, z)
    return Query(x_t=x_t.detach(), t=t, x_base=x_base.detach(), t_base=t_base, z=z, x0=x0,
                 importance=importance, mixed=mixed)


__all__ = ["Query", "build_query", "calibrated_laws", "importance_weight", "sample_times", "select_steps"]

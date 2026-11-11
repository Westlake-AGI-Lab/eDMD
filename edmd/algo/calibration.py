"""The calibrated read law of eDMD (``query.waypoint: calibrated``): where the steps before the last are read.

Offline, once per teacher (``edmd.cli.calibrate``), a table measures for each landing point ``s = sigma_{k+1}`` of the
student grid and coarser levels ``t in [s, 0.99]`` how far the one-step guided answer at ``t`` moves from the answer at
``s`` (the delivered tables use a moment lower bound of the drift part of that change), in units of the conditional
teacher's own error at ``s`` (``d_s(s) = 0``). Step ``k < K - 1`` is then read at ``t ~ p_s(t) ∝ exp(-d_s(t))`` on
``[max(s, lo), upper)`` (``upper = sigma_0 - margin``), with d linear in t between the table's levels and clamped at
its ends: a piecewise-exponential density with closed-form CDF and inverse. The draw is the query's own uniform number
(inverse-CDF sampling: no draw is added, no importance weight). The last step keeps the uniform level.

All arithmetic is float64 on the CPU (a law has a few dozen knots).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from torch import Tensor

#: How far (in sigma) a table row's landing point may sit from the rollout grid's ``sigma_{k+1}`` (Qwen-Image-2.1's
#: aspect buckets move its landing points by up to about 1e-3; another schedule by more than 0.05).
LANDING_TOLERANCE = 5e-3
#: Segments shorter than this (in sigma) are dropped when the knots are merged.
_MIN_SEGMENT = 1e-12


def _phi(x: Tensor) -> Tensor:
    """``(1 - exp(-x)) / x`` with its limit 1 at 0 (float64, stable for |x| small and large)."""
    small = x.abs() < 1e-8
    safe = torch.where(small, torch.ones_like(x), x)
    return torch.where(small, 1.0 - 0.5 * x, -torch.expm1(-safe) / safe)


@dataclass(frozen=True)
class LandingLaw:
    """``p(t) ∝ exp(-d(t))`` on ``[knots[0], knots[-1])`` with d linear between the knots (float64 CPU tensors).

    ``cum[i]`` is the unnormalised mass of ``[knots[0], knots[i]]``."""

    knots: Tensor
    d: Tensor
    cum: Tensor

    @property
    def lower(self) -> float:
        return float(self.knots[0])

    @property
    def upper(self) -> float:
        return float(self.knots[-1])

    @property
    def total(self) -> float:
        return float(self.cum[-1])

    def _segment(self, i: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        a, b = self.knots[i], self.knots[i + 1]
        p, q = self.d[i], self.d[i + 1]
        return a, b - a, p, (q - p) / (b - a)

    def density(self, t: Tensor) -> Tensor:
        """The normalised density at ``t`` (0 outside ``[lower, upper)``)."""
        t = torch.as_tensor(t, dtype=torch.float64).reshape(-1)
        d = torch.from_numpy(_interp(t.numpy(), self.knots.numpy(), self.d.numpy()))
        inside = (t >= self.knots[0]) & (t < self.knots[-1])
        return torch.where(inside, torch.exp(-d) / self.total, torch.zeros_like(t))

    def cdf(self, t: Tensor) -> Tensor:
        """``P(T <= t)`` (closed form, float64)."""
        t = torch.as_tensor(t, dtype=torch.float64).reshape(-1)
        clamped = t.clamp(self.lower, self.upper)
        i = (torch.searchsorted(self.knots, clamped, right=True) - 1).clamp(0, self.knots.numel() - 2)
        a, _, p, g = self._segment(i)
        tau = clamped - a
        partial = torch.exp(-p) * tau * _phi(g * tau)
        return ((self.cum[i] + partial) / self.total).clamp(0.0, 1.0)

    def inverse(self, u: Tensor) -> Tensor:
        """The level whose CDF is ``u`` in [0, 1]: closed-form inverse per segment, clamped into it against rounding."""
        u = torch.as_tensor(u, dtype=torch.float64).reshape(-1).clamp(0.0, 1.0)
        mass = u * self.total
        i = (torch.searchsorted(self.cum, mass, right=True) - 1).clamp(0, self.knots.numel() - 2)
        a, h, p, g = self._segment(i)
        r = (mass - self.cum[i]).clamp_min(0.0) * torch.exp(p)          # mass into the segment, at unit density
        linear = (g * r).abs() < 1e-10
        safe_g = torch.where(linear, torch.ones_like(g), g)
        arg = (-safe_g * r).clamp_min(-1.0 + 1e-16)                       # 1 - g r e^{p} > 0 up to rounding
        tau = torch.where(linear, r, -torch.log1p(arg) / safe_g)
        return a + torch.minimum(tau.clamp_min(0.0), h)

    def mean(self) -> float:
        """``E[t]`` (closed form per segment)."""
        a, h, p, g = self._segment(torch.arange(self.knots.numel() - 1))
        # int_0^h (a + tau) e^{-p - g tau} dtau = e^{-p} [a h phi(g h) + h^2 psi(g h)], psi(x) = int_0^1 s e^{-x s} ds
        x = g * h
        small = x.abs() < 1e-6
        safe = torch.where(small, torch.ones_like(x), x)
        psi = torch.where(small, 0.5 - x / 3.0, (1.0 - torch.exp(-safe) * (1.0 + safe)) / safe.square())
        first = torch.exp(-p) * (a * h * _phi(x) + h.square() * psi)
        return float(first.sum() / self.total)

    def mass_within(self, width: float) -> float:
        """``P(T < lower + width)``."""
        return float(self.cdf(torch.tensor([self.lower + float(width)], dtype=torch.float64))[0])

    def entropy(self) -> float:
        """Differential entropy (nats), by 24-node Gauss-Legendre quadrature per segment."""
        nodes, weights = _gauss_legendre(24)
        total = 0.0
        for i in range(self.knots.numel() - 1):
            a, b = float(self.knots[i]), float(self.knots[i + 1])
            t = torch.tensor([a + 0.5 * (b - a) * (1.0 + x) for x in nodes], dtype=torch.float64)
            dens = self.density(torch.minimum(t, torch.tensor(math.nextafter(b, a), dtype=torch.float64)))
            value = torch.where(dens > 0, -dens * torch.log(dens), torch.zeros_like(dens))
            total += 0.5 * (b - a) * float((value * torch.tensor(weights, dtype=torch.float64)).sum())
        return total

    def summary(self, near: float = 0.05) -> dict:
        """Readings of one step's law: the mean read level and its offset above the landing point, the mass within
        ``+near`` of it, the mean offset as a fraction of the window (uniform: 0.5) and the effective support fraction
        ``exp(H) / (upper - lower)`` (uniform: 1)."""
        width = self.upper - self.lower
        mean = self.mean()
        return {"landing": self.lower, "upper": self.upper, "mean": mean, "mean_offset": mean - self.lower,
                f"mass_within_{near:g}": self.mass_within(near), "mean_offset_fraction": (mean - self.lower) / width,
                "uniform_mass_within": min(1.0, near / width),
                "effective_support_fraction": math.exp(self.entropy()) / width}


_GL_CACHE: dict[int, tuple[list[float], list[float]]] = {}


def _gauss_legendre(n: int) -> tuple[list[float], list[float]]:
    if n not in _GL_CACHE:
        import numpy as np

        x, w = np.polynomial.legendre.leggauss(n)
        _GL_CACHE[n] = ([float(v) for v in x], [float(v) for v in w])
    return _GL_CACHE[n]


def _interp(x, xp, fp):
    import numpy as np

    return np.interp(np.asarray(x, dtype=np.float64), np.asarray(xp, dtype=np.float64), np.asarray(fp, dtype=np.float64))


def landing_law(levels: Sequence[float], d: Sequence[float], *, lower: float, upper: float) -> LandingLaw:
    """The law on ``[lower, upper)`` from one table row: knots = ``lower``, the row's levels strictly inside, ``upper``;
    d at a knot = the row linearly interpolated in t, clamped at its first / last level."""
    levels = [float(v) for v in levels]
    values = [float(v) for v in d]
    if len(levels) < 2 or len(values) != len(levels) or any(b <= a for a, b in zip(levels, levels[1:])):
        raise ValueError("a calibration row needs >= 2 strictly increasing levels and one d per level")
    if not (math.isfinite(lower) and math.isfinite(upper) and upper > lower):
        raise ValueError(f"empty calibrated window [{lower}, {upper})")
    if any((not math.isfinite(v)) or v < 0 for v in values):
        raise ValueError("calibration d must be finite and >= 0")
    inner = [t for t in levels if lower + _MIN_SEGMENT < t < upper - _MIN_SEGMENT]
    knots = [lower] + inner + [upper]
    knots_t = torch.tensor(knots, dtype=torch.float64)
    d_t = torch.from_numpy(_interp(knots, levels, values))
    a, b = knots_t[:-1], knots_t[1:]
    p, q = d_t[:-1], d_t[1:]
    masses = (b - a) * torch.exp(-p) * _phi(q - p)
    cum = torch.cat([torch.zeros(1, dtype=torch.float64), torch.cumsum(masses, 0)])
    if not bool(torch.isfinite(cum).all()) or float(cum[-1]) <= 0:
        raise ValueError("the calibrated law has no mass")
    return LandingLaw(knots=knots_t, d=d_t, cum=cum)


def step_laws(calibration, sigma_grid: Tensor, *, lo: float, upper: float,
              tolerance: float = LANDING_TOLERANCE) -> tuple[LandingLaw, ...]:
    """One law per step before the last: row k of ``calibration`` on ``[max(sigma_{k+1}, lo), upper)`` (float32 grid
    values, as float64). Raises if the table's landing points differ from this grid's by more than ``tolerance``."""
    grid = sigma_grid.detach().to(device="cpu", dtype=torch.float32)
    steps = int(grid.shape[0]) - 1
    if len(calibration.landing) != steps - 1:
        raise ValueError(f"query.calibration has {len(calibration.landing)} rows; this {steps}-step grid has "
                         f"{steps - 1} steps before the last")
    laws = []
    for k in range(steps - 1):
        s = float(grid[k + 1])
        if abs(s - float(calibration.landing[k])) > tolerance:
            raise ValueError(f"query.calibration row {k} was measured at landing point {calibration.landing[k]:.6g}; "
                             f"this grid lands step {k} at {s:.6g} (another schedule or model: measure its table)")
        laws.append(landing_law(calibration.levels[k], calibration.d[k], lower=max(s, float(lo)), upper=float(upper)))
    return tuple(laws)


def draw_levels(laws: Sequence[LandingLaw], index: Tensor, u: Tensor, *, fallback: Tensor) -> Tensor:
    """Per row: ``laws[k].inverse(u)`` for the rows whose step ``k`` has a law, ``fallback`` for the others (the last
    step). Float64 on the CPU, ``[B]``."""
    k = index.detach().to(device="cpu", dtype=torch.long).reshape(-1)
    uu = u.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    out = fallback.detach().to(device="cpu", dtype=torch.float64).reshape(-1).clone()
    for j, law in enumerate(laws):
        rows = k == j
        if law is not None and bool(rows.any()):
            out[rows] = law.inverse(uu[rows])
    return out


def readings(calibration, sigma_grid: Tensor, *, lo: float = 0.02, upper: Optional[float] = None,
             near: float = 0.05) -> list[dict]:
    """The readings of a table on a grid (:meth:`LandingLaw.summary` per step before the last)."""
    if upper is None:
        upper = float(torch.tensor(1.0, dtype=torch.float32) - 0.01)
    return [law.summary(near) for law in step_laws(calibration, sigma_grid, lo=lo, upper=upper)]


__all__ = ["LANDING_TOLERANCE", "LandingLaw", "draw_levels", "landing_law", "readings", "step_laws"]

"""Fake (critic) regression losses.

Two labels (``CriticSpec.label``):

* ``x0`` -- native TDM: regress the fake's x0 read on the student's one-step x0 prediction, weighted per sample by
  ``min(SNR(t), cap)`` and the importance weight. The label is biased for ``s > 0`` (it is the x0 of the phase, not the
  posterior of ``x_t``); only the TDM presets use it.
* ``fresh`` -- the unbiased fresh-noise label: with ``A = (1-t)/(1-s)``, ``q^2 = t^2 - (A s)^2``,
  ``residual = q v - [(t + (1-t) s^2/(1-s)^2) z - (q/(1-s)) x_base]``. At ``s = 0`` this is ``t (v - (z - x0))``,
  i.e. the flow-matching label scaled by t, so the weights below reproduce the baselines' losses:

  ========================  ======================  =============================================
  kind                      weight on residual^2    at s = 0, on the velocity error ``|v-(z-x0)|^2``
  ========================  ======================  =============================================
  eps_space                 ((1-t)/t)^2             (1-t)^2  = DMD2's eps-MSE
  eps_space+terminal_clamp  clamp(SNR, lo, hi)@s=0  eDMD's clamp(SNR, 1, 10); lo None: min(SNR, hi)
  v_space                   1/q^2                   1        = CDM's velocity MSE
  ========================  ======================  =============================================

Coefficients and labels are detached float32; only the fake velocity carries a gradient.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from . import fm

ALPHA_FLOOR = fm.ALPHA_FLOOR


def _per_sample(value: Tensor) -> Tensor:
    return value.reshape(value.shape[0], -1).mean(dim=1)


def x0_label_loss(fake_v: Tensor, x_t: Tensor, t: Tensor, x0_label: Tensor, *, cap: Optional[float] = 5.0,
                  importance: Optional[Tensor] = None) -> tuple[Tensor, dict]:
    """TDM's fake loss: ``mean_b( w_b * mean((x_t - t v_F - x0)^2) )``, ``w = min(SNR, cap)`` [* IS]."""
    batch = fake_v.shape[0]
    fake_x0 = x_t - t.reshape(batch, *([1] * (x_t.ndim - 1))) * fake_v
    err = (fm.wide(fake_x0) - fm.wide(x0_label)).square()
    per_sample = _per_sample(err)
    snr = fm.snr(t)
    if cap is not None:
        snr = snr.clamp(max=float(cap))
    # TDM's operation order: (per_sample * importance) * snr
    loss = (per_sample * snr).mean() if importance is None else (per_sample * fm.wide(importance) * snr).mean()
    metrics = {"critic/x0_mse": err.detach().mean(), "critic/snr_weight_mean": snr.detach().mean()}
    if importance is not None:
        metrics["critic/importance_mean"] = importance.detach().float().mean()
    return loss, metrics


def fresh_label_loss(fake_v: Tensor, x_base: Tensor, t_base: Tensor, t: Tensor, z: Tensor, *, kind: str,
                     terminal_clamp: Optional[tuple[Optional[float], float]] = None) -> tuple[Tensor, dict]:
    """The unbiased fresh-noise regression (weights per the module table)."""
    if fake_v.ndim < 2 or fake_v.shape != x_base.shape or x_base.shape != z.shape:
        raise ValueError("fake_v, x_base and z must share a batched latent shape.")
    batch = fake_v.shape[0]
    device = fake_v.device
    time_dtype = torch.promote_types(torch.promote_types(torch.as_tensor(t).dtype, fake_v.dtype), torch.float32)

    def times(value: Tensor) -> Tensor:
        value = torch.as_tensor(value, device=device).detach().to(time_dtype)
        return value.reshape(1).expand(batch) if value.numel() == 1 else value

    def expand(value: Tensor) -> Tensor:
        return value.reshape((batch,) + (1,) * (fake_v.ndim - 1))

    with torch.no_grad():
        s, tt = times(t_base), times(t)
        valid = torch.isfinite(s) & torch.isfinite(tt) & (s >= 0) & (tt >= s) & (tt < 1) & ((1.0 - s) >= ALPHA_FLOOR)
        if not valid.all():
            raise ValueError("fresh-label loss requires 0 <= t_base <= t < 1 and 1 - t_base >= 1e-6")
        a_s, a_t = 1.0 - s, 1.0 - tt
        scale = a_t / a_s
        variance = (tt.square() - (scale * s).square()).clamp_min(0.0)
        sigma = variance.sqrt()
        target = (expand(tt + a_t * s.square() / a_s.square()) * fm.wide(z.detach())
                  - expand(sigma / a_s) * fm.wide(x_base.detach()))
        positive_t = tt.clamp_min(ALPHA_FLOOR)
        if kind == "eps_space":
            weight = (a_t / positive_t).square()
            if terminal_clamp is not None:
                lo, hi = terminal_clamp                    # lo None: no floor, min(SNR, hi)
                terminal = s == 0
                weight = torch.where(terminal, weight.clamp(min=None if lo is None else float(lo), max=float(hi)),
                                     weight)
        elif kind == "v_space":
            weight = variance.clamp_min(ALPHA_FLOOR ** 2).reciprocal()
        else:
            raise ValueError(f"unknown fresh-label weight kind {kind!r}")
    residual = expand(sigma) * fm.wide(fake_v) - target
    per_sample = _per_sample(residual.square())
    loss = (weight * per_sample).mean()
    metrics = {"critic/scaled_residual_mse": per_sample.detach().mean(), "critic/weight_mean": weight.mean(),
               "critic/local_sigma_mean": sigma.mean()}
    if terminal_clamp is not None:
        metrics["critic/effective_velocity_weight_mean"] = (weight * variance).mean()
    return loss, metrics


def critic_loss(spec, *, fake_v: Tensor, x_t: Tensor, t: Tensor, x_base: Tensor, t_base: Tensor, z: Tensor,
                x0_label: Tensor, importance: Optional[Tensor]) -> tuple[Tensor, dict]:
    """Dispatch on ``CriticSpec`` (label + weight)."""
    weight = spec.weight
    if spec.label == "x0":
        return x0_label_loss(fake_v, x_t, t, x0_label, cap=weight.cap,
                             importance=importance if weight.importance else None)
    return fresh_label_loss(fake_v, x_base, t_base, t, z, kind=weight.kind, terminal_clamp=weight.terminal_clamp)


__all__ = ["critic_loss", "fresh_label_loss", "x0_label_loss"]

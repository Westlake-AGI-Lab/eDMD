"""Flow-matching conventions shared by all models (pure tensor algebra, no model code).

``x_t = (1 - t) x0 + t eps``, ``v = eps - x0``, so ``x0 = x_t - t v`` and ``eps = x_t + (1 - t) v``.
t = 0 is clean, t = 1 is pure noise. DDPM baselines map onto this: TDM's DDIM step is the deterministic Euler step
``x + (t_next - t) v``, DMD2's "predict x0, re-noise with fresh noise" is ``(1 - t_next) x0 + t_next z``, and the VP SNR
``alpha^2 / sigma^2`` equals ``((1 - t) / t)^2`` pointwise.

Time tensors are ``[B]`` (or scalars) and broadcast over the non-batch dims of ``[B, ...]`` latents.
"""

from __future__ import annotations

import torch
from torch import Tensor

ALPHA_FLOOR = 1e-6  # floor of 1 - t_from in the SDE kernel


def wide(x: Tensor) -> Tensor:
    """At least float32: bf16 -> fp32; fp32 and fp64 are unchanged (``.float()`` would narrow fp64)."""
    return x.to(torch.promote_types(x.dtype, torch.float32))


_FLOAT64: dict[str, bool] = {}


def float64_ok(device: torch.device | str) -> bool:
    """Whether ``device`` really computes float64 (on the Ascend NPU ``x.double()`` silently returns float32)."""
    kind = torch.device(device).type
    if kind not in _FLOAT64:
        _FLOAT64[kind] = kind in ("cpu", "cuda") or torch.ones(1, device=device).double().dtype == torch.float64
    return _FLOAT64[kind]


def float64_device(device: torch.device | str) -> torch.device:
    """Where a float64 step runs: ``device`` itself, or the host when ``device`` has no float64."""
    device = torch.device(device)
    return device if float64_ok(device) else torch.device("cpu")


def bt(t: Tensor | float, like: Tensor) -> Tensor:
    """Broadcast per-sample times over ``like``'s non-batch dims (float, on ``like``'s device)."""
    if not isinstance(t, Tensor):
        t = torch.tensor(float(t), device=like.device, dtype=torch.float32)
    t = t.to(device=like.device)
    if t.ndim == 0:
        t = t.expand(like.shape[0])
    if t.ndim != 1 or t.shape[0] != like.shape[0]:
        raise ValueError(f"times must be scalar or [B={like.shape[0]}], got {tuple(t.shape)}")
    return t.reshape(t.shape[0], *([1] * (like.ndim - 1)))


def state(x0: Tensor, eps: Tensor, t: Tensor | float) -> Tensor:
    """``(1 - t) x0 + t eps``."""
    te = bt(t, x0).to(x0.dtype)
    return (1.0 - te) * x0 + te * eps


def x0_from_v(x: Tensor, v: Tensor, t: Tensor | float) -> Tensor:
    return x - bt(t, x).to(x.dtype) * v


def euler_step(x: Tensor, t: Tensor | float, t_next: Tensor | float, v: Tensor) -> Tensor:
    """``x + (t_next - t) v`` in FlowMatchEulerDiscreteScheduler's precision order (fp32 sigmas, product in the
    velocity's dtype, fp32 accumulation, cast back)."""
    time_dtype = torch.promote_types(torch.float32, v.dtype)
    t0 = bt(t, x).to(time_dtype)
    t1 = bt(t_next, x).to(time_dtype)
    dt = t1 - t0
    if v.dtype != time_dtype and bool((dt == dt.reshape(-1)[:1]).all()):
        # A low-precision velocity with one time for all samples (an evaluation rollout): multiply by the 0-dim dt as
        # the scheduler does, since backends round this mixed op differently (CPU casts dt to bf16, torch_npu rounds
        # the fp32 product once).
        delta = dt.reshape(-1)[0] * v
    else:
        delta = dt.to(v.dtype) * v
    return (wide(x) + delta).to(v.dtype)


def sde_renoise(x: Tensor, t_from: Tensor | float, t_to: Tensor | float, z: Tensor) -> Tensor:
    """TDM's variance-completing forward kernel from ``t_from`` to ``t_to >= t_from``.

    ``A = (1 - t_to) / (1 - t_from)``, ``q^2 = t_to^2 - (A t_from)^2`` (clamped at 0),
    ``x_to = A x + q z``. With ``t_from = 0`` this is ``(1 - t_to) x + t_to z``.
    """
    tf = bt(t_from, x).to(x.dtype)
    tt = bt(t_to, x).to(x.dtype)
    scale = (1.0 - tt) / (1.0 - tf).clamp_min(ALPHA_FLOOR)
    variance = (tt.square() - (scale * tf).square()).clamp_min(0.0)
    return scale * x + variance.sqrt() * z


def snr(t: Tensor) -> Tensor:
    """``((1 - t) / t)^2``, the SNR of linear flow matching."""
    t = wide(t)
    return ((1.0 - t) / t.clamp_min(ALPHA_FLOOR)).square()


__all__ = ["ALPHA_FLOOR", "bt", "wide", "euler_step", "sde_renoise", "snr", "state", "x0_from_v"]

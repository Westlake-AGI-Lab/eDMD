"""The student's x0-space corrections, their normalizers and the surrogate loss.

Every method writes the student update as a detached target on the step's clean prediction ``x0`` (the only
tensor with a graph back to the student): ``x0 + C`` with a correction ``C`` read at a re-noised query.

Corrections (``TermSpec.kind``):

* ``coupled`` (TDM, DMD2): ``C = (T_c - F) + (w - 1)(T_c - T_u)``, reference ``T^w = T_c + (w - 1)(T_c - T_u)``; with
  the precise teacher the guided read is replaced by the (blended) walk read ``T_read`` and ``C = T_read - F``.
* ``dm`` = ``T_c - F`` (reference ``T_c``); ``ca`` = ``x0(guide(v_c, v_u, w)) - T_c`` (= ``(w - 1)(T_c - T_u)`` for
  linear CFG).

Normalizer per term: ``D = mean|x0 - reference|`` over the non-batch dims (+eps, >= floor, in float32 or float64).
Loss per term, with ``form`` = square or ``sqrt(e^2 + c^2) - c``:

* ``placement=loss``   ``scale * mean(form(x0 - sg(x0 + C)) / D)``        (TDM, eDMD)
* ``placement=target`` ``scale * mean(form(x0 - sg(x0 + C / D)))``        (DMD2 with scale 0.5, CDM, D-DMD)

For MSE the two placements have the same gradient ``-2 scale C / (D N)`` and differ only in the logged value; for
Huber they differ (TDM's Huber loss is the ``loss`` placement).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from . import fm


@dataclass
class TermReading:
    """One term's contribution before normalization."""

    kind: str
    anchor: Tensor                 # x0 with a graph to the student (the step's, or CDM's off-trajectory one)
    correction: Tensor             # detached x0-space correction
    reference: Optional[Tensor]    # detached reference read for the normalizer
    weight: float = 1.0
    t: Optional[Tensor] = None     # query sigma (metrics)
    t_base: Optional[Tensor] = None  # the query's base sigma (0 = a clean-sample query)


def coupled_correction(teacher_cond_x0: Tensor, fake_x0: Tensor, teacher_uncond_x0: Optional[Tensor],
                       guidance_scale: float) -> tuple[Tensor, Tensor]:
    """``(C, T^w)``: the coupled correction and its guided normalizer reference."""
    correction = teacher_cond_x0 - fake_x0
    guided = teacher_cond_x0
    if teacher_uncond_x0 is not None:
        ca = (float(guidance_scale) - 1.0) * (teacher_cond_x0 - teacher_uncond_x0)
        correction = correction + ca
        guided = teacher_cond_x0 + ca
    elif float(guidance_scale) != 1.0:
        raise ValueError("the unconditional teacher read is required when cfg != 1")
    return correction.detach(), guided.detach()


def normalizer(anchor: Tensor, reference: Optional[Tensor], spec) -> Tensor:
    """``D`` of ``NormalizationSpec``; shape [B, 1, ...], detached."""
    if reference is None:
        raise ValueError("per-term normalization needs a reference read")
    dims = tuple(range(1, anchor.ndim))
    if spec.dtype == "float64":   # on a device without float64 (NPU) the reduction runs on the host in float64
        where = fm.float64_device(anchor.device)
        factor = (anchor.detach().to(where, torch.float64) - reference.detach().to(where, torch.float64)).abs() \
            .mean(dim=dims, keepdim=True)
    else:
        dtype = torch.promote_types(anchor.dtype, torch.float32)
        factor = (anchor.detach().to(dtype) - reference.detach().to(dtype)).abs().mean(dim=dims, keepdim=True)
    if spec.eps:
        factor = factor + float(spec.eps)
    if spec.floor is not None:
        factor = factor.clamp(min=float(spec.floor))
    if not spec.nan_to_num and ((factor <= 0).any() or not torch.isfinite(factor).all()):
        raise FloatingPointError("DMD normalizer became zero or non-finite (set a floor/eps only if intended)")
    if factor.device != anchor.device:   # back to the device, in the widest dtype it has
        factor = factor.to(device=anchor.device, dtype=torch.float64 if fm.float64_ok(anchor.device) else torch.float32)
    return factor.detach()


def surrogate_loss(anchor: Tensor, correction: Tensor, factor: Optional[Tensor], loss_spec, *,
                   nan_to_num: bool = False) -> Tensor:
    """One term's surrogate (see module docstring); gradient flows only through ``anchor``."""
    if loss_spec.placement == "target":
        step = correction.detach() if factor is None else correction.detach().to(factor.dtype) / factor
        if nan_to_num:
            step = torch.nan_to_num(step)
        target = (anchor.detach().to(step.dtype) + step).detach()
        err = fm.wide(anchor) - fm.wide(target)
        elem = err.square() if loss_spec.form == "mse" else (err.square() + loss_spec.huber_c ** 2).sqrt() - loss_spec.huber_c
    else:
        target = (anchor.detach() + correction.detach()).detach()
        err = fm.wide(anchor) - fm.wide(target)
        elem = err.square() if loss_spec.form == "mse" else (err.square() + loss_spec.huber_c ** 2).sqrt() - loss_spec.huber_c
        if factor is not None:
            elem = elem / factor
            if nan_to_num:
                elem = torch.nan_to_num(elem)
    return float(loss_spec.scale) * elem.mean()


def generator_loss(readings: list[TermReading], norm_spec, loss_spec) -> tuple[Tensor, dict]:
    """``sum_terms weight * surrogate``, and per-term metrics (loss, D mean, correction L1, query sigma)."""
    total: Optional[Tensor] = None
    metrics: dict[str, Tensor] = {}
    for index, reading in enumerate(readings):
        factor = normalizer(reading.anchor, reading.reference, norm_spec)
        loss = surrogate_loss(reading.anchor, reading.correction, factor, loss_spec, nan_to_num=norm_spec.nan_to_num)
        weighted = loss * float(reading.weight)
        total = weighted if total is None else total + weighted
        tag = f"generator/{index}_{reading.kind}"
        metrics[f"{tag}/loss"] = loss.detach()
        metrics[f"{tag}/correction_l1"] = reading.correction.detach().float().abs().mean()
        metrics[f"{tag}/normalizer_mean"] = factor.detach().float().mean()
        if reading.t is not None:
            metrics[f"{tag}/sigma_mean"] = reading.t.detach().float().mean()
    assert total is not None
    metrics["loss/generator"] = total.detach()
    return total, metrics


__all__ = ["TermReading", "coupled_correction", "generator_loss", "normalizer", "surrogate_loss"]

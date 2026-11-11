"""The configuration schema: model + method + run (+ the optional eval), every setting under its component.

Algorithm fields of :class:`MethodSpec` without a default are required (the presets in ``configs/methods/`` spell them
out). Fields that apply only to some component values are validated in ``__post_init__``: an inapplicable non-default
value raises instead of being ignored, except the sub-keys of a feature turned off by its ``enabled`` flag.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from typing import Literal, Optional

Space = Literal["sigma", "raw"]


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _unit(value: float, name: str) -> None:
    _check(math.isfinite(value) and 0.0 <= value <= 1.0, f"{name} must lie in [0, 1], got {value}")


# ============================================================================ method: queries
@dataclass(frozen=True)
class CalibrationSpec:
    """The teacher's calibration table for the calibrated read law (``query.waypoint: calibrated``).

    Measured once per teacher (``edmd.cli.calibrate``) on its own samples at its default guidance: for every landing
    point ``s = sigma_{k+1}`` of the student grid (one row per step before the last) and coarser levels ``t``
    (``levels[k]``, increasing from ``s`` to the top of the read window), ``d[k]`` is the part of the change of the
    one-step guided answer between ``s`` and ``t`` that the coarser read cannot explain by the information it lost -- a
    moment lower bound, in units of the conditional teacher's own error at ``s`` (``d_s(s) = 0``). Step ``k`` is then
    read at ``t ~ exp(-d_s(t))`` on ``[s, sigma_0 - margin)`` (``edmd.algo.calibration``). ``cfg`` is the guidance the
    table was measured at and must equal ``engine.cfg``. A run file names the table (``calibration: <model>``, i.e.
    ``configs/calibration/<model>.json``); composing the config reads its numbers in.
    """

    landing: tuple[float, ...]
    levels: tuple[tuple[float, ...], ...]
    d: tuple[tuple[float, ...], ...]
    cfg: float

    def __post_init__(self) -> None:
        key = "query.calibration"
        _check(len(self.landing) > 0, f"{key} has no measured table (empty landing): run edmd.cli.calibrate first")
        n = len(self.landing)
        _check(len(self.levels) == n and len(self.d) == n, f"{key} needs one levels row and one d row per landing point")
        _check(all(math.isfinite(s) and 0.0 < s < 1.0 for s in self.landing)
               and all(a > b for a, b in zip(self.landing, self.landing[1:])),
               f"{key}.landing must decrease strictly inside (0, 1) (sigma_1, sigma_2, ...)")
        for k, (row, values, s) in enumerate(zip(self.levels, self.d, self.landing)):
            _check(len(row) >= 2 and len(values) == len(row), f"{key} row {k} needs >= 2 levels and one d per level")
            _check(all(math.isfinite(t) and 0.0 < t < 1.0 for t in row) and all(a < b for a, b in zip(row, row[1:])),
                   f"{key}.levels[{k}] must increase strictly inside (0, 1)")
            _check(abs(row[0] - s) <= 1e-6, f"{key}.levels[{k}] must start at its landing point {s:g}")
            _check(all(math.isfinite(v) and v >= 0.0 for v in values), f"{key}.d[{k}] must be finite and >= 0")
        _check(math.isfinite(self.cfg) and self.cfg >= 1.0, f"{key}.cfg must be >= 1")


@dataclass(frozen=True)
class QuerySpec:
    """Where a teacher / fake read is taken: a re-noised point built from the student's selected step.

    * ``source``: ``next_state`` re-noises the step's next on-policy state ``(1 - s) x0 + s eps`` (``s =
      sigma_{k+1}``) to the query level with TDM's variance-completing kernel; ``x0`` re-noises the step's clean
      prediction with fresh noise (``t_base = 0``; DMD2).
    * ``range``: ``interval`` -- TDM's ``t in [max(t_base, lo), upper)`` with ``upper = sigma_k - margin``
      (``separate``) or ``sigma_0 - margin`` (``overlap``); ``uniform`` -- ``t in [lo, hi]``; ``below_step`` --
      cleaner than the step's input level (D-DMD's CA range).
    * ``bound``: ``truncate`` draws inside the bounds; ``clamp`` draws on (0, 1) and clamps (CDM's ``sample_sigmas``).
    * ``space``: ``sigma`` draws the noise level directly; ``raw`` draws the model's raw time and warps it with the
      student grid's shift (the flow-matching analogue of a uniform integer timestep).
    * ``waypoint: calibrated`` (eDMD; student ``next_state`` overlap reads in sigma space): every step but the last is
      read at a level drawn from the teacher's calibrated law (``calibration``) by the inverse CDF of the query's own
      uniform number; the last step keeps its uniform level. ``waypoint_noise`` picks the point those reads use:
      ``fresh`` -- the step's clean prediction re-noised with the query's fresh noise, ``(1 - t) x0_k + t z``;
      ``own`` -- the step's own next state carried up to ``t`` by TDM's forward kernel (so the noise the step hands on
      is judged too).
    """

    source: Literal["next_state", "x0"]
    range: Literal["interval", "uniform", "below_step"]
    space: Space
    lo: float
    hi: float = 1.0
    interval: Optional[Literal["separate", "overlap"]] = None
    margin: float = 0.0
    bound: Literal["truncate", "clamp"] = "truncate"
    waypoint: Optional[Literal["calibrated"]] = None
    waypoint_noise: Literal["own", "fresh"] = "own"
    calibration: Optional[CalibrationSpec] = None

    def __post_init__(self) -> None:
        _check(self.waypoint in (None, "calibrated"), "query.waypoint must be null or calibrated")
        _check(self.waypoint_noise in ("own", "fresh"), "query.waypoint_noise must be own or fresh")
        if self.waypoint == "calibrated":
            _check(self.source == "next_state" and self.range == "interval" and self.interval == "overlap"
                   and self.space == "sigma",
                   "query.waypoint=calibrated reads a sigma-space next_state overlap interval query")
            _check(self.calibration is not None,
                   "query.waypoint=calibrated draws each step's level from query.calibration (the teacher's table)")
            top = 1.0 - float(self.margin)
            _check(all(row[-1] >= top - 1e-6 for row in self.calibration.levels),
                   f"query.calibration must cover every level up to the window's top {top:g} (sigma_0 - margin)")
        else:
            _check(self.calibration is None, "query.calibration is only read by query.waypoint=calibrated")
            _check(self.waypoint_noise == "own", "query.waypoint_noise only applies to query.waypoint=calibrated")
        _unit(self.lo, "query.lo")
        _unit(self.hi, "query.hi")
        _check(self.lo < self.hi or self.range == "interval", "query.lo must be below query.hi")
        if self.range == "interval":
            _check(self.interval is not None, "query.range=interval needs query.interval (separate|overlap)")
            _check(self.bound == "truncate", "interval queries are always truncated (TDM draws inside the band)")
            _check(math.isfinite(self.margin) and 0.0 <= self.margin < 1.0, "query.margin must lie in [0, 1)")
        else:
            _check(self.interval is None, "query.interval only applies to range=interval")
            _check(self.margin == 0.0, "query.margin only applies to range=interval")


# ============================================================================ method: rollout
@dataclass(frozen=True)
class RolloutSpec:
    """The student's sampler during training (and at inference).

    ``euler``: deterministic ``x <- x + (s' - s) v`` (TDM's DDIM, eDMD, D-DMD, CDM). ``x0_renoise`` with
    ``to_exit``: DMD2's backward simulation -- predict x0, then ``(1 - s') x0 + s' z`` with fresh ``z`` -- up to a
    shared exit step. ``grid: dynamic`` is CDM's random grid: ``n ~ U{1..dynamic_max_steps}`` (shared by all ranks),
    interior raw points ``U(dynamic_raw_range)`` sorted descending and warped with the model's shift for an
    ``n``-step sampler.
    """

    sampler: Literal["euler", "x0_renoise"]
    grid: Literal["fixed", "dynamic"]
    to_exit: bool
    dynamic_max_steps: int = 28
    dynamic_raw_range: tuple[float, float] = (0.25, 0.95)

    def __post_init__(self) -> None:
        if self.grid == "dynamic":
            _check(self.dynamic_max_steps >= 1, "rollout.dynamic_max_steps must be >= 1")
            lo, hi = self.dynamic_raw_range
            _check(0.0 < lo < hi < 1.0, "rollout.dynamic_raw_range must satisfy 0 < lo < hi < 1")
        else:
            _check(self.dynamic_max_steps == 28 and tuple(self.dynamic_raw_range) == (0.25, 0.95),
                   "rollout.dynamic_* only apply to grid=dynamic")
        _check(self.sampler != "x0_renoise" or (self.to_exit and self.grid == "fixed"),
               "rollout.sampler=x0_renoise is DMD2's backward simulation on the fixed grid (to_exit: true)")
        _check(not self.to_exit or self.sampler == "x0_renoise", "rollout.to_exit is the x0_renoise backward simulation")


# ============================================================================ method: critic
@dataclass(frozen=True)
class CriticWeightSpec:
    """Per-sample weight of the fake's regression.

    ``label: x0`` (TDM): ``min_snr`` = ``min(SNR, cap)``, times TDM's importance weight when ``importance`` is on.
    ``label: fresh`` (the residual of the fresh-noise label): ``eps_space`` = ``((1 - t) / t)^2`` (DMD2's eps-MSE at
    ``s = 0``; ``terminal_clamp: [lo, hi]`` clamps it, ``lo: null`` = no floor); ``v_space`` = ``1 / q^2`` (a plain
    velocity MSE: CDM, D-DMD).
    """

    kind: Literal["min_snr", "eps_space", "v_space"]
    cap: Optional[float] = None
    importance: bool = False
    terminal_clamp: Optional[tuple[Optional[float], float]] = None

    def __post_init__(self) -> None:
        _check(self.importance is False or self.kind == "min_snr", "critic.weight.importance only with kind=min_snr")
        _check(self.cap is None or self.kind == "min_snr", "critic.weight.cap only with kind=min_snr")
        _check(self.cap is None or (math.isfinite(self.cap) and self.cap > 0), "critic.weight.cap must be positive")
        _check(self.terminal_clamp is None or self.kind == "eps_space", "critic.weight.terminal_clamp only with eps_space")
        if self.terminal_clamp is not None:
            lo, hi = self.terminal_clamp
            _check(math.isfinite(hi) and hi > 0 and (lo is None or 0.0 < lo <= hi),
                   "critic.weight.terminal_clamp must satisfy 0 < lo <= hi (lo null: no floor)")


@dataclass(frozen=True)
class CriticSpec:
    """What the fake (critic) is trained on.

    ``source``: ``phase_state`` (TDM: a random step's next state, recomputed from the student's one-step prediction),
    ``final`` (eDMD: the rollout's final output, the delivered image), ``exit_x0`` (DMD2: the exit step's clean
    prediction), ``anchor_x0`` (CDM / D-DMD: the one-step prediction at the rollout anchor). ``merge_student`` (CDM):
    ``fake <- d fake + (1 - d) student`` after every fake step.
    """

    source: Literal["phase_state", "final", "exit_x0", "anchor_x0"]
    query: QuerySpec
    label: Literal["x0", "fresh"]
    weight: CriticWeightSpec
    merge_student: Optional[float] = None

    def __post_init__(self) -> None:
        _check(self.query.waypoint is None,
               "critic queries cover the whole noise range (query.waypoint applies to student reads only)")
        if self.source == "phase_state":
            _check(self.query.source == "next_state", "critic.source=phase_state reads next-state queries")
        else:
            _check(self.query.source == "x0", f"critic.source={self.source} samples are clean (query.source=x0)")
        _check(self.label == "fresh" or self.weight.kind == "min_snr", "critic.label=x0 uses weight.kind=min_snr")
        _check(self.label == "x0" or self.weight.kind != "min_snr", "min_snr is the x0-label weight")
        _check(not self.weight.importance or self.source == "phase_state",
               "importance weights only exist for phase-state sources (s > 0)")
        _check(self.merge_student is None or 0.0 < self.merge_student < 1.0, "critic.merge_student must lie in (0, 1)")


# ============================================================================ method: generator
@dataclass(frozen=True)
class CdmTermSpec:
    """CDM's off-trajectory term: the student at ``x' = x_k + (s - sigma_k) v_k`` (the rollout's velocity), ``s`` drawn
    on ``[student_lo, student_hi]``; the teacher and the fake read the detached prediction re-noised per ``query``."""

    student_lo: float = 0.02
    student_hi: float = 0.98
    student_bound: Literal["truncate", "clamp"] = "clamp"


#: The normalizer reference each term kind reads: the guided teacher for coupled terms, the conditional teacher for
#: dm, ca and cdm terms.
TERM_REFERENCE = {"coupled": "guided", "dm": "cond", "ca": "cond", "cdm": "cond"}


@dataclass(frozen=True)
class TermSpec:
    """One additive term of the student's x0-space correction, read at its own query.

    ``coupled``: ``T_read - F`` with ``T_read`` the guided teacher (or the precise teacher's walk) -- TDM, eDMD, DMD2.
    ``dm``: ``T_cond - F``. ``ca``: ``guide(T_c, T_u) - T_c`` (``(w - 1)(T_c - T_u)`` for linear CFG) -- D-DMD, CDM.
    ``cdm``: CDM's off-trajectory distribution-matching term (``cdm`` sub-spec).
    """

    kind: Literal["coupled", "dm", "ca", "cdm"]
    query: QuerySpec
    weight: float = 1.0
    cdm: Optional[CdmTermSpec] = None

    def __post_init__(self) -> None:
        _check(math.isfinite(self.weight) and self.weight >= 0, "term.weight must be >= 0")
        _check((self.kind == "cdm") == (self.cdm is not None), "only cdm terms carry a cdm sub-spec")
        if self.kind == "cdm":
            _check(self.query.source == "x0", "the cdm term re-noises its detached off-trajectory prediction")


@dataclass(frozen=True)
class NormalizationSpec:
    """Per-sample normalizer ``D = mean|x0 - reference|`` of each term over the non-batch dims.

    ``reference``: ``auto`` (each term its own: the guided teacher for ``coupled`` terms, the conditional teacher for
    ``dm``, ``ca`` and ``cdm``), or ``guided`` / ``cond``, which must then be every term's. ``eps`` is added and
    ``floor`` clamps (CDM: ``+1e-8``, ``>= 0.1``); ``nan_to_num`` replaces non-finite values (DMD2, CDM).
    """

    reference: Literal["auto", "guided", "cond"] = "auto"
    floor: Optional[float] = None
    eps: float = 0.0
    dtype: Literal["float32", "float64"] = "float32"
    nan_to_num: bool = False

    def __post_init__(self) -> None:
        _check(self.eps >= 0 and (self.floor is None or self.floor > 0), "normalization eps >= 0, floor > 0")


@dataclass(frozen=True)
class LossSpec:
    """Surrogate loss on the anchor: ``scale * mean(form(x0 - sg(x0 + C / D)))`` (placement ``target``) or
    ``scale * mean(form(x0 - sg(x0 + C)) / D)`` (placement ``loss``). For MSE both have the gradient
    ``-2 scale C / (D N)``; they differ only in the logged value. Huber is ``sqrt(e^2 + c^2) - c``."""

    form: Literal["mse", "huber"]
    placement: Literal["loss", "target"]
    scale: float = 1.0
    huber_c: float = 1e-3

    def __post_init__(self) -> None:
        _check(math.isfinite(self.scale) and self.scale > 0, "loss.scale must be positive")
        _check(self.huber_c == 1e-3 or self.form == "huber", "loss.huber_c only applies to form=huber")
        _check(self.huber_c > 0, "loss.huber_c must be positive")


@dataclass(frozen=True)
class GeneratorSpec:
    """The student's objective: one surrogate per term, summed."""

    terms: tuple[TermSpec, ...]
    normalization: NormalizationSpec
    loss: LossSpec

    def __post_init__(self) -> None:
        _check(len(self.terms) >= 1, "generator.terms must not be empty")
        kinds = [term.kind for term in self.terms]
        _check(kinds.count("coupled") <= 1 and kinds.count("cdm") <= 1, "at most one coupled and one cdm term")
        _check(not ("coupled" in kinds and ("dm" in kinds or "ca" in kinds)),
               "a coupled term already contains DM and CA; use dm + ca for decoupled reads")


# ============================================================================ method: engine
@dataclass(frozen=True)
class PreciseWindowSpec:
    """The reads that receive the precise teacher's walk: ``t`` in ``[lo, hi]`` times the last step's input level
    ``sigma_{K-1}`` (the bounds follow any grid)."""

    lo: float
    hi: float

    def __post_init__(self) -> None:
        _unit(self.lo, "precise_teacher.window.lo")
        _unit(self.hi, "precise_teacher.window.hi")
        _check(self.lo < self.hi, "precise_teacher.window must be increasing")


@dataclass(frozen=True)
class PreciseTeacherSpec:
    """The precise teacher: the guided teacher walked ``nodes`` uniform Euler steps from the read level ``t`` to 0. On
    the last step's reads inside ``window`` (default: the last step's whole band, ``[0, 1]``) the walk's answer
    replaces the one-step guided answer; ``weight`` is the share of the walk increment. ``guidance`` is the CFG scale
    the walk carries, on the walked reads alone -- the reads outside the window keep their one-step answer at
    ``engine.cfg`` (null: the walk reads at ``engine.cfg`` too). With ``enabled: false`` the sub-keys are ignored."""

    enabled: bool
    nodes: int = 4
    window: Optional[PreciseWindowSpec] = None
    weight: float = 1.0
    guidance: Optional[float] = None

    def __post_init__(self) -> None:
        if not self.enabled:
            for spec in fields(self):
                if spec.name != "enabled":
                    object.__setattr__(self, spec.name, spec.default)
            return
        _check(isinstance(self.nodes, int) and not isinstance(self.nodes, bool) and self.nodes >= 1,
               f"precise_teacher.nodes must be an integer >= 1, got {self.nodes!r}")
        if self.window is None:
            object.__setattr__(self, "window", PreciseWindowSpec(lo=0.0, hi=1.0))
        _check(math.isfinite(self.weight) and 0.0 < self.weight <= 1.0, "precise_teacher.weight must lie in (0, 1]")
        _check(self.guidance is None or (math.isfinite(self.guidance) and self.guidance >= 1.0),
               "precise_teacher.guidance must be >= 1 (null: follow engine.cfg)")


@dataclass(frozen=True)
class PagSpec:
    """Perturbed-attention guidance on the teacher read: ``T^w <- T^w + scale * (T_c - T_c^deg)``, ``T_c^deg`` the
    conditional teacher with the image-to-image attention of ``blocks`` replaced by the identity (the adapter's
    ``predict_velocity_degraded``). It enters every preference read -- the coupled correction and its guided
    normalizer reference alike, the walk of the precise teacher at every node, a split ``ca`` term -- and no ``dm``
    read. ``band: compose`` adds it only at reads above the last step's input level ``sigma_{K-1}`` (the scales the
    earlier steps compose), ``all`` at every read."""

    scale: float
    blocks: tuple[int, ...]
    band: Literal["all", "compose"] = "all"

    def __post_init__(self) -> None:
        _check(self.band in ("all", "compose"), "engine.pag.band must be all or compose")
        _check(math.isfinite(self.scale) and self.scale > 0, "engine.pag.scale must be positive (null turns PAG off)")
        _check(len(self.blocks) > 0 and len(set(self.blocks)) == len(self.blocks)
               and all(not isinstance(b, bool) and b >= 0 for b in self.blocks),
               "engine.pag.blocks must be distinct non-negative block indices")


@dataclass(frozen=True)
class EngineSpec:
    """How the teacher answers a read: classifier-free guidance at ``cfg`` (``u + w (c - u)``; 1 = the conditional
    teacher alone), the optional precise teacher and the optional PAG increment."""

    cfg: float
    precise_teacher: PreciseTeacherSpec
    pag: Optional[PagSpec] = None

    def __post_init__(self) -> None:
        _check(math.isfinite(self.cfg) and self.cfg >= 1.0, "engine.cfg must be >= 1")

    @property
    def precise_cfg(self) -> float:
        """The CFG scale the precise teacher's walk carries: ``precise_teacher.guidance`` when set, else ``cfg``."""
        guidance = self.precise_teacher.guidance
        return float(self.cfg if guidance is None else guidance)


# ============================================================================ method: updates
@dataclass(frozen=True)
class LrScheduleSpec:
    """``constant``; ``constant_with_warmup`` (linear from 0; DMD2, CDM); ``cosine`` (a half cosine after a linear
    warmup; TDM); ``constant_then_cosine``: ``constant_with_warmup`` up to ``decay_start``, then a half cosine from 1
    to 0 that ends with the role's last update."""

    kind: Literal["constant", "constant_with_warmup", "cosine", "constant_then_cosine"]
    warmup: int = 0
    decay_start: Optional[int] = None

    def __post_init__(self) -> None:
        _check(self.warmup >= 0, "lr schedule warmup must be >= 0")
        _check(self.kind != "constant" or self.warmup == 0, "kind=constant has no warmup")
        if self.kind == "constant_then_cosine":
            _check(isinstance(self.decay_start, int) and not isinstance(self.decay_start, bool)
                   and self.decay_start >= self.warmup, "constant_then_cosine needs an integer decay_start >= warmup")
        else:
            _check(self.decay_start is None, "decay_start only applies to constant_then_cosine")


@dataclass(frozen=True)
class OptimSpec:
    """AdamW settings, the global gradient-norm clip (null = none) and the learning-rate schedule of one role."""

    lr: float
    betas: tuple[float, float]
    weight_decay: float
    eps: float
    clip: Optional[float]
    schedule: LrScheduleSpec

    def __post_init__(self) -> None:
        _check(self.lr > 0 and self.eps > 0 and self.weight_decay >= 0, "optimizer lr, eps > 0 and weight_decay >= 0")
        _check(all(0.0 <= b < 1.0 for b in self.betas), "betas must lie in [0, 1)")
        _check(self.clip is None or self.clip > 0, "clip must be positive or null (no clipping)")


@dataclass(frozen=True)
class EmaConfig:
    """EMA of the student's trainable parameters (exported as ``student_ema`` with every checkpoint)."""

    enabled: bool
    decay: float = 0.999
    update_every: int = 1
    warmup: bool = True
    offset: int = 0            # warm-up counter offset (CDM: 1)

    def __post_init__(self) -> None:
        _check(0.0 < self.decay < 1.0 and self.update_every >= 1, "ema.decay in (0, 1), update_every >= 1")
        _check(self.offset >= 0, "ema.offset must be >= 0")


@dataclass(frozen=True)
class UpdatesSpec:
    """``ratio`` fake updates per student step, each on a fresh prompt batch and rollout. ``critic_first``: the fake
    updates, then the student on the last fake update's rollout (TDM, eDMD, D-DMD, CDM). ``generator_first``: the
    student on the first rollout, then the fake updates (DMD2: the first fake update reuses that rollout's
    pre-update sample)."""

    ratio: int
    order: Literal["critic_first", "generator_first"]
    max_steps: int
    student: OptimSpec
    fake: OptimSpec
    ema: EmaConfig

    def __post_init__(self) -> None:
        _check(self.ratio >= 1 and self.max_steps >= 1, "ratio and max_steps must be >= 1")


# ============================================================================ method
@dataclass(frozen=True)
class MethodSpec:
    """A distillation method: the student's sampler and step selection, the critic, the generator's terms, the teacher
    engine and the update schedule. Nothing in the training step branches on ``name``."""

    name: str
    num_steps: int
    schedule: str
    rollout: RolloutSpec
    step_select: Literal["per_sample", "shared", "anchor"]
    critic: CriticSpec
    generator: GeneratorSpec
    engine: EngineSpec
    updates: UpdatesSpec

    def __post_init__(self) -> None:
        from ..models.base import parse_schedule_family

        parse_schedule_family(self.schedule)
        _check(self.num_steps >= 1, "num_steps must be >= 1")
        _check(self.rollout.to_exit == (self.step_select == "shared"),
               "rollout.to_exit pairs with step_select=shared (DMD2's backward simulation)")
        _check(self.step_select != "anchor" or self.critic.source == "anchor_x0",
               "step_select=anchor pairs with critic.source=anchor_x0")
        _check(self.critic.source != "anchor_x0" or self.step_select == "anchor",
               "critic.source=anchor_x0 needs step_select=anchor")
        _check(self.critic.source != "exit_x0" or self.step_select == "shared",
               "critic.source=exit_x0 needs step_select=shared")
        _check(not self.rollout.to_exit or self.critic.source == "exit_x0",
               f"rollout.to_exit stops the rollout at its exit step, so critic.source must be exit_x0, not "
               f"{self.critic.source} (which reads a step the rollout never took)")
        _check(self.rollout.grid == "fixed" or self.step_select == "anchor", "the dynamic grid is CDM's (step_select=anchor)")
        kinds = {term.kind for term in self.generator.terms}
        _check(self.engine.cfg > 1.0 or "ca" not in kinds, "a CA term with cfg = 1 is identically zero")
        _check(not self.engine.precise_teacher.enabled or "coupled" in kinds,
               f"engine.precise_teacher replaces a coupled term's guided read, but the generator terms are "
               f"{sorted(kinds)}: it would have no effect")
        pag = self.engine.pag
        if pag is not None:
            _check(pag.band == "all" or self.rollout.grid == "fixed", "engine.pag.band=compose needs a fixed rollout grid")
            _check("cdm" not in kinds, "engine.pag is not combined with CDM's off-trajectory term")
            _check(bool(kinds & {"coupled", "ca"}), "engine.pag needs a preference read to act on (a coupled or a ca term)")
        calibrated = [term for term in self.generator.terms if term.query.waypoint == "calibrated"]
        if calibrated:
            _check(kinds == {"coupled"},
                   "query.waypoint=calibrated reads one coupled term (the table measures the coupled guided answer)")
            _check(self.rollout.sampler == "euler" and self.rollout.grid == "fixed" and self.step_select == "per_sample",
                   "query.waypoint=calibrated reads each step of a fixed Euler rollout: step_select=per_sample")
            table = calibrated[0].query.calibration
            _check(len(table.landing) == self.num_steps - 1,
                   f"query.calibration has {len(table.landing)} rows; a {self.num_steps}-step student has "
                   f"{self.num_steps - 1} steps before the last")
            _check(float(table.cfg) == float(self.engine.cfg),
                   f"query.calibration was measured at cfg {table.cfg:g} but engine.cfg is {self.engine.cfg:g}")
        norm = self.generator.normalization
        if norm.reference != "auto":
            natural = {TERM_REFERENCE[kind] for kind in kinds}
            _check(natural == {norm.reference},
                   f"generator.normalization.reference={norm.reference} but the terms {sorted(kinds)} normalize by "
                   f"{sorted(natural)} (coupled: guided; dm / ca / cdm: cond); write auto for mixed terms")


# ============================================================================ model
@dataclass(frozen=True)
class LoraConfig:
    rank: int
    alpha: float
    dropout: float = 0.0
    init: Literal["gaussian", "kaiming"] = "gaussian"
    targets: Optional[tuple[str, ...]] = None     # None = the adapter's default_lora_targets


@dataclass(frozen=True)
class RoleConfig:
    """How a trainable role (student or fake) is parameterized: LoRA on the teacher's weights, or all weights."""

    train: Literal["lora", "full"]
    lora: Optional[LoraConfig] = None
    activation_checkpointing: bool = True
    load_path: str = ""

    def __post_init__(self) -> None:
        _check((self.train == "lora") == (self.lora is not None), "role.lora is required for train=lora and only then")


@dataclass(frozen=True)
class RolesConfig:
    student: RoleConfig
    fake: RoleConfig


@dataclass(frozen=True)
class ModelSpec:
    """``name`` selects the adapter and ``path`` the diffusers checkpoint directory (``$EDMD_MODEL_ROOT`` and ``~``
    are expanded); ``options`` are adapter-specific and validated by the adapter."""

    name: Literal["flux2_klein", "qwen_image_21", "sd3_medium", "toy"]
    variant: str
    path: str
    resolution: int
    roles: RolesConfig
    negative_prompt: str = ""
    text_encoder_shard: bool = False
    reshard_after_forward: bool = True
    options: dict = field(default_factory=dict)
    wandb_project: str = ""

    def __post_init__(self) -> None:
        _check(self.resolution > 0 and self.resolution % 32 == 0, "model.resolution must be a positive multiple of 32")


# ============================================================================ eval
@dataclass(frozen=True)
class EvalSpec:
    """An evaluation: prompts, seeds, metrics and geometry (``configs/eval/``; ``edmd.cli.evaluate``).

    ``prompt_file`` holds the render prompts, ``score_prompt_file`` (default: the same) the prompts the reward models
    score against; image ``i`` uses seed ``seed + i``. ``geometry: aspect`` renders each case at its render file's
    ``wh_ratio`` in the adapter's area tier ``geometry_tier`` (Qwen-Image); ``square`` renders every case at the
    model's resolution. ``metrics`` and ``extra_metrics`` are all scored and reported. ``preview_count`` and
    ``inline_chart`` (a registered W&B chart preset, ``edmd.tracking.media``; empty: no gallery) shape the W&B image
    gallery of the periodic evaluation (``edmd.cli.coordinate``).
    """

    name: str
    prompt_file: str
    score_prompt_file: str = ""
    seed: int = 42
    metrics: tuple[str, ...] = ("hpsv3", "pickscore", "aesthetic", "hpsv2", "hpsv3pp")
    extra_metrics: tuple[str, ...] = ()
    geometry: Literal["square", "aspect"] = "square"
    geometry_tier: str = ""
    preview_count: int = 8
    inline_chart: str = ""

    def __post_init__(self) -> None:
        _check(bool(self.name) and "/" not in self.name and not any(c.isspace() for c in self.name),
               "eval.name must be a short label without spaces or '/'")
        _check((self.geometry == "aspect") == bool(self.geometry_tier),
               "eval.geometry_tier names the area tier of geometry=aspect, and only then")
        _check(self.preview_count >= 0, "eval.preview_count must be >= 0")


# ============================================================================ run
@dataclass(frozen=True)
class WandbSpec:
    """Weights & Biases logging; an empty ``project`` is ``model.wandb_project``, an empty ``entity`` / ``base_url``
    the logged-in account's default. ``mode: online`` without W&B credentials logs offline (``edmd.tracking.wandb``)."""

    enabled: bool = True
    project: str = ""
    entity: str = ""
    base_url: str = ""
    mode: Literal["online", "offline", "disabled"] = "online"
    group: str = ""
    tags: tuple[str, ...] = ()
    notes: str = ""


@dataclass(frozen=True)
class GeometrySpec:
    """Training geometry. ``fixed``: the model's own ``resolution`` (square) at every step.

    ``buckets``: each student step draws one area tier for all ranks, a pure function of (seed, step) with the given
    weights (the first ``largest_first`` steps use the tier with the most tokens, so an out-of-memory shows early). Each
    prompt's aspect ratio comes from its ``wh_ratio`` (jsonl corpus) or, when the corpus carries none, from
    ``aspect_weights`` by a hash of the prompt text; a blank or unknown ratio is 1:1. The adapter maps (tier, aspect) to
    its size table.
    """

    mode: Literal["fixed", "buckets"] = "fixed"
    tiers: dict = field(default_factory=dict)            # {tier: weight}
    aspect_weights: dict = field(default_factory=dict)   # {ratio: weight} for prompts without wh_ratio
    largest_first: int = 2

    def __post_init__(self) -> None:
        if self.mode == "fixed":
            _check(not self.tiers and not self.aspect_weights and self.largest_first == 2,
                   "data.geometry tiers / aspect_weights / largest_first only apply to mode=buckets")
            return
        _check(bool(self.tiers) and all(float(w) >= 0 for w in self.tiers.values())
               and sum(float(w) for w in self.tiers.values()) > 0, "data.geometry.tiers needs positive weights")
        _check(all(float(w) >= 0 for w in self.aspect_weights.values()), "data.geometry.aspect_weights must be >= 0")
        _check(self.largest_first >= 0, "data.geometry.largest_first must be >= 0")


@dataclass(frozen=True)
class DataSpec:
    """``batch_size`` prompts per rank and micro-batch. ``global_batch_size``: prompts per update across all ranks; the
    loop reaches it with ``global_batch_size / (batch_size x world size)`` micro-batches per update (gradient
    accumulation), so a run keeps its global batch on 16 NPUs (one micro-batch) and on 8 GPUs (two). ``None`` = no
    accumulation."""

    path: str
    batch_size: int = 1
    max_samples: int = -1
    num_workers: int = 0
    geometry: GeometrySpec = field(default_factory=GeometrySpec)
    global_batch_size: Optional[int] = None

    def __post_init__(self) -> None:
        _check(self.geometry.mode == "fixed" or self.batch_size == 1,
               "data.geometry buckets give each prompt its own aspect ratio: use a per-rank batch of 1")
        _check(self.batch_size >= 1, "data.batch_size must be >= 1")
        _check(self.global_batch_size is None or (not isinstance(self.global_batch_size, bool)
                                                  and self.global_batch_size >= self.batch_size),
               "data.global_batch_size must be null or at least data.batch_size")


@dataclass(frozen=True)
class RunSpec:
    """Where and how a run executes. ``output_root`` is relative to the repository root unless absolute
    (``$EDMD_OUTPUT_ROOT`` and ``~`` are expanded). ``memory_fraction`` caps the device caching allocator on every
    accelerator; null is the platform default: 0.85 on Ascend NPUs (their allocator otherwise reserves far beyond its
    peak), no cap elsewhere.

    ``eval_every`` and ``eval_ema`` are read only by the periodic-evaluation driver (``edmd.cli.coordinate``): it
    evaluates the checkpoints at 0, ``eval_every``, ... ``max_steps`` with the run's ``eval`` config, and with
    ``eval_ema`` their EMA weights too; 0 turns the evaluation off. ``edmd.cli.train`` ignores both."""

    name: str
    output_root: str
    seed: int
    data: DataSpec
    wandb: WandbSpec
    checkpoint_every: int = 100
    eval_every: int = 0
    log_every: int = 1
    memory_fraction: Optional[float] = None
    eval_ema: bool = False

    def __post_init__(self) -> None:
        _check(self.memory_fraction is None or 0.0 < self.memory_fraction <= 1.0,
               "run.memory_fraction must be in (0, 1] or null")
        _check(bool(self.name) and "/" not in self.name, "run.name must be a directory name")
        _check(self.checkpoint_every >= 1 and self.log_every >= 1, "checkpoint_every and log_every must be >= 1")
        _check(self.eval_every >= 0, "run.eval_every must be >= 0 (0: no periodic evaluation)")


@dataclass(frozen=True)
class Config:
    """A resolved run: model + method + run, and the optional ``eval`` config of the periodic evaluation."""

    model: ModelSpec
    method: MethodSpec
    run: RunSpec
    eval: Optional[EvalSpec] = None

    def __post_init__(self) -> None:
        if self.run.eval_ema:
            _check(self.method.updates.ema.enabled, "run.eval_ema needs method.updates.ema.enabled (EMA weights)")


__all__ = [
    "CalibrationSpec", "CdmTermSpec", "Config", "CriticSpec", "CriticWeightSpec", "DataSpec", "EmaConfig",
    "EngineSpec", "EvalSpec", "GeneratorSpec", "GeometrySpec", "LoraConfig", "LossSpec", "LrScheduleSpec",
    "MethodSpec", "ModelSpec", "NormalizationSpec", "OptimSpec", "PagSpec", "PreciseTeacherSpec",
    "PreciseWindowSpec", "QuerySpec", "RoleConfig", "RolesConfig", "RolloutSpec", "RunSpec", "TERM_REFERENCE",
    "TermSpec", "UpdatesSpec", "WandbSpec",
]

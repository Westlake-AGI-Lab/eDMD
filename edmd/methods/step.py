"""The unified distribution-matching training step, model-agnostic.

One call of :meth:`DistillationStep.student_step` is one student step: ``ratio`` fake updates, each on a fresh prompt
batch and rollout, and one student update -- first (``generator_first``: DMD2) or last (``critic_first``: TDM, eDMD,
D-DMD, CDM). Every difference between methods is a component value of the ``MethodSpec``; nothing here branches on the
method's name.

Gradient paths:

1. the only student forward with a graph is the recomputation of the selected step from its detached rollout state
   (plus CDM's off-trajectory forward); critic rollouts stay under no_grad;
2. the target, the normalizer and every teacher / fake read are detached, and so are query points before they reach
   the teacher or the fake;
3. during the student update neither the fake nor the teacher accumulates gradients, and during the fake update the
   student has no graph; each update ends with ``zero_grad`` on its optimizer and the step asserts that the other role
   holds no gradient.

Gradient accumulation (``accumulate`` = A, from ``run.data.global_batch_size``): every update runs on A micro-batches,
each with its own prompts, noise and rollout, backpropagates each micro-batch's loss scaled by 1/A, then clips and
steps once. This equals A times as many ranks without accumulation (FSDP2 averages the gradients over ranks, the 1/A
over micro-batches). Draws a reference shares across ranks (DMD2's exit step, CDM's step count) are shared across the
micro-batches of an update too; per-sample and per-rank draws are drawn per micro-batch (named ``<name>.m<j>``). With
A = 1 every call and draw name is the unaccumulated step's.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol

import torch
from torch import Tensor

from ..algo import engine, fm, queries, rollout
from ..algo.critic import critic_loss
from ..algo.draws import Draws
from ..algo.generator import TermReading, coupled_correction, generator_loss

VelocityCall = Callable[[Tensor, Tensor, Any], Tensor]


@dataclass
class Batch:
    """An encoded prompt batch and its initial noise; ``schedule`` when its geometry has its own grid (multi-resolution:
    the shift depends on the token count), else the step's."""

    cond: Any
    uncond: Any
    noise: Tensor
    prompts: list[str] = field(default_factory=list)
    geometry: Any = None
    schedule: Optional["Schedule"] = None

    @property
    def size(self) -> int:
        return int(self.noise.shape[0])


class BatchSource(Protocol):
    def next(self) -> Batch: ...


@dataclass
class Roles:
    """Model access. ``student``/``fake``/``teacher`` return float32 velocities; ``guide`` is the pipeline's CFG.

    ``teacher_pair(x, t, cond, uncond) -> (v_cond, v_uncond)`` serves the guided teacher reads when set (the adapter's
    ``predict_velocity_pair``: SD3 batches CFG like its pipeline); None = two ``teacher`` calls.
    ``teacher_degraded(x, t, cond)`` is PAG's weak read; required when ``engine.pag`` is set, unused otherwise.
    """

    student: VelocityCall
    fake: VelocityCall
    teacher: VelocityCall
    guide: Callable[[Tensor, Tensor, float], Tensor]
    student_module: torch.nn.Module
    fake_module: torch.nn.Module
    teacher_pair: Optional[Callable[[Tensor, Tensor, Any, Any], tuple[Tensor, Tensor]]] = None
    teacher_degraded: Optional[VelocityCall] = None


@dataclass
class Updaters:
    """Optimizers, LR schedulers and the (global) clipping function."""

    student: torch.optim.Optimizer
    fake: torch.optim.Optimizer
    student_params: list
    fake_params: list
    clip: Callable[[list, Optional[float]], Tensor]
    student_scheduler: Any = None
    fake_scheduler: Any = None
    ema: Any = None
    merge: Optional[Callable[[float], None]] = None     # CDM: fake <- student merge


@dataclass
class Schedule:
    """The student's fixed grid (sigma), its raw grid and warp, and CDM's n-step warp."""

    grid: Tensor
    raw_grid: Tensor
    warp: Callable[[Tensor], Tensor]
    warp_steps: Callable[[Tensor, int], Tensor]


def _gather(points: list[Tensor], index: Tensor) -> Tensor:
    stacked = torch.stack(points, dim=1)
    rows = torch.arange(stacked.shape[0], device=stacked.device)
    return stacked[rows, index.to(stacked.device)]


class DistillationStep:
    def __init__(self, method: Any, roles: Roles, updaters: Updaters, batches: BatchSource, draws: Draws,
                 schedule: Schedule, *, shared_randint: Optional[Callable[[str, int], int]] = None,
                 assert_grads: bool = True, accumulate: int = 1) -> None:
        if isinstance(accumulate, bool) or not isinstance(accumulate, int) or accumulate < 1:
            raise ValueError(f"accumulate must be a positive integer, got {accumulate!r}")
        self.m = method
        self.accumulate = accumulate            # micro-batches per update (module docstring)
        self._shared: dict[str, int] = {}       # this student step's shared draws, by name
        self.roles = roles
        self.up = updaters
        self.batches = batches
        self.draws = draws
        self.schedule = schedule
        self.shared_randint = shared_randint or (lambda name, high: int(draws.randint(name, high, (1,), device=torch.device("cpu"))[0]))
        self.assert_grads = assert_grads
        self.step_index = 0
        kinds = {term.kind for term in method.generator.terms}
        # the precise teacher's walk replaces the coupled term's one-step guided read inside its window
        self.walk_in_coupled = method.engine.precise_teacher.enabled and "coupled" in kinds
        # the unconditional read is needed by any guided read: the one-step coupled / ca reads at engine.cfg, and the
        # walked read at engine.precise_cfg (precise_teacher.guidance when set, else engine.cfg)
        self.needs_uncond = ((method.engine.cfg != 1.0 and bool(kinds & {"coupled", "ca"}))
                             or (self.walk_in_coupled and method.engine.precise_cfg != 1.0))
        # an Euler rollout keeps its one-step x0 predictions and velocities only for the components that read them:
        # the anchor critic sample (CDM, D-DMD) and CDM's off-trajectory term
        self.record_rollout = method.critic.source == "anchor_x0" or "cdm" in kinds
        self.pag = method.engine.pag
        if self.pag is not None and roles.teacher_degraded is None:
            raise ValueError("engine.pag needs Roles.teacher_degraded (the adapter's predict_velocity_degraded)")
        self._laws: dict[tuple, Any] = {}       # query.waypoint: calibrated, the read laws per rollout grid

    def _schedule(self, batch: Batch) -> Schedule:
        return batch.schedule if batch.schedule is not None else self.schedule

    def _query_coordinates(self, batch: Batch, roll: rollout.Rollout):
        """The raw grid and warp of a query: this rollout's sampled grid and n-step warp when the grid is dynamic."""
        s = self._schedule(batch)
        if roll.raw_grid is not None:
            return roll.raw_grid, lambda raw: s.warp_steps(raw, roll.num_steps)
        if self.m.rollout.grid == "dynamic" and (self.m.critic.query.space == "raw" or
                any(term.query.space == "raw" for term in self.m.generator.terms)):
            raise ValueError("dynamic raw queries need this rollout's sampled raw grid")
        return s.raw_grid, s.warp

    # ------------------------------------------------------------------ names and shared draws
    def _name(self, base: str, micro: int) -> str:
        """``base`` for micro-batch ``micro`` of an update: unchanged without accumulation, else ``base.m<micro>``."""
        return base if self.accumulate == 1 else f"{base}.m{micro}"

    def _shared_randint(self, name: str, high: int) -> int:
        """A draw every rank shares (broadcast from rank 0), made once per name and student step: the micro-batches of
        one update then share it too, as the ranks they stand in for would."""
        if name not in self._shared:
            self._shared[name] = self.shared_randint(name, high)
        return self._shared[name]

    # ------------------------------------------------------------------ rollouts
    def rollout(self, batch: Batch, tag: str, *, shared_tag: Optional[str] = None) -> rollout.Rollout:
        """The student's rollout of ``batch``; its shared draws are named by ``shared_tag`` (default ``tag``)."""
        m, s = self.m, self._schedule(batch)
        shared = shared_tag or tag
        student = lambda x, t: self.roles.student(x, t, batch.cond)  # noqa: E731
        if m.rollout.sampler == "x0_renoise":
            exit_step = self._shared_randint(f"{shared}.exit", m.num_steps)
            return rollout.backward_simulation(student, batch.noise, s.grid, exit_step, draws=self.draws, prefix=tag)
        if m.rollout.grid == "dynamic":
            n = self._shared_randint(f"{shared}.n", m.rollout.dynamic_max_steps) + 1
            lo, hi = m.rollout.dynamic_raw_range
            interior = self.draws.rand(f"{tag}.interior", (n - 1,), device=torch.device("cpu")) * (hi - lo) + lo
            raw_grid = rollout.dynamic_raw_grid(n, interior).to(batch.noise.device)
            grid = rollout.dynamic_grid(n, interior, s.warp_steps).to(batch.noise.device)
        else:
            n, grid = m.num_steps, s.grid
            raw_grid = None
        roll = rollout.euler_rollout(student, batch.noise, grid, record=self.record_rollout,
                                     strict_grid=m.rollout.grid != "dynamic")
        roll.raw_grid = raw_grid
        if m.step_select == "anchor":
            # CDM: one grid point per sample, not shared across ranks
            roll.anchor = self.draws.randint(f"{tag}.anchor", n, (batch.size,), device=batch.noise.device)
        return roll

    # ------------------------------------------------------------------ student one-step prediction
    def _predict(self, roll: rollout.Rollout, index: Tensor, batch: Batch, *, grad: bool) -> dict[str, Tensor]:
        x_left = _gather(roll.states, index).detach()
        t_left = roll.grid.to(x_left.device)[index]
        context = torch.enable_grad() if grad else torch.no_grad()
        with context:
            v = self.roles.student(x_left, t_left, batch.cond)
            x0 = x_left - t_left.reshape(-1, *([1] * (x_left.ndim - 1))) * v
            eps = x0 + v
        return {"x_left": x_left, "t_left": t_left, "v": v, "x0": x0, "eps": eps}

    # ------------------------------------------------------------------ critic
    def critic_sample(self, batch: Batch, roll: rollout.Rollout, tag: str) -> dict[str, Any]:
        """The clean (or next-state) sample one fake update trains on, per ``critic.source``."""
        source = self.m.critic.source
        device = batch.noise.device
        if source == "phase_state":
            index = queries.select_steps("per_sample", batch.size, roll.num_steps, self.draws, device=device,
                                         name=f"{tag}.k")
            pred = self._predict(roll, index, batch, grad=False)
            x0, eps = pred["x0"], pred["eps"]
        elif source == "final":
            index = queries.select_steps("final", batch.size, roll.num_steps, self.draws, device=device,
                                         name=f"{tag}.k")
            pred = self._predict(roll, index, batch, grad=False)
            x0, eps = pred["x0"], pred["eps"]
        elif source == "exit_x0":
            index = queries.select_steps("shared", batch.size, roll.num_steps, self.draws, device=device,
                                         name=f"{tag}.k", exit_step=roll.exit_step)
            pred = self._predict(roll, index, batch, grad=False)
            x0, eps = pred["x0"], None
        elif source == "anchor_x0":
            index = roll.anchor.to(device)
            x0, eps = _gather(roll.x0s, index).detach(), None
        else:
            raise ValueError(source)
        sample = {"index": index, "x0": x0.detach(), "eps": None if eps is None else eps.detach(), "grid": roll.grid}
        if roll.raw_grid is not None:
            sample["raw_grid"] = roll.raw_grid
        return sample

    def critic_update(self, batch: Batch, sample: dict[str, Any], tag: str) -> dict[str, Tensor]:
        """One fake update on one batch (the unaccumulated update; :meth:`critic_group` is the general one)."""
        self.up.fake.zero_grad(set_to_none=True)
        metrics = self._critic_backward(batch, sample, tag, scale=1.0)
        metrics["optim/fake_grad_norm"] = self._critic_step()
        return metrics

    def critic_group(self, micro: list[tuple[Batch, rollout.Rollout]], samples: list[dict[str, Any]],
                     base: str) -> dict[str, Tensor]:
        """One fake update over the micro-batches of an update (one micro-batch: exactly :meth:`critic_update`)."""
        if len(micro) == 1:
            return self.critic_update(micro[0][0], samples[0], base)
        self.up.fake.zero_grad(set_to_none=True)
        parts = [self._critic_backward(batch, sample, self._name(base, j), scale=1.0 / len(micro))
                 for j, ((batch, _), sample) in enumerate(zip(micro, samples))]
        metrics = _mean_metrics(parts)
        metrics["optim/fake_grad_norm"] = self._critic_step()
        return metrics

    def _fake(self, x: Tensor, t: Tensor, batch: Batch) -> Tensor:
        return self.roles.fake(x, t, batch.cond)

    def _critic_backward(self, batch: Batch, sample: dict[str, Any], tag: str, *, scale: float) -> dict[str, Tensor]:
        """Query, fake read, critic loss and its backward (scaled by ``scale``); gradients accumulate on the fake."""
        spec = self.m.critic
        raw_grid = sample.get("raw_grid")
        if self.m.rollout.grid == "dynamic" and spec.query.space == "raw" and raw_grid is None:
            raise ValueError("dynamic raw critic query needs this rollout's sampled raw grid")
        s = self._schedule(batch)
        q = queries.build_query(spec.query, x0=sample["x0"], eps=sample["eps"], index=sample["index"],
                                sigma_grid=sample["grid"], raw_grid=raw_grid if raw_grid is not None else s.raw_grid,
                                warp=(lambda raw: s.warp_steps(raw, len(sample["grid"]) - 1)) if raw_grid is not None else s.warp,
                                draws=self.draws, name=f"{tag}.q", compute_importance=spec.weight.importance)
        fake_v = self._fake(q.x_t, q.t, batch)
        loss, metrics = critic_loss(spec, fake_v=fake_v, x_t=q.x_t, t=q.t, x_base=q.x_base, t_base=q.t_base, z=q.z,
                                    x0_label=q.x0, importance=q.importance)
        (loss if scale == 1.0 else loss * scale).backward()
        metrics = dict(metrics)
        metrics.update({"loss/critic": loss.detach(), "critic/sigma_mean": q.t.mean(),
                        "critic/sigma_base_mean": q.t_base.mean()})
        return metrics

    def _critic_step(self) -> Tensor:
        """Clip, step and schedule the fake on the gradients accumulated since the update began; merge (CDM)."""
        spec = self.m.critic
        if self.assert_grads:
            _assert_no_grad(self.up.student_params, "student", "the fake update")
        grad_norm = self.up.clip(self.up.fake_params, self.m.updates.fake.clip)
        self.up.fake.step()
        if self.up.fake_scheduler is not None:
            self.up.fake_scheduler.step()
        self.up.fake.zero_grad(set_to_none=True)
        if spec.merge_student is not None:
            if self.up.merge is None:
                raise ValueError("critic.merge_student needs Updaters.merge")
            self.up.merge(float(spec.merge_student))
        return grad_norm

    # ------------------------------------------------------------------ teacher reads
    def _teacher_pair(self, x: Tensor, t: Tensor, batch: Batch) -> tuple[Tensor, Tensor]:
        if self.roles.teacher_pair is not None:
            return self.roles.teacher_pair(x, t, batch.cond, batch.uncond)
        return self.roles.teacher(x, t, batch.cond), self.roles.teacher(x, t, batch.uncond)

    def _teacher_reads(self, x_t: Tensor, t: Tensor, batch: Batch, *, uncond: bool) -> tuple[Tensor, Optional[Tensor]]:
        with torch.no_grad():
            if uncond:
                return self._teacher_pair(x_t, t, batch)
            return self.roles.teacher(x_t, t, batch.cond), None

    def _guided_fn(self, batch: Batch, first: dict[str, Tensor], roll: Optional[rollout.Rollout] = None,
                   scale: Optional[Tensor] = None) -> Callable[[Tensor, Tensor], Tensor]:
        """The guided teacher's velocity field (CFG, plus PAG when set) -- what the precise teacher walks. ``scale``
        carries the CFG scale per sample for a walk guided at its own scale (``precise_teacher.guidance``): walked reads
        take that scale and the rest stay at ``engine.cfg``, since the walk's first node is also the one-step answer of
        the reads outside the window. Without it every sample reads at ``engine.cfg``. The first call stores the
        conditional and unconditional reads in ``first``."""
        w = float(self.m.engine.cfg)
        use_pair = scale is not None or w != 1.0

        def guided(x: Tensor, t: Tensor) -> Tensor:
            if use_pair:
                v_c, v_u = self._teacher_pair(x, t, batch)
            else:
                v_c = v_u = self.roles.teacher(x, t, batch.cond)
            if not first:
                first.update(cond=v_c, uncond=v_u)
            if scale is not None:   # roles.guide per sample (the pipelines' CFG is linear in the scale)
                shaped = scale.to(device=v_c.device, dtype=v_c.dtype).reshape(-1, *([1] * (v_c.ndim - 1)))
                # as ``guide`` does: a read at scale 1 is the conditional velocity itself, not a rounded sum
                v = torch.where(shaped == 1, v_c, v_u + shaped * (v_c - v_u))
            else:
                v = self.roles.guide(v_c, v_u, w) if w != 1.0 else v_c
            if self.pag is not None:
                # the walk integrates the PAG-tilted field; in x0 space this is the coupled read's increment
                v = v + self._pag_increment(v_c, self.roles.teacher_degraded(x, t, batch.cond), t, roll)
            return v

        return guided

    def _pag_increment(self, strong: Tensor, weak: Tensor, t: Tensor, roll: Optional[rollout.Rollout]) -> Tensor:
        """``scale * (strong - weak)`` (velocity or x0 space alike: ``x0_from_v`` is affine in v at fixed x, t), zero
        on reads at or below the last step's starting level under ``band: compose``."""
        increment = float(self.pag.scale) * (strong - weak)
        if self.pag.band == "compose":   # only above the last step's starting level (every rank reads)
            if roll is None:
                raise ValueError("engine.pag.band=compose needs the rollout grid")
            start = roll.grid.to(device=t.device, dtype=t.dtype)[-2]
            increment = increment * (t > start).to(increment.dtype).reshape(-1, *([1] * (increment.ndim - 1)))
        return increment

    def _calibrated_laws(self, query, grid: Tensor):
        """``query.waypoint: calibrated``: the read law of every step before the last on this rollout's grid (cached by
        grid; a multi-resolution run has one grid per geometry)."""
        key = (tuple(float(s) for s in grid.detach().to("cpu").reshape(-1).tolist()), float(query.lo),
               float(query.margin), id(query.calibration))
        if key not in self._laws:
            self._laws[key] = queries.calibrated_laws(query, grid)
        return self._laws[key]

    # ------------------------------------------------------------------ generator
    def _read_term(self, term, index: int, pred: dict[str, Tensor], k: Tensor, batch: Batch,
                   roll: rollout.Rollout, tag: str) -> TermReading:
        """One generator term's detached correction and normalizer reference at its query."""
        w = float(self.m.engine.cfg)
        name = f"{tag}.term{index}"
        anchor = pred["x0"]
        if term.kind == "cdm":
            return self._read_cdm(term, pred, k, batch, roll, name)
        raw_grid, warp = self._query_coordinates(batch, roll)
        laws = self._calibrated_laws(term.query, roll.grid) if term.query.waypoint == "calibrated" else None
        q = queries.build_query(term.query, x0=pred["x0"], eps=pred["eps"], index=k, sigma_grid=roll.grid,
                                raw_grid=raw_grid, warp=warp, draws=self.draws, name=name, laws=laws)
        x_t, t = q.x_t, q.t
        with torch.no_grad():
            if term.kind == "coupled" and self.walk_in_coupled:
                return self._read_walked(term, anchor, q, k, batch, roll)
            if term.kind == "coupled":
                v_c, v_u = self._teacher_reads(x_t, t, batch, uncond=self.needs_uncond and w != 1.0)
                fake_x0 = fm.x0_from_v(x_t, self._fake(x_t, t, batch), t)
                correction, guided = coupled_correction(
                    fm.x0_from_v(x_t, v_c, t), fake_x0, None if v_u is None else fm.x0_from_v(x_t, v_u, t), w)
                if self.pag is not None:
                    # PAG: T^w <- T^w + scale (T_c - T_c^deg), in the correction and its normalizer reference alike
                    degraded_v = self.roles.teacher_degraded(x_t, t, batch.cond)
                    increment = self._pag_increment(fm.x0_from_v(x_t, v_c, t), fm.x0_from_v(x_t, degraded_v, t), t, roll)
                    correction = (correction + increment).detach()
                    guided = (guided + increment).detach()
                return TermReading("coupled", anchor, correction, guided, term.weight, t, q.t_base)
            if term.kind == "dm":
                v_c, _ = self._teacher_reads(x_t, t, batch, uncond=False)
                fake_v = self._fake(x_t, t, batch)
                cond_x0 = fm.x0_from_v(x_t, v_c, t)
                return TermReading("dm", anchor, (cond_x0 - fm.x0_from_v(x_t, fake_v, t)).detach(),
                                   cond_x0.detach(), term.weight, t, q.t_base)
            if term.kind == "ca":
                v_c, v_u = self._teacher_reads(x_t, t, batch, uncond=True)
                cond_x0 = fm.x0_from_v(x_t, v_c, t)
                guided_x0 = fm.x0_from_v(x_t, self.roles.guide(v_c, v_u, w), t)
                correction = guided_x0 - cond_x0
                if self.pag is not None:
                    # the second strong-weak preference: scale (T_c - T_c^deg) next to CFG's; the reference stays T_c
                    degraded_x0 = fm.x0_from_v(x_t, self.roles.teacher_degraded(x_t, t, batch.cond), t)
                    correction = correction + self._pag_increment(cond_x0, degraded_x0, t, roll)
                return TermReading("ca", anchor, correction.detach(), cond_x0.detach(), term.weight, t, q.t_base)
        raise ValueError(term.kind)

    def _read_walked(self, term, anchor: Tensor, q: queries.Query, k: Tensor, batch: Batch,
                     roll: rollout.Rollout) -> TermReading:
        """The coupled read with the precise teacher: inside the window (the last step's reads) the guided teacher's
        walk from the query to 0 replaces its one-step answer with share ``weight``."""
        pt = self.m.engine.precise_teacher
        x_t, t = q.x_t, q.t
        first: dict[str, Tensor] = {}
        mask = engine.window_mask(t, k, roll.num_steps, roll.grid, pt)
        grid = engine.walk_grid(t, pt.nodes)
        scale = None
        if self.m.engine.precise_cfg != float(self.m.engine.cfg):   # the walk carries its own CFG scale
            scale = torch.where(mask, torch.full_like(mask, self.m.engine.precise_cfg, dtype=torch.float32),
                                torch.full_like(mask, float(self.m.engine.cfg), dtype=torch.float32))
        v_bar, v_first, _ = engine.walk(self._guided_fn(batch, first, roll, scale), x_t, grid)
        share = engine.walk_share(t, mask, pt)
        teacher_x0 = fm.x0_from_v(x_t, engine.blend(v_bar, v_first, share), t)
        fake_x0 = fm.x0_from_v(x_t, self._fake(x_t, t, batch), t)
        correction = (teacher_x0 - fake_x0).detach()
        return TermReading("coupled", anchor, correction, teacher_x0.detach(), term.weight, t, q.t_base)

    def _read_cdm(self, term, pred: dict[str, Tensor], k: Tensor, batch: Batch, roll: rollout.Rollout,
                  name: str) -> TermReading:
        """CDM's off-trajectory term."""
        spec = term.cdm
        batch_size = batch.size
        u = self.draws.rand(f"{name}.student_t", (batch_size,), device=batch.noise.device, dtype=fm.wide(roll.grid).dtype)
        if spec.student_bound == "clamp":
            s = u.clamp(spec.student_lo, spec.student_hi)
        else:
            s = spec.student_lo + u * (spec.student_hi - spec.student_lo)
        x_k = pred["x_left"]
        sigma_k = pred["t_left"]
        if not roll.velocities:
            raise ValueError("the cdm term needs the rollout's recorded velocities")
        v_k = _gather(roll.velocities, k).detach()
        x_prime = (x_k + fm.bt(s - sigma_k, x_k).to(x_k.dtype) * v_k).detach()
        with torch.enable_grad():
            v_prime = self.roles.student(x_prime, s, batch.cond)
            anchor = x_prime - fm.bt(s, x_prime).to(x_prime.dtype) * v_prime
        raw_grid, warp = self._query_coordinates(batch, roll)
        q = queries.build_query(term.query, x0=anchor.detach(), eps=None, index=k, sigma_grid=roll.grid,
                                raw_grid=raw_grid, warp=warp, draws=self.draws, name=f"{name}.teacher")
        with torch.no_grad():
            v_c = self.roles.teacher(q.x_t, q.t, batch.cond)
            fake_v = self._fake(q.x_t, q.t, batch)
            cond_x0 = fm.x0_from_v(q.x_t, v_c, q.t)
            correction = cond_x0 - fm.x0_from_v(q.x_t, fake_v, q.t)
        return TermReading("cdm", anchor, correction.detach(), cond_x0.detach(), term.weight, q.t, q.t_base)

    def generator_update(self, batch: Batch, roll: rollout.Rollout, tag: str) -> tuple[dict[str, Tensor], dict[str, Any]]:
        """One student update on one batch (the unaccumulated update; :meth:`generator_group` is the general one)."""
        self.up.student.zero_grad(set_to_none=True)
        metrics, sample = self._generator_backward(batch, roll, tag, scale=1.0)
        metrics["optim/student_grad_norm"] = self._generator_step()
        return metrics, sample

    def generator_group(self, micro: list[tuple[Batch, rollout.Rollout]], base: str
                        ) -> tuple[dict[str, Tensor], list[dict[str, Any]]]:
        """One student update over the micro-batches of an update (one micro-batch: exactly :meth:`generator_update`);
        returns the metrics and each micro-batch's generated sample."""
        if len(micro) == 1:
            metrics, sample = self.generator_update(micro[0][0], micro[0][1], base)
            return metrics, [sample]
        self.up.student.zero_grad(set_to_none=True)
        parts, samples = [], []
        for j, (batch, roll) in enumerate(micro):
            metrics, sample = self._generator_backward(batch, roll, f"{base}.m{j}", scale=1.0 / len(micro))
            parts.append(metrics)
            samples.append(sample)
        metrics = _mean_metrics(parts)
        metrics["optim/student_grad_norm"] = self._generator_step()
        return metrics, samples

    def _generator_backward(self, batch: Batch, roll: rollout.Rollout, tag: str, *,
                            scale: float) -> tuple[dict[str, Tensor], dict[str, Any]]:
        """Step selection, student prediction, detached term reads and the surrogate's backward (scaled by ``scale``);
        gradients accumulate on the student."""
        m = self.m
        device = batch.noise.device
        k = queries.select_steps(m.step_select, batch.size, roll.num_steps, self.draws, device=device,
                                 name=f"{tag}.k", anchor=roll.anchor, exit_step=roll.exit_step)
        pred = self._predict(roll, k, batch, grad=True)
        readings = [self._read_term(term, i, pred, k, batch, roll, tag) for i, term in enumerate(m.generator.terms)]
        loss, metrics = generator_loss(readings, m.generator.normalization, m.generator.loss)
        (loss if scale == 1.0 else loss * scale).backward()
        metrics = dict(metrics)
        metrics["generator/step_mean"] = k.float().mean()
        if m.generator.terms[0].query.waypoint == "calibrated":
            # the share of reads from the steps before the last, the mass of their offset above the landing point
            # (sum (t - s) / B; their mean offset = mass / share) and the share within 0.05 of it
            t_read = readings[0].t.detach()
            steps = k.to(t_read.device)
            landing = roll.grid.to(device=t_read.device, dtype=t_read.dtype)[1:][steps]
            inner = steps < roll.num_steps - 1
            offset = torch.where(inner, t_read - landing, torch.zeros_like(t_read))
            metrics["generator/calibrated/inner_share"] = inner.float().mean().to(device)
            metrics["generator/calibrated/offset_mass"] = offset.float().mean().to(device)
            metrics["generator/calibrated/near_share"] = (inner & (offset < 0.05)).float().mean().to(device)
        sample = {"index": k, "x0": pred["x0"].detach(),
                  "eps": None if pred["eps"] is None else pred["eps"].detach(), "grid": roll.grid,
                  "x_left": pred["x_left"], "t_left": pred["t_left"]}
        return metrics, sample

    def _generator_step(self) -> Tensor:
        """Clip, step and schedule the student on the gradients accumulated since the update began; EMA."""
        if self.assert_grads:
            _assert_no_grad(self.up.fake_params, "fake", "the student update")
        grad_norm = self.up.clip(self.up.student_params, self.m.updates.student.clip)
        self.up.student.step()
        if self.up.student_scheduler is not None:
            self.up.student_scheduler.step()
        self.up.student.zero_grad(set_to_none=True)
        if self.up.ema is not None:
            self.up.ema.after_optimizer_step()
        return grad_norm

    # ------------------------------------------------------------------ one student step
    def _micro_rollouts(self, tag: str) -> list[tuple[Batch, rollout.Rollout]]:
        """The micro-batches of one update: a fresh prompt batch and its rollout each (shared draws: once per update)."""
        micro = []
        for j in range(self.accumulate):
            batch = self.batches.next()
            micro.append((batch, self.rollout(batch, self._name(f"{tag}.roll", j), shared_tag=f"{tag}.roll")))
        return micro

    def student_step(self) -> dict[str, float]:
        """``ratio`` fake updates and one student update; returns the step's metrics (fake ones averaged)."""
        m = self.m
        ratio = m.updates.ratio
        totals: dict[str, float] = {}
        self._shared = {}

        def accumulate(values: dict[str, Tensor], scale: float = 1.0) -> None:
            for key, value in values.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach().float().mean()) * scale

        last: Optional[list[tuple[Batch, rollout.Rollout]]] = None
        for r in range(ratio):
            tag = f"s{self.step_index}.r{r}"
            micro = self._micro_rollouts(tag)
            if m.updates.order == "generator_first" and r == 0:
                # Only an exit_x0 critic shares the generator's selected prediction (DMD2); other sources select their
                # own sample while the student is still the pre-update one.
                samples = None if m.critic.source == "exit_x0" else [
                    self.critic_sample(batch, roll, self._name(f"{tag}.critic", j))
                    for j, (batch, roll) in enumerate(micro)]
                gen_metrics, gen_samples = self.generator_group(micro, f"{tag}.gen")
                accumulate(gen_metrics)
                if samples is None:
                    samples = [{"index": gen_sample["index"], "x0": gen_sample["x0"], "eps": None, "grid": roll.grid}
                               for (_, roll), gen_sample in zip(micro, gen_samples)]
            else:
                samples = [self.critic_sample(batch, roll, self._name(f"{tag}.critic", j))
                           for j, (batch, roll) in enumerate(micro)]
            accumulate(self.critic_group(micro, samples, f"{tag}.critic"), 1.0 / ratio)
            last = micro
        if m.updates.order == "critic_first":
            assert last is not None
            gen_metrics, _ = self.generator_group(last, f"s{self.step_index}.gen")
            accumulate(gen_metrics)
        self.step_index += 1
        return totals


def _mean_metrics(parts: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """The mean of each metric over the micro-batches of an update (each micro-batch's value is itself a batch mean)."""
    keys = list(parts[0])
    return {key: torch.stack([part[key].detach().float().mean() for part in parts]).mean() for key in keys}


def _assert_no_grad(params: list, role: str, phase: str) -> None:
    for parameter in params:
        if parameter.grad is not None and bool(torch.count_nonzero(parameter.grad.detach())):
            raise RuntimeError(f"{role} parameters accumulated gradients during {phase}")


__all__ = ["Batch", "BatchSource", "DistillationStep", "Roles", "Schedule", "Updaters"]

"""The training loop: run file -> adapter -> sharded roles -> ``DistillationStep`` -> checkpoints and scalars.

The algorithm lives in ``edmd.methods.step``; this module wires the runtime around it: FSDP2 roles, AdamW and an LR
schedule per role (the fake schedule steps once per fake update, horizon ``max_steps * ratio``), the resumable prompt
loader, checkpoints every ``run.checkpoint_every`` student steps (with ``weights/inference.json``, the sampler, grid
and roles sampling and evaluation use) and W&B training scalars from rank 0.

``--resume`` restores everything, including the RNG streams and the sampler position; ``--stop-after N`` ends the
run at step N with a checkpoint (``edmd.cli.coordinate`` evaluates between such segments).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import socket
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from ..algo import fm
from ..algo.draws import TorchDraws
from ..config.core import dump_yaml, expand_path
from ..config.load import compose, parse_assignments, run_output_dir
from ..methods.step import Batch, DistillationStep, Roles, Schedule, Updaters
from ..models.base import ModelAdapter, build_adapter
from ..runtime import checkpoint as ckpt
from ..runtime import device as device_lib
from ..runtime.distributed import (barrier, broadcast_object, broadcast_tensor, clip_grad_norm, init_distributed,
                                   is_main_process, reduce_max, reduce_mean, trainable_parameters)
from ..runtime.ema import EmaSpec, ShadowEma, merge_parameters
from ..runtime.lora import LoraSpec
from ..runtime.roles import RoleSpec, build_role
from .data import DistributedKRepeatSampler, PromptFileDataset


def _log(message: str) -> None:
    if is_main_process():
        print(message, flush=True)


def _set_seed(seed: int, rank: int, device: torch.device) -> None:
    value = int(seed) + int(rank)
    random.seed(value)
    np.random.seed(value % (2 ** 32))
    torch.manual_seed(value)
    device_lib.manual_seed_all(value, device)   # the NPU's generators (torch.manual_seed already covers CUDA)


#: ``run.memory_fraction: null`` on an Ascend NPU: headroom for HCCL, which allocates outside the caching allocator
NPU_MEMORY_FRACTION = 0.85


def memory_cap(fraction: Optional[float], device: torch.device) -> Optional[float]:
    """The allocator cap: ``run.memory_fraction``, or when it is null the platform default (Ascend only)."""
    if fraction is not None:
        return float(fraction)
    return NPU_MEMORY_FRACTION if device.type == "npu" else None


def micro_batches(data: Any, world_size: int) -> int:
    """Micro-batches per update: ``data.global_batch_size / (data.batch_size x world_size)``, 1 when it is unset."""
    if data.global_batch_size is None:
        return 1
    per_pass = int(data.batch_size) * int(world_size)
    if int(data.global_batch_size) % per_pass:
        raise ValueError(f"run.data.global_batch_size={data.global_batch_size} is not a multiple of batch_size x world "
                         f"size = {data.batch_size} x {world_size}; pick a global batch the cards divide")
    return int(data.global_batch_size) // per_pass


def _load_path(section: Any) -> str:
    """``model.roles.<role>.load_path`` through ``expand_path`` (a relative path is the repository's on every rank)."""
    return str(expand_path(section.load_path)) if section.load_path else ""


def role_spec(adapter: Any, config: Any, role: str) -> RoleSpec:
    section = getattr(config.model.roles, role)
    if section.train == "lora":
        lora = section.lora
        targets = tuple(lora.targets) if lora.targets else tuple(adapter.default_lora_targets)
        return RoleSpec(train="lora", lora=LoraSpec(rank=lora.rank, alpha=float(lora.alpha), dropout=float(lora.dropout),
                                                    init=lora.init, target_modules=targets),
                        activation_checkpointing=section.activation_checkpointing, load_path=_load_path(section))
    return RoleSpec(train="full", activation_checkpointing=section.activation_checkpointing,
                    load_path=_load_path(section))


def constant_then_cosine_factor(step: int, warmup: int, decay_start: int, end: int) -> float:
    """Diffusers' ``constant_with_warmup`` factor before ``decay_start``, then a half cosine from 1 to 0 over
    ``[decay_start, end]`` and 0 after it."""
    if step < decay_start:
        if step < warmup:
            return float(step) / float(max(1.0, warmup))
        return 1.0
    progress = min(1.0, float(step - decay_start) / float(max(1, end - decay_start)))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


def make_scheduler(optimizer: torch.optim.Optimizer, spec: Any, horizon: int):
    """The LR scheduler of ``LrScheduleSpec`` over ``horizon`` optimizer steps (None for a constant rate)."""
    if spec.kind == "constant":
        return None
    if spec.kind == "constant_then_cosine":
        warmup, decay_start, end = int(spec.warmup), int(spec.decay_start), int(horizon)
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: constant_then_cosine_factor(step, warmup, decay_start, end))
    from diffusers.optimization import get_scheduler

    if spec.kind == "constant_with_warmup":
        return get_scheduler("constant_with_warmup", optimizer=optimizer, num_warmup_steps=int(spec.warmup),
                             num_training_steps=int(horizon))
    return get_scheduler("cosine_with_restarts", optimizer=optimizer, num_warmup_steps=int(spec.warmup),
                         num_training_steps=int(horizon))


class LoaderBatches:
    """Prompt batches for the step: encode, cached unconditional conditioning, the adapter's noise sampler.

    With a ``GeometryPlan`` each batch is encoded and noised by the adapter's view at its geometry and carries that
    geometry's schedule; ``current_step`` gives the student step the plan's tier depends on.
    """

    def __init__(self, adapter: Any, loader: Any, sampler: Any, device: torch.device, needs_uncond: bool, *,
                 plan: Any = None, schedule_for: Any = None, current_step: Any = None) -> None:
        self.adapter, self.sampler, self.device, self.needs_uncond = adapter, sampler, device, needs_uncond
        self.plan, self.schedule_for, self.current_step = plan, schedule_for, current_step
        self.iterator = iter(loader)
        self.consumed = 0
        self.pixels: list[int] = []     # the geometries used since the last log (their areas)
        self._uncond: dict[tuple, Any] = {}
        self._views: dict[Any, Any] = {}

    def _view(self, geometry: Any) -> Any:
        if geometry is None or geometry == self.adapter.geometry():
            return self.adapter
        if geometry not in self._views:
            self._views[geometry] = self.adapter.at(geometry)
        return self._views[geometry]

    def next(self) -> Batch:
        prompts, metadata = next(self.iterator)
        if not prompts:
            raise RuntimeError("The prompt loader returned an empty batch.")
        self.sampler.mark_batch_consumed()
        self.consumed += 1
        geometry = None
        if self.plan is not None:
            geometry = self.plan.geometry(int(self.current_step()), list(prompts), list(metadata))
        view = self._view(geometry)
        cond = view.encode(list(prompts))
        uncond = None
        if self.needs_uncond:
            key = (geometry, len(prompts))
            if key not in self._uncond:
                self._uncond[key] = view.encode_negative(len(prompts))
            uncond = self._uncond[key]
        noise = view.sample_noise(len(prompts), device=self.device)
        shape = geometry or self.adapter.geometry()
        self.pixels.append(int(shape.height) * int(shape.width))
        schedule = None if view is self.adapter else self.schedule_for(geometry)
        return Batch(cond=cond, uncond=uncond, noise=noise, prompts=list(prompts), geometry=geometry,
                     schedule=schedule)


def resolve_resume(value: Optional[str], output_dir: Path) -> Optional[Path]:
    """The checkpoint to continue from: ``auto`` -> the run directory's latest complete checkpoint (None, a fresh
    start, when it has none), a path -> that checkpoint. Without ``--resume`` the run directory must hold no complete
    checkpoint: a fresh start never mixes with an earlier run's (``SystemExit`` with the message otherwise)."""
    checkpoints = ckpt.complete_checkpoints(output_dir)
    if value is None:
        if checkpoints:
            raise SystemExit(f"{output_dir} already holds checkpoint-{max(checkpoints)}: continue it with --resume "
                             "auto, or start a new run under another run.name")
        return None
    if value == "auto":
        return checkpoints[max(checkpoints)] if checkpoints else None
    path = Path(value).expanduser().resolve()
    try:
        ckpt.checkpoint_step(path)
    except (ValueError, OSError) as error:
        raise SystemExit(f"--resume {value}: {error}") from None
    return path


def memory_state(memory: Any, device: torch.device) -> dict[str, float]:
    """This rank's memory (GiB): allocator peak allocated / reserved since the last reset, and the device's whole used
    memory (``mem_get_info``: also what HCCL / NCCL pools and the runtime hold outside the allocator)."""
    gib = 1024 ** 3
    state = {"allocated_peak": memory.max_memory_allocated(device) / gib,
             "reserved_peak": memory.max_memory_reserved(device) / gib}
    try:
        free, total = memory.mem_get_info(device)
        state["device_used"] = (total - free) / gib
    except (AttributeError, RuntimeError):   # pragma: no cover - backends without mem_get_info
        pass
    return state


def allocator_retries(memory: Any, device: torch.device) -> Optional[float]:
    """This rank's cumulative ``num_alloc_retries`` (the caching allocator freed its whole cache and retried), ``None``
    when the backend does not report it. A count that climbs every step means the allocator is working against its
    cap (``run.memory_fraction``) and the step time pays for it."""
    try:
        value = memory.memory_stats(device).get("num_alloc_retries")
    except (AttributeError, RuntimeError):   # pragma: no cover - backends without memory_stats
        return None
    return None if value is None else float(value)


def reduce_memory(state: dict[str, float], device: torch.device) -> dict[str, float]:
    """Max and min over ranks of each ``memory_state`` value: an imbalance between ranks shows as max >> min."""
    highs = reduce_max({f"performance/{key}_gib_max": value for key, value in state.items()}, device)
    lows = reduce_max({f"performance/{key}_gib_min": -value for key, value in state.items()}, device)
    return {**highs, **{key: -value for key, value in lows.items()}}


def inference_record(adapter: Any, config: Any) -> dict[str, Any]:
    """``weights/inference.json``: the renderable roles (student, and student_ema with EMA) and how to sample."""
    from ..evaluation.protocol import build_inference_record

    train = config.model.roles.student.train
    roles = {"student": train, **({"student_ema": train} if config.method.updates.ema.enabled else {})}
    return build_inference_record(config, adapter, roles=roles)


def train(args: argparse.Namespace) -> None:
    config, resolved = compose(args.run, parse_assignments(args.set))
    output_dir = run_output_dir(config)
    if args.wandb_mode is not None:
        os.environ["WANDB_MODE"] = args.wandb_mode    # overrides run.wandb.mode (edmd.tracking.wandb.effective_mode)
    if args.dry_run:
        print(f"# {output_dir}/config.yaml\n{dump_yaml(resolved)}", end="")
        return

    rank, world_size, device = init_distributed()
    memory = device_lib.module(device)   # torch.npu / torch.cuda: the allocator and memory API (None on the CPU)
    memory_fraction = memory_cap(config.run.memory_fraction, device)
    if memory is not None and memory_fraction is not None:
        # before any allocation: the cap leaves headroom for what HCCL / NCCL allocate outside the allocator
        memory.set_per_process_memory_fraction(memory_fraction, device)
    # rank 0 reads the run directory and every rank follows its answer (after init_distributed: before it every rank
    # counts as rank 0)
    resume, failure = None, None
    if is_main_process():
        try:
            resume = resolve_resume(args.resume, output_dir)
        except SystemExit as error:
            failure = str(error)
    resume, failure = broadcast_object((resume, failure))
    if failure:
        if is_main_process():
            print(failure, file=sys.stderr, flush=True)
        raise SystemExit(1)
    start_step = ckpt.checkpoint_step(resume) if resume is not None else 0
    end_step = (config.method.updates.max_steps if args.stop_after is None
                else min(config.method.updates.max_steps, args.stop_after))
    initial_checkpoint = args.stop_after == 0 and resume is None    # checkpoint-0: the curve's start (coordinator)
    if end_step <= start_step and not initial_checkpoint:
        _log(f"{resume} is at step {start_step}; nothing to train up to step {end_step}")
        return
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.yaml").write_text(dump_yaml(resolved), encoding="utf-8")
        if config.run.eval_every and args.stop_after is None:
            print("note: edmd.cli.train only trains (run.eval_every is read by the periodic-evaluation driver, "
                  "python -m edmd.cli.coordinate)", flush=True)
    barrier()
    _set_seed(config.run.seed, rank, device)
    method = config.method
    adapter = build_adapter(config.model)
    if config.run.data.batch_size > 1 and not adapter.padded_prompt_batches:
        raise ValueError(f"{config.model.name}: prompt batches with different token lengths are not exact (padding "
                         f"shifts positions); use run.data.batch_size 1 (per rank), got {config.run.data.batch_size}")
    if (method.engine.pag is not None
            and type(adapter).predict_velocity_degraded is ModelAdapter.predict_velocity_degraded):
        # before any weight is read: the teacher's first PAG read would fail only after every role is built
        raise ValueError(f"method.engine.pag needs a perturbed-attention teacher read, which {type(adapter).__name__} "
                         f"does not implement (predict_velocity_degraded); drop engine.pag for {config.model.name}")
    # the teacher, the prompt file and the batch split before any model is built: a mistake fails in seconds
    if not expand_path(config.model.path).is_dir():
        raise FileNotFoundError(f"model.path {expand_path(config.model.path)} does not exist: download the teacher "
                                "(README, Installation) or pass --set model.path=<dir>")
    data_path = expand_path(config.run.data.path)
    if not data_path.is_file():
        raise FileNotFoundError(f"run.data.path {data_path} does not exist: see README, Data, or pass "
                                "--set run.data.path=<file>")
    accumulate = micro_batches(config.run.data, world_size)
    started = time.perf_counter()

    # --- roles ------------------------------------------------------------------------------------------------
    base_state = None
    if is_main_process():
        probe = adapter.build_on_meta()
        base_state = adapter.read_base_state(set(probe.state_dict().keys()), dtype=adapter.compute_dtype)
        del probe
        _log(f"read base weights ({len(base_state)} tensors) in {time.perf_counter() - started:.1f}s")
    common = dict(base_state=base_state, device=device, seed=config.run.seed,
                  reshard_after_forward=config.model.reshard_after_forward, read_adapter=adapter.read_role_weights)
    teacher = build_role(adapter, "teacher", RoleSpec(train="frozen", activation_checkpointing=False), **common)
    student = build_role(adapter, "student", role_spec(adapter, config, "student"), **common)
    fake = build_role(adapter, "fake", role_spec(adapter, config, "fake"), **common)
    del base_state, common
    adapter.load_conditioning(device, shard=config.model.text_encoder_shard)
    _log(f"built teacher / student / fake and conditioning in {time.perf_counter() - started:.1f}s")
    if memory is not None:
        built = reduce_memory(memory_state(memory, device), device)
        _log("memory after building (GiB, max/min over ranks): " + ", ".join(
            f"{key.split('/')[1][:-8]} {built[key]:.1f}/{built[key[:-3] + 'min']:.1f}"
            for key in sorted(built) if key.endswith("_gib_max")))

    # --- optimizers ---------------------------------------------------------------------------------------------
    updates = method.updates
    student_params, fake_params = trainable_parameters(student), trainable_parameters(fake)

    def adamw(params: list, spec: Any) -> torch.optim.Optimizer:
        foreach = False if device.type == "cpu" else None   # CPU: torch_npu's patched foreach probe grabs an NPU
        return torch.optim.AdamW(params, lr=spec.lr, betas=tuple(spec.betas), weight_decay=spec.weight_decay,
                                 eps=spec.eps, foreach=foreach)

    student_opt, fake_opt = adamw(student_params, updates.student), adamw(fake_params, updates.fake)
    student_sched = make_scheduler(student_opt, updates.student.schedule, updates.max_steps)
    fake_sched = make_scheduler(fake_opt, updates.fake.schedule, updates.max_steps * updates.ratio)
    ema = None
    if updates.ema.enabled:
        ema = ShadowEma(student, EmaSpec(updates.ema.decay, updates.ema.update_every, updates.ema.warmup,
                                       updates.ema.offset))
    updaters = Updaters(student=student_opt, fake=fake_opt, student_params=student_params, fake_params=fake_params,
                        clip=clip_grad_norm, student_scheduler=student_sched, fake_scheduler=fake_sched, ema=ema,
                        merge=lambda decay: merge_parameters(fake, student, decay))

    # --- data ---------------------------------------------------------------------------------------------------
    dataset = PromptFileDataset(str(data_path), max_samples=config.run.data.max_samples)
    sampler = DistributedKRepeatSampler(dataset=dataset, batch_size=config.run.data.batch_size, k=1,
                                        num_replicas=world_size, rank=rank, seed=config.run.seed, track_consumption=True)

    # --- schedule and step ---------------------------------------------------------------------------------------
    k_steps, family = method.num_steps, method.schedule

    def make_schedule(view: Any) -> Schedule:
        return Schedule(grid=view.student_grid(k_steps, family, device=device).float(),
                        raw_grid=view.raw_grid(k_steps, family, device=device),
                        warp=lambda raw: view.warp(raw, k_steps, family),
                        warp_steps=lambda raw, n: view.warp(raw, n, family))

    schedule = make_schedule(adapter)
    grid = schedule.grid
    schedules: dict[Any, Schedule] = {}

    def schedule_for(geometry: Any) -> Schedule:   # the shift depends on the token count (Qwen: exp(mu(tokens)))
        if geometry not in schedules:
            schedules[geometry] = make_schedule(adapter.at(geometry))
        return schedules[geometry]

    plan = None
    if config.run.data.geometry.mode == "buckets":
        from .geometry import GeometryPlan

        plan = GeometryPlan(adapter, config.run.data.geometry, seed=config.run.seed)
    draws = TorchDraws()

    def shared_randint(name: str, high: int) -> int:
        value = draws.randint(name, high, (1,), device=device)   # every rank draws (DMD2: broadcast from rank 0)
        return int(broadcast_tensor(value)[0])

    pag = config.method.engine.pag
    roles = Roles(student=lambda x, t, c: adapter.predict_velocity(student, x, t, c),
                  fake=lambda x, t, c: adapter.predict_velocity(fake, x, t, c),
                  teacher=lambda x, t, c: adapter.predict_velocity(teacher, x, t, c),
                  guide=adapter.guide, student_module=student, fake_module=fake,
                  teacher_pair=lambda x, t, c, u: adapter.predict_velocity_pair(teacher, x, t, c, u),
                  teacher_degraded=(None if pag is None else
                                    lambda x, t, c: adapter.predict_velocity_degraded(teacher, x, t, c, pag.blocks)))
    step = DistillationStep(method, roles, updaters, None, draws, schedule, shared_randint=shared_randint,
                            accumulate=accumulate)

    # --- resume ---------------------------------------------------------------------------------------------------
    role_modules = {"student": (student, student_opt), "fake": (fake, fake_opt)}
    global_step = 0
    batches_consumed = 0
    if resume is not None:
        state = ckpt.load_checkpoint(resume, roles=role_modules, ema=ema)
        global_step = int(state["step"])
        if student_sched is not None:
            student_sched.load_state_dict(state["student_scheduler"])
        if fake_sched is not None:
            fake_sched.load_state_dict(state["fake_scheduler"])
        if ema is not None:
            ema.load_state_dict(state["ema"])
        batches_consumed, exact = ckpt.restore_runtime_state(resume, rank=rank, world_size=world_size,
                                                             device=device, sampler=sampler, step=global_step)
        if not exact:     # another world size: the data continues, the random streams start afresh
            _set_seed(config.run.seed + global_step, rank, device)
        step.step_index = global_step
        _log(f"resumed {resume} at step {global_step}")
    elif ema is not None:
        ema.copy_from_model()
    loader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler, num_workers=config.run.data.num_workers,
                                         collate_fn=dataset.collate_fn,
                                         generator=torch.Generator().manual_seed(config.run.seed + rank))
    batches = LoaderBatches(adapter, loader, sampler, device, step.needs_uncond, plan=plan, schedule_for=schedule_for,
                            current_step=lambda: step.step_index)
    batches.consumed = batches_consumed
    step.batches = batches

    from ..tracking.wandb import open_training_tracker

    tracker = open_training_tracker(config, output_dir, is_main=is_main_process())
    if resume is not None:
        tracker.check_resume(global_step)
    if resume is not None and config.run.eval_ema and tracker.enabled:
        # the periodic evaluation's raw-vs-EMA table joins this segment's first training row, so W&B shows it as a
        # history panel (the panel then reads the summary copy `edmd.cli.evaluate publish` keeps current)
        from ..tracking.wandb import RESULT_TABLE_KEY, raw_vs_ema_history_table

        try:
            table = raw_vs_ema_history_table(config, output_dir)
            if table is not None:
                tracker.log_pending({RESULT_TABLE_KEY: table}, step=global_step + 1)
        except Exception as error:  # noqa: BLE001 - a display aid must never stop training
            print(f"W&B: {RESULT_TABLE_KEY} not added to the history ({type(error).__name__}: {error})", flush=True)
    record = inference_record(adapter, config)
    if record["grid_float32_hex"] != [float(value).hex() for value in grid.detach().cpu().tolist()]:
        raise RuntimeError("the inference record's grid differs from the training grid")
    weights_metadata = {role: {"train": getattr(config.model.roles, role).train,
                               "lora": (role_spec(adapter, config, role).lora.identity()
                                        if getattr(config.model.roles, role).train == "lora" else None)}
                        for role in ("student", "fake")}
    from .. import platform as platform_lib

    runtime = {"hostname": socket.gethostname(), "world_size": world_size, "adapter": adapter.describe(),
               "platform": platform_lib.detect(), "device": device_lib.describe(device),
               "grid": grid.detach().cpu().tolist(),
               "per_rank_batch_size": config.run.data.batch_size, "micro_batches_per_update": accumulate,
               "data": {"path": str(data_path), "prompts": len(dataset)},
               "float64_on_device": fm.float64_ok(device),   # False on the NPU: float64 steps run on the host
               "memory_fraction": memory_fraction,
               "global_batch_size": config.run.data.batch_size * world_size * accumulate}
    metrics_file = output_dir / "logs" / "metrics.jsonl"   # a local copy of every logged row (W&B or not)
    if is_main_process():
        (output_dir / "runtime.json").write_text(json.dumps(runtime, indent=2, default=str) + "\n", encoding="utf-8")
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
    _log(f"edmd: model={config.model.name}/{config.model.variant} method={method.name} grid={runtime['grid']} "
         f"ratio={updates.ratio} order={updates.order} world={world_size} micro_batches={accumulate} "
         f"global_batch={runtime['global_batch_size']} output={output_dir}")

    def save(at_step: int) -> None:
        began = time.perf_counter()
        if memory is not None:
            memory.empty_cache()   # the checkpoint's collectives need HCCL / NCCL memory outside the allocator
        ckpt.save_checkpoint(
            output_dir, at_step, roles=role_modules,
            role_train={"student": config.model.roles.student.train, "fake": config.model.roles.fake.train},
            ema=ema, weights_metadata=weights_metadata,
            training_state={"step": int(at_step),
                            "student_scheduler": None if student_sched is None else student_sched.state_dict(),
                            "fake_scheduler": None if fake_sched is None else fake_sched.state_dict(),
                            "ema": None if ema is None else ema.state_dict()},
            rank=rank, world_size=world_size, device=device, sampler=sampler, batches_consumed=batches.consumed,
            inference_record=record)
        _log(f"checkpoint-{at_step} written in {time.perf_counter() - began:.1f}s")

    sums: defaultdict[str, float] = defaultdict(float)
    peaks: dict[str, float] = {}
    count = 0
    retries = None if memory is None else allocator_retries(memory, device)
    if initial_checkpoint:
        save(0)
        tracker.log({"progress/prompt_batches": 0.0}, step=0)   # a step-0 row: the score panels then show cp0
    while global_step < end_step:
        if memory is not None:
            memory.reset_peak_memory_stats(device)
        began = time.perf_counter()
        metrics = step.student_step()
        global_step += 1
        metrics["performance/step_seconds"] = time.perf_counter() - began
        if batches.pixels:
            metrics["data/megapixels"] = sum(batches.pixels) / len(batches.pixels) / 1e6
            batches.pixels.clear()
        for key, value in metrics.items():
            if not math.isfinite(value):
                raise FloatingPointError(f"non-finite {key} at step {global_step}: {value}")
            sums[key] += value
        if memory is not None:
            for key, value in memory_state(memory, device).items():
                peaks[key] = max(peaks.get(key, 0.0), value)
        count += 1
        if global_step == 1 or global_step % config.run.log_every == 0 or global_step == end_step:
            averaged = reduce_mean({key: value / count for key, value in sums.items()}, device)
            if peaks:
                averaged.update(reduce_memory(peaks, device))
                averaged["performance/peak_memory_gib"] = averaged["performance/allocated_peak_gib_max"]
            if retries is not None:            # the same backend on every rank: all of them reduce, or none
                now = allocator_retries(memory, device)
                averaged["performance/alloc_retries_per_step_max"] = reduce_max(
                    {"retries": (now - retries) / count}, device)["retries"]
                retries = now
            averaged["optim/student_lr"] = float(student_opt.param_groups[0]["lr"])
            averaged["optim/fake_lr"] = float(fake_opt.param_groups[0]["lr"])
            averaged["progress/prompt_batches"] = float(batches.consumed)
            if ema is not None:
                averaged["ema/decay"] = ema.current_decay()
            tracker.log(averaged, step=global_step)
            if is_main_process():
                with metrics_file.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"step": global_step, **averaged}, allow_nan=True) + "\n")
            _log(f"step {global_step}/{updates.max_steps} generator={averaged.get('loss/generator', float('nan')):.5f} "
                 f"critic={averaged.get('loss/critic', float('nan')):.5f} "
                 f"grad_s={averaged.get('optim/student_grad_norm', float('nan')):.3f} "
                 f"grad_f={averaged.get('optim/fake_grad_norm', float('nan')):.3f} "
                 f"step_s={averaged['performance/step_seconds']:.1f}")
            sums.clear()
            peaks.clear()
            count = 0
        if global_step % config.run.checkpoint_every == 0 or global_step == end_step:
            save(global_step)
    tracker.finish()
    _log(f"finished at step {global_step}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="edmd training (run it under torchrun)")
    # --run-file, not --run: torchrun's own parser reads --run as an abbreviation of its --run-path
    parser.add_argument("--run-file", dest="run", required=True,
                        help="configs/runs/<model>/<method>.yaml, or a run directory's config.yaml")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override a config value, e.g. method.updates.max_steps=500 (repeatable)")
    parser.add_argument("--resume", default=None, metavar="auto|CHECKPOINT",
                        help="auto: continue from the run directory's latest complete checkpoint (or start fresh "
                             "without one); or a checkpoint-<N> directory")
    parser.add_argument("--stop-after", type=int, default=None, metavar="N",
                        help="stop at step N with a checkpoint (edmd.cli.coordinate evaluates between such segments)")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default=None,
                        help="override run.wandb.mode")
    parser.add_argument("--dry-run", action="store_true", help="print the resolved config and exit")
    return parser.parse_args(argv)


__all__ = ["LoaderBatches", "NPU_MEMORY_FRACTION", "inference_record", "make_scheduler", "memory_cap", "micro_batches",
           "parse_args", "resolve_resume", "role_spec", "train"]

"""``python -m edmd.cli.calibrate``: the teacher's calibration table for ``query.waypoint: calibrated``.

The table is measured once per teacher, guidance and student grid, from the teacher alone (base weights, default
sampler), on its own samples X of fixed training prompts. For every landing point ``s = sigma_{k+1}`` and level
``t = s + (top - s) f`` up to the window's top, a Markov pair ``y_s = (1 - s) X + s xi``,
``y_t = sde_renoise(y_s, s, t, z)`` gives ``Delta = A1(y_t, t) - A1(y_s, s)`` (A1: the guided one-step x0 answer)
and ``err_s = E|T_c(y_s, s) - X|^2`` (T_c: the conditional answer). The table's ``d`` is

    d_s(t) = max_phi (E<Delta, phi>)^2 / (E|phi|^2 err_s),    phi in {A1, T_c, A1 - T_c} at (y_t, t),

a lower bound on the drift part of the change, ``E|A1(y_t, t) - E[A1(y_s, s) | y_t]|^2 / err_s``; ``d_total`` is
``E|Delta|^2 / (err_s + eps)``. Means run over the latent's elements, E pools prompts and noise pairs (ratios of
means), and the CIs are prompt-cluster bootstraps.

    plan        --run-file R --out D [--prompts 256] [--seed 20261008] [--exclude P ...]
    shard       --run-file R --plan D/plan.jsonl --device cuda:0 --rank 0 --world N --out D     (one per device)
    aggregate   D --json configs/calibration/<model>.json
    readings    T.json [T.json ...]
    selftest

``--run-file`` names the teacher, ``engine.cfg``, the student grid, corpus and geometry. ``plan`` and ``shard`` ignore
its calibration, so the calibrated run file serves before its table exists; it reads the table as
``calibration: <model>``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Sequence

FORMAT_ROW = "edmd_landing_calibration_row_v1"
FORMAT_PLAN = "edmd_landing_calibration_plan_v1"
#: Steps of the teacher's default sampler (Euler on the pipeline's own grid); else ``shard --teacher-steps``.
TEACHER_STEPS = {"sd3_medium": 28, "flux2_klein": 50, "qwen_image_21": 40, "toy": 12}
#: A landing point's levels ``s + (top - s) f``: dense near ``s`` (f = 0 is the landing point itself, d = 0).
FRACTIONS = (0.0, 1 / 64, 1 / 32, 1 / 16, 1 / 8, 3 / 16, 1 / 4, 5 / 16, 3 / 8, 1 / 2, 5 / 8, 3 / 4, 7 / 8, 1.0)
#: The test functions phi of ``(y_t, t)``: a = A1, c = T_c, g = A1 - T_c.
FEATURES = "acg"
EPS = 1e-8
BOOTSTRAP = 2000                        # prompt-cluster bootstrap draws (seed 0)
PROMPTS_DEFAULT = 256
PLAN_SEED = 20261008
SAMPLE_SEED_BASE = 81_000_000           # teacher sample seed = base + slot (disjoint from the evaluation seeds)
NEAR = 0.05                             # readings: the law's mass within +NEAR of the landing point
#: Per-row statistics summed per prompt by ``aggregate``: one value per level.
LEVEL_KEYS = ("num",) + tuple(f"{kind}_{phi}" for kind in ("ip", "nn") for phi in FEATURES)


def log(message: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {message}", flush=True)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def model_key(model: Any) -> str:
    """The adapter name: ``sd3_medium``, ``flux2_klein``, ``qwen_image_21`` or ``toy``."""
    return str(model["name"] if isinstance(model, dict) else model.name)


def level_row(s: float, top: float) -> list[float]:
    return [float(s) + (float(top) - float(s)) * f for f in FRACTIONS]


def read_query(method: Any) -> Any:
    """The query of the method's coupled generator term: the read the table serves (its margin sets the top)."""
    queries = [term.query for term in method.generator.terms if term.kind == "coupled"]
    if len(queries) != 1:
        raise SystemExit(f"the calibration serves a method with one coupled generator term, found {len(queries)}")
    return queries[0]


def window_top(query: Any) -> float:
    return 1.0 - float(query.margin)


def calibration_holders(doc: Any) -> list[dict]:
    """The mappings of a parsed run file that hold a ``calibration`` key (a calibrated run file has one)."""
    found = []

    def walk(node):
        if isinstance(node, dict):
            if "calibration" in node:
                found.append(node)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(doc)
    return found


def measurement_config(run_file: str) -> Any:
    """The composed run file without the calibrated read's waypoint and table: a run file can be measured before its
    table exists, and the table it holds never enters the measurement."""
    from ..config.core import load_yaml
    from ..config.load import compose

    doc = load_yaml(run_file)
    for query in calibration_holders(doc):
        for key in ("waypoint", "waypoint_noise", "calibration"):
            query.pop(key, None)
    config, _ = compose(doc)
    return config


# ============================================================================ plan (CPU)
def read_corpus(path: Path) -> tuple[list[str], list[dict]]:
    from ..training.data import PromptFileDataset

    dataset = PromptFileDataset(path)
    return list(dataset.prompts), list(dataset.metadata)


def excluded_prompts(paths: Sequence[str]) -> set[str]:
    """Prompt texts the plan must not use: lines of a ``.txt`` file, or the ``prompt`` / ``render`` / ``raw_prompt``
    fields of a ``.jsonl`` file (e.g. the evaluation prompts)."""
    out: set[str] = set()
    for path in map(Path, paths):
        if not path.is_file():
            raise SystemExit(f"exclusion list {path} not found")
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            if path.suffix != ".jsonl":
                out.add(line.strip())
                continue
            record = json.loads(line)
            for key in ("prompt", "render", "raw_prompt"):
                if isinstance(record.get(key), str):
                    out.add(record[key].strip())
    return out


def make_plan(prompts: Sequence[str], metadata: Sequence[dict], *, count: int, seed: int,
              exclude: set[str]) -> list[dict]:
    """``count`` distinct non-empty corpus prompts outside ``exclude`` (each at its first line), seeded, in corpus
    order; slot = position in the plan, sample seed = ``SAMPLE_SEED_BASE + slot``."""
    import numpy as np

    seen: set[str] = set()
    eligible = []
    for line, text in enumerate(prompts):
        key = text.strip()
        if not key or key in seen or key in exclude:
            continue
        seen.add(key)
        eligible.append(line)
    if len(eligible) < count:
        raise SystemExit(f"only {len(eligible)} eligible prompts, {count} asked")
    chosen = sorted(np.random.RandomState(seed).choice(len(eligible), size=count, replace=False).tolist())
    plan = []
    for slot, index in enumerate(chosen):
        line = eligible[index]
        case = {"slot": slot, "line": line, "prompt": prompts[line], "seed": SAMPLE_SEED_BASE + slot}
        ratio = (metadata[line] or {}).get("wh_ratio")
        if ratio is not None:
            case["wh_ratio"] = ratio
        plan.append(case)
    return plan


def run_plan(args) -> None:
    from ..config.core import expand_path

    config = measurement_config(args.run_file)
    corpus = expand_path(config.run.data.path)
    prompts, metadata = read_corpus(corpus)
    exclude = excluded_prompts(args.exclude)
    plan = make_plan(prompts, metadata, count=args.prompts, seed=args.seed, exclude=exclude)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in plan)
    (out / "plan.jsonl").write_text(text, encoding="utf-8")
    info = {"format": FORMAT_PLAN, "run_file": str(args.run_file), "corpus": str(corpus),
            "corpus_sha256": sha256_bytes(corpus.read_bytes()), "corpus_lines": len(prompts), "prompts": len(plan),
            "seed": args.seed, "excluded_texts": len(exclude), "exclude": list(args.exclude),
            "plan_sha256": sha256_bytes(text.encode()),
            "prompt_list_sha256": sha256_bytes("\n".join(case["prompt"] for case in plan).encode())}
    (out / "plan.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    log(f"plan: {len(plan)} prompts from {corpus.name} ({len(prompts)} lines, {len(exclude)} texts excluded) "
        f"-> {out / 'plan.jsonl'} (sha256 {info['plan_sha256'][:12]})")


# ============================================================================ measurement (torch)
def repeat_cond(cond: Any, n: int):
    """A batch-1 conditioning repeated to ``n`` rows: ``expand`` when the class has it (FLUX.2), else every batch-1
    tensor field of the dataclass (SD3, the toy). Qwen-Image-2.1 runs at micro-batch 1 instead."""
    import dataclasses

    import torch

    if int(cond.batch_size) == n:
        return cond
    if hasattr(cond, "expand"):
        return cond.expand(n)
    if dataclasses.is_dataclass(cond):
        values = {}
        for field in dataclasses.fields(cond):
            value = getattr(cond, field.name)
            if not (torch.is_tensor(value) and value.ndim >= 1 and value.shape[0] == 1):
                raise TypeError(f"cannot repeat {type(cond).__name__}.{field.name}")
            values[field.name] = value.expand(n, *value.shape[1:]).contiguous()
        return type(cond)(**values)
    raise TypeError(f"cannot repeat a {type(cond).__name__}")


class TeacherAnswers:
    """``(y [n, ...], t [n]) -> (T_c, A1)``: the conditional and the guided one-step x0 answers (float32),
    micro-batched; on out-of-memory the micro-batch halves (down to 1)."""

    def __init__(self, adapter: Any, teacher: Any, cond: Any, negative: Any, cfg: float, micro: int, device: Any):
        self.adapter, self.teacher, self.cond, self.negative = adapter, teacher, cond, negative
        self.cfg, self.micro, self.device = float(cfg), max(1, int(micro)), device
        self._conds: dict[int, tuple[Any, Any]] = {}

    def _rows(self, n: int):
        if n not in self._conds:
            self._conds[n] = (repeat_cond(self.cond, n),
                              None if self.negative is None else repeat_cond(self.negative, n))
        return self._conds[n]

    def _call(self, y, t):
        from ..algo import fm

        cond, negative = self._rows(int(y.shape[0]))
        if self.cfg == 1.0:
            x0 = fm.x0_from_v(y, self.adapter.predict_velocity(self.teacher, y, t, cond).float(), t)
            return x0, x0
        v_c, v_u = self.adapter.predict_velocity_pair(self.teacher, y, t, cond, negative)
        v_c, v_u = v_c.float(), v_u.float()
        return fm.x0_from_v(y, v_c, t), fm.x0_from_v(y, self.adapter.guide(v_c, v_u, self.cfg), t)

    def __call__(self, y, t):
        import torch

        outs, i, total = [], 0, int(y.shape[0])
        while i < total:
            n = min(self.micro, total - i)
            try:
                outs.append(self._call(y[i:i + n], t[i:i + n]))
            except RuntimeError as error:
                text = str(error).lower()
                if self.micro > 1 and ("out of memory" in text or "oom" in text.split()):
                    self.micro = max(1, self.micro // 2)
                    _empty_cache(self.device)
                    log(f"out of memory at micro-batch {n}; micro-batch now {self.micro}")
                    continue
                raise
            i += n
        return torch.cat([o[0] for o in outs]), torch.cat([o[1] for o in outs])


def _empty_cache(device: Any) -> None:
    import gc

    import torch

    gc.collect()
    kind = torch.device(device).type
    module = getattr(torch, kind, None) if kind in ("npu", "cuda") else None
    if module is not None:
        module.empty_cache()


def seeded_noise(shape, *, slot: int, k: int, r: int, kind: int):
    """The noise of (prompt slot, landing point k, pair r, xi = 0 / z = 1): CPU float32 from its own seeded
    generator, so a row depends on its indices only (not on the shard layout or the device)."""
    import torch

    seed = (int(slot) * 1_000_003 + int(k) * 10_007 + int(r) * 101 + int(kind) * 7 + 20_261_008) % (2 ** 62)
    return torch.randn(tuple(shape), generator=torch.Generator().manual_seed(seed), dtype=torch.float32)


def pair_statistics(X, s: float, levels: Sequence[float], answers: Callable, xi, z) -> dict:
    """The statistics of one Markov pair per row of ``X`` ([S, *latent]) at landing point ``s`` and every level of
    ``levels`` (the first is s itself), per row as float64 lists:

    * ``den`` [S]: ``mean (T_c(y_s, s) - X)^2``;
    * ``num`` [S][L]: ``mean Delta^2``;
    * ``ip_<phi>`` / ``nn_<phi>`` [S][L]: ``mean Delta phi`` and ``mean phi^2`` for phi in ``FEATURES``.

    Every statistic is 0 at the landing point itself."""
    import torch

    from ..algo import fm

    n = int(X.shape[0])
    ones = [1] * (X.ndim - 1)
    s32 = torch.full((n,), float(s), device=X.device, dtype=torch.float32)       # adapters take float32 sigma [B]
    y_s = fm.state(X, xi.to(X.dtype), s32)
    tc_s, a_s = answers(y_s, s32)

    def mse(e):
        return fm.wide(e).reshape(e.shape[0], -1).square().mean(dim=1).double().cpu()

    def inner(e, f):
        return (fm.wide(e) * fm.wide(f)).reshape(e.shape[0], -1).mean(dim=1).double().cpu()

    den = mse(tc_s - X)
    zero = torch.zeros(n, dtype=torch.float64)
    columns = {key: [zero] for key in LEVEL_KEYS}
    coarse = [float(t) for t in levels[1:]]
    if coarse:
        m = len(coarse)
        t_rows = torch.tensor(coarse, device=X.device, dtype=torch.float32).repeat_interleave(n)   # level-major
        y_t = fm.sde_renoise(y_s.repeat(m, *ones), s32.repeat(m), t_rows, z.to(X.dtype).repeat(m, *ones))
        tc_t, a_t = answers(y_t, t_rows)
        change = a_t - a_s.repeat(m, *ones)
        values = {"num": mse(change)}
        for phi, feature in (("a", a_t), ("c", tc_t), ("g", a_t - tc_t)):
            values[f"ip_{phi}"] = inner(change, feature)
            values[f"nn_{phi}"] = mse(feature)
        for key, value in values.items():
            value = value.reshape(m, n)
            columns[key].extend(value[j] for j in range(m))
    return {"den": den.tolist(), **{key: torch.stack(rows, dim=1).tolist() for key, rows in columns.items()}}


def teacher_sample(view: Any, teacher: Any, prompt: str, *, seed: int, steps: int, cfg: float, device: Any):
    """The teacher's own sample as a teacher reference image renders it (Euler on the pipeline grid at CFG ``cfg``;
    ``edmd.evaluation.generate``): float32 latents [1, ...] and the grid."""
    from ..evaluation.generate import resolve_grid, sample_teacher

    grid = resolve_grid(view, {"kind": "pipeline", "num_steps": int(steps)})
    generator = view.eval_generator(int(seed), device)
    with view.sampling_context(device):
        cond = view.encode([prompt])
        negative = view.encode_negative(1) if cfg != 1.0 else None
        noise = view.sample_noise(1, device=device, generator=generator).to(view.sampling_state_dtype)
        latents = sample_teacher(view, teacher, cond, noise, grid, cfg=cfg, negative=negative, generator=generator)
    return latents.float(), grid


def case_view(adapter: Any, config: Any, case: dict):
    """The adapter view a prompt trains at: the run's geometry (``buckets``: the prompt's aspect ratio in the run's
    single tier with positive weight; ``fixed``: the model's own geometry)."""
    spec = getattr(config.run.data, "geometry", None)
    if spec is None or getattr(spec, "mode", "fixed") == "fixed":
        return adapter
    from ..training.geometry import aspect_for_prompt

    tiers = [tier for tier, weight in spec.tiers.items() if float(weight) > 0]
    if len(tiers) != 1:
        raise SystemExit(f"calibration supports one geometry tier, the run has {sorted(spec.tiers)}")
    table = adapter.geometry_table()[tiers[0]]
    metadata = {"wh_ratio": case["wh_ratio"]} if "wh_ratio" in case else None
    ratio = aspect_for_prompt(case["prompt"], metadata, spec.aspect_weights)
    return adapter.at(table.get(ratio, table["1:1"]))


def measure_case(view: Any, teacher: Any, config: Any, case: dict, *, device: Any, steps: int, noises: int,
                 micro: int, header: dict) -> list[dict]:
    """Every row of one prompt: the teacher's sample, then one row per landing point and noise pair."""
    import torch

    method = config.method
    cfg = float(method.engine.cfg)
    top = window_top(read_query(method))
    X, pipeline_grid = teacher_sample(view, teacher, case["prompt"], seed=int(case["seed"]), steps=steps, cfg=cfg,
                                      device=device)
    if not bool(torch.isfinite(X).all()):
        raise FloatingPointError(f"slot {case['slot']}: non-finite teacher sample")
    cond = view.encode([case["prompt"]])                              # the conditioning of the training reads
    negative = view.encode_negative(1) if cfg != 1.0 else None
    answers = TeacherAnswers(view, teacher, cond, negative, cfg, micro, device)
    grid = view.student_grid(int(method.num_steps), str(method.schedule)).detach().float().cpu()
    rows = []
    for k in range(int(method.num_steps) - 1):
        s = float(grid[k + 1])
        levels = level_row(s, top)
        for r in range(int(noises)):
            xi = seeded_noise(X.shape, slot=case["slot"], k=k, r=r, kind=0).to(device)
            z = seeded_noise(X.shape, slot=case["slot"], k=k, r=r, kind=1).to(device)
            stats = pair_statistics(X, s, levels, answers, xi, z)
            rows.append({**header, "format": FORMAT_ROW, "slot": case["slot"], "line": case.get("line"),
                         "seed": case["seed"], "wh_ratio": case.get("wh_ratio"),
                         "geometry": [int(view.geometry().height), int(view.geometry().width)],
                         "k": k, "s": s, "r": r, "levels": levels, "den": stats["den"][0],
                         **{key: stats[key][0] for key in LEVEL_KEYS},
                         "student_grid": [float(g) for g in grid], "pipeline_grid_len": int(pipeline_grid.numel()),
                         "x_rms": float(X.reshape(-1).square().mean().sqrt())})
    return rows


def run_shard(args) -> None:
    import torch

    from ..models.base import build_adapter
    from ..runtime import device as device_lib

    config = measurement_config(args.run_file)
    method = config.method
    key = model_key(config.model)
    steps = int(args.teacher_steps or TEACHER_STEPS.get(key, 0))
    if steps <= 0:
        raise SystemExit(f"no default teacher sampler for {key}: pass --teacher-steps")
    device = torch.device(args.device)
    device_lib.set_device(device)
    torch.set_grad_enabled(False)
    adapter = build_adapter(config.model)
    plan_bytes = Path(args.plan).read_bytes()
    plan = [json.loads(line) for line in plan_bytes.decode("utf-8").splitlines() if line.strip()]
    cases = [case for case in plan if int(case["slot"]) % args.world == args.rank]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    out_path = out / f"rank-{args.rank}.jsonl"
    status_path = out / f"rank-{args.rank}.status.json"
    micro = int(args.micro_batch or (1 if key == "qwen_image_21" else 4))
    header = {"model": key, "variant": getattr(config.model, "variant", None), "cfg": float(method.engine.cfg),
              "num_steps": int(method.num_steps), "schedule": str(method.schedule), "teacher_steps": steps,
              "sampler": "euler", "run_file": str(args.run_file), "device": str(device), "rank": args.rank,
              "world": args.world, "plan_sha256": sha256_bytes(plan_bytes), "noises": int(args.noises)}
    status = {"model": key, "rank": args.rank, "started": time.time(), "prompts": len(cases), "prompts_done": 0}
    status_path.write_text(json.dumps(status))
    log(f"{key}: cfg {header['cfg']:g}, teacher {steps}-step Euler, {len(cases)} prompts on rank {args.rank}/"
        f"{args.world}, {args.noises} noise pairs, micro-batch {micro}")
    adapter.load_conditioning(device)
    teacher = adapter.load_inference_denoiser(None, device).eval().requires_grad_(False)
    with out_path.open("w") as handle:
        for case in cases:
            tick = time.time()
            rows = measure_case(case_view(adapter, config, case), teacher, config, case, device=device, steps=steps,
                                noises=args.noises, micro=micro, header=header)
            for row in rows:
                handle.write(json.dumps(row) + "\n")
            handle.flush()
            status["prompts_done"] += 1
            status_path.write_text(json.dumps(status))
            log(f"slot {case['slot']}: {len(rows)} rows in {time.time() - tick:.1f}s")
    status.update(phase="done", elapsed=time.time() - status["started"])
    status_path.write_text(json.dumps(status))
    log(f"done: {status['prompts_done']} prompts -> {out_path}")


# ============================================================================ aggregate
def load_rows(directory: Path) -> list[dict]:
    rows = []
    for path in sorted(Path(directory).glob("rank-*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("format") == FORMAT_ROW:
                    rows.append(row)
    return rows


def check_coverage(directory: Path, rows: list[dict]) -> None:
    """When ``directory/plan.jsonl`` is the plan the rows were measured from, every planned prompt must have rows at
    every landing point: an unfinished shard is an error, not a smaller table."""
    path = Path(directory) / "plan.jsonl"
    if not path.is_file() or sha256_bytes(path.read_bytes()) != rows[0].get("plan_sha256"):
        return
    planned = {int(json.loads(line)["slot"]) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    for k in range(int(rows[0]["num_steps"]) - 1):
        missing = sorted(planned - {int(row["slot"]) for row in rows if int(row["k"]) == k})
        if missing:
            raise SystemExit(f"landing point {k}: no rows for {len(missing)} of {len(planned)} planned prompts "
                             f"(slots {missing[:8]}{' ...' if len(missing) > 8 else ''}); finish their shards")


def bootstrap(per_prompt, stat: Callable) -> tuple[float, float, float]:
    """``stat`` of the per-prompt rows and its 95% prompt-cluster bootstrap interval (``BOOTSTRAP`` draws, seed 0)."""
    import numpy as np

    a = np.asarray(per_prompt, dtype=float)
    estimate = stat(a)
    if len(a) < 2:
        return estimate, float("nan"), float("nan")
    idx = np.random.default_rng(0).integers(0, len(a), size=(BOOTSTRAP, len(a)))
    draws = np.array([stat(a[i]) for i in idx])
    return estimate, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def ratio_of_means(rows) -> float:
    """``sum(num) / (sum(den) + count eps)`` over per-prompt rows ``[num, den, count]``."""
    return float(rows[:, 0].sum() / (rows[:, 1].sum() + EPS * rows[:, 2].sum()))


def moment_r(rows) -> float:
    """``sum(ip) / sqrt(sum(den) sum(nn))`` over per-prompt rows ``[ip, den, nn]``: the signed moment whose square is
    a lower bound on the drift (0 when phi vanishes, e.g. A1 - T_c at cfg 1)."""
    scale = float(rows[:, 1].sum()) * float(rows[:, 2].sum())
    return float(rows[:, 0].sum()) / math.sqrt(scale) if scale > 0 else 0.0


def squared_ci(lo: float, hi: float) -> list[float]:
    """The interval of r^2 given an interval of r."""
    if lo <= 0.0 <= hi:
        return [0.0, max(lo * lo, hi * hi)]
    return sorted([lo * lo, hi * hi])


def drift_bound(moments: dict, moments_ci: dict) -> tuple[list[list[float]], list[list[list[float]]]]:
    """The table's ``d`` and ``ci95`` from the signed moments: per landing point k and level j, ``max_phi r_phi^2``
    (0 at the landing point itself) and the squared interval of the maximising phi."""
    d, ci = [], []
    for k, row in enumerate(moments[FEATURES[0]]):
        d_row, ci_row = [], []
        for j in range(len(row)):
            phi = max(FEATURES, key=lambda p: moments[p][k][j] ** 2)
            d_row.append(0.0 if j == 0 else moments[phi][k][j] ** 2)
            ci_row.append([0.0, 0.0] if j == 0 else squared_ci(*moments_ci[phi][k][j]))
        d.append(d_row)
        ci.append(ci_row)
    return d, ci


def per_prompt_sums(rows: list[dict], levels: int) -> dict[int, dict]:
    """Per prompt slot (in row order): ``den`` and the row ``count``, and every ``LEVEL_KEYS`` statistic ([levels]),
    summed over its rows."""
    import numpy as np

    sums: dict[int, dict] = {}
    for row in rows:
        entry = sums.setdefault(int(row["slot"]), {"den": 0.0, "count": 0,
                                                    **{key: np.zeros(levels) for key in LEVEL_KEYS}})
        entry["den"] += float(row["den"])
        entry["count"] += 1
        for key in LEVEL_KEYS:
            entry[key] += np.asarray(row[key], dtype=float)
    return sums


def summarise(rows: list[dict]) -> dict:
    """The measurement: per landing point k (the median landing point over the rows' geometries), its levels, ``d``
    (the drift bound) with its CI, ``d_total`` with its CI, the signed moments ``moment_r`` behind ``d`` and counts.
    :func:`table_text` keeps what training reads."""
    import numpy as np

    if not rows:
        raise SystemExit("no rows")
    first = rows[0]
    keys = ("model", "variant", "cfg", "num_steps", "schedule", "teacher_steps", "plan_sha256")
    for row in rows:
        if any(row.get(key) != first.get(key) for key in keys):
            raise SystemExit(f"rows disagree on {[key for key in keys if row.get(key) != first.get(key)]}")
    seen = set()
    # each (prompt, landing point, noise pair) is measured once; a second copy would weigh that prompt twice
    for row in rows:
        key = (int(row["slot"]), int(row["k"]), int(row["r"]))
        if key in seen:
            raise SystemExit(f"two rows for slot {key[0]}, landing point {key[1]}, noise pair {key[2]}: the directory "
                             "mixes shards of two measurements (e.g. rank-*.jsonl left from another --world); remove "
                             "the stale files and aggregate again")
        seen.add(key)
    missing = sorted({key for row in rows for key in LEVEL_KEYS if key not in row})
    if missing:
        raise SystemExit(f"rows without {missing}: measure them with this tool")
    K = int(first["num_steps"])
    top = float(first["levels"][-1])
    landing, levels, counts, prompts, spread, den_mean = [], [], [], [], [], []
    total, total_ci = [], []
    moments = {phi: [] for phi in FEATURES}
    moments_ci = {phi: [] for phi in FEATURES}
    for k in range(K - 1):
        mine = [row for row in rows if int(row["k"]) == k]
        if not mine:
            raise SystemExit(f"no rows for landing point {k}")
        L = len(mine[0]["levels"])
        if any(len(row["num"]) != L for row in mine):
            raise SystemExit(f"landing point {k}: rows with different level counts")
        s_values = np.array([row["s"] for row in mine])
        s_med = float(np.median(s_values))
        landing.append(s_med)
        levels.append(level_row(s_med, top))
        spread.append([float(s_values.min()), float(s_values.max())])
        sums = list(per_prompt_sums(mine, L).values())
        row_d, row_ci = [], []
        for j in range(L):
            est, lo, hi = bootstrap([[e["num"][j], e["den"], e["count"]] for e in sums], ratio_of_means)
            row_d.append(0.0 if j == 0 else est)
            row_ci.append([0.0, 0.0] if j == 0 else [lo, hi])
        total.append(row_d)
        total_ci.append(row_ci)
        for phi in FEATURES:
            row_r, row_r_ci = [], []
            for j in range(L):
                est, lo, hi = bootstrap([[e[f"ip_{phi}"][j], e["den"], e[f"nn_{phi}"][j]] for e in sums], moment_r)
                row_r.append(0.0 if j == 0 else est)
                row_r_ci.append([0.0, 0.0] if j == 0 else [lo, hi])
            moments[phi].append(row_r)
            moments_ci[phi].append(row_r_ci)
        counts.append(len(mine))
        prompts.append(len(sums))
        den_mean.append(float(np.mean([row["den"] for row in mine])))
    d, ci = drift_bound(moments, moments_ci)
    return {"model": first["model"], "variant": first.get("variant"), "cfg": first["cfg"], "num_steps": K,
            "teacher_steps": first["teacher_steps"], "top": top, "landing": landing, "levels": levels,
            "d": d, "ci95": ci, "d_total": total, "ci95_total": total_ci,
            "moment_r": moments, "moment_r_ci95": moments_ci,
            "rows": counts, "prompts": prompts, "landing_spread": spread, "den_mean": den_mean}


def table_text(table: dict) -> str:
    """The table as ``configs/calibration/<model>.json`` holds it: the teacher, its guidance and grid size, and the
    numbers training reads (levels to 8 significant digits, d to 6), one row per line."""
    rounded = {"landing": [float("%.8g" % v) for v in table["landing"]],
               "levels": [[float("%.8g" % v) for v in row] for row in table["levels"]],
               "d": [[float("%.6g" % v) for v in row] for row in table["d"]]}
    head = {key: table[key] for key in ("model", "variant", "cfg", "num_steps")}
    lines = [f"  {json.dumps(key)}: {json.dumps(value)}," for key, value in head.items()]
    lines.append(f'  "landing": {json.dumps(rounded["landing"])},')
    for key in ("levels", "d"):
        rows = ",\n".join(f"    {json.dumps(row)}" for row in rounded[key])
        lines.append(f'  "{key}": [\n{rows}\n  ]' + ("," if key == "levels" else ""))
    return "{\n" + "\n".join(lines) + "\n}\n"


def as_calibration(table: dict) -> SimpleNamespace:
    """A table as ``query.calibration`` reads it (landing / levels / d)."""
    return SimpleNamespace(landing=tuple(float(v) for v in table["landing"]),
                           levels=tuple(tuple(float(v) for v in row) for row in table["levels"]),
                           d=tuple(tuple(float(v) for v in row) for row in table["d"]))


def table_readings(table: dict, near: float = NEAR) -> list[dict]:
    """Each step's law on the table's own grid (the landing points it was measured at), with the delivered query's
    lo = 0.02 and the window top in the query's float32 arithmetic (``LandingLaw.summary``)."""
    import torch

    from ..algo.calibration import readings

    grid = torch.tensor([1.0] + list(table["landing"]) + [0.0], dtype=torch.float32)
    top = float(table.get("top", table["levels"][0][-1]))
    upper = float(torch.tensor(1.0, dtype=torch.float32) - (1.0 - top))
    return readings(as_calibration(table), grid, lo=0.02, upper=upper, near=near)


def print_table(table: dict) -> None:
    """A table (``configs/calibration/*.json``) or a fresh measurement (with its CIs and counts) and the law it gives
    each step."""
    variant = f"/{table['variant']}" if table.get("variant") else ""
    measured = f"  teacher {table['teacher_steps']}-step Euler  rows {table['rows']}  prompts {table['prompts']}" \
        if "rows" in table else ""
    print(f"== {table['model']}{variant}  cfg {table['cfg']:g}{measured}")
    for k, (s, r) in enumerate(zip(table["landing"], table_readings(table))):
        if "landing_spread" in table:
            lo, hi = table["landing_spread"][k]
            print(f"  step {k}  s = {s:.4f} (spread {lo:.4f}-{hi:.4f})  err_s = {table['den_mean'][k]:.4g}")
            print("    d(t):       " + "  ".join(f"{t:.4f}:{v:.3f}[{a:.3f},{b:.3f}]" for t, v, (a, b) in
                                              zip(table["levels"][k][1:], table["d"][k][1:], table["ci95"][k][1:])))
        else:
            print(f"  step {k}  s = {s:.4f}")
            print("    d(t):       " + "  ".join(f"{t:.4f}:{v:.3f}" for t, v in
                                              zip(table["levels"][k][1:], table["d"][k][1:])))
        if "d_total" in table:
            print("    d_total(t): " + "  ".join(f"{t:.4f}:{v:.3f}" for t, v in
                                              zip(table["levels"][k][1:], table["d_total"][k][1:])))
        print(f"    law: E[t] = {r['mean']:.4f} (uniform {(r['landing'] + r['upper']) / 2:.4f}), effective support "
              f"{r['effective_support_fraction']:.3f} (uniform 1), P(t < s + {NEAR:g}) = "
              f"{r[f'mass_within_{NEAR:g}']:.3f} (uniform {r['uniform_mass_within']:.3f})")


def run_aggregate(args) -> None:
    rows = load_rows(Path(args.dir))
    if rows:
        check_coverage(Path(args.dir), rows)
    table = summarise(rows)
    print_table(table)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(table_text(table))
        print(f"table -> {args.json} (a run file reads it as calibration: {Path(args.json).stem})")


def run_readings(args) -> None:
    for path in args.tables:
        print_table(json.loads(Path(path).read_text()))


# ============================================================================ self-test (CPU)
def _ok(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"  ok  {message}")


class GaussianAnswers:
    """Closed-form teacher answers on elementwise Gaussian data ``X ~ N(mu, v)``, the guided law the teacher samples at
    cfg ``w`` (``mu = w mu_c - (w - 1) mu_u``). At ``drift = 0``, A1 is the posterior mean of the law X is drawn from:
    ``E[A1(y_s, s) | y_t] = A1(y_t, t)``, so every moment ``E<Delta, phi>`` vanishes."""

    def __init__(self, mu_c: float, mu_u: float, v: float, w: float, drift: float = 0.0):
        self.mu_c, self.mu_u, self.v, self.w, self.drift = mu_c, mu_u, v, w, drift
        self.mu = w * mu_c - (w - 1) * mu_u

    def k(self, t):
        return (1 - t) * self.v / ((1 - t) ** 2 * self.v + t ** 2)

    def __call__(self, y, t):
        tt = t.reshape(-1, *([1] * (y.ndim - 1))).to(y.dtype)
        k = self.k(tt)
        m = self.mu + self.drift * tt
        return self.mu_c + k * (y - (1 - tt) * self.mu_c), m + k * (y - (1 - tt) * m)

    def expected(self, s: float, t: float) -> dict:
        """Per element: ``den = E(T_c(y_s) - X)^2``, ``num = E Delta^2`` and ``ip_<phi>`` / ``nn_<phi>``.

        With ``y_t - (1 - t) mu = a (y_s - (1 - s) mu) + q z`` (``a = (1 - t) / (1 - s)``, ``q^2 = t^2 - a^2 s^2``) and
        ``c_t = 1 - k_t (1 - t)``: ``Delta = (k_t a - k_s)(y_s - (1 - s) mu) + k_t q z + drift (t c_t - s c_s)``.
        Each phi is affine in ``y_t`` and ``k_t Var(y_t) = (1 - t) v = a k_s Var(y_s)``, so ``Cov(Delta, phi) = 0``
        and ``E Delta phi = E Delta E phi``."""
        a = (1 - t) / (1 - s)
        ks, kt = self.k(s), self.k(t)
        cs, ct = 1 - ks * (1 - s), 1 - kt * (1 - t)
        var_s, var_t = (1 - s) ** 2 * self.v + s ** 2, (1 - t) ** 2 * self.v + t ** 2
        mean_delta = self.drift * (t * ct - s * cs)
        out = {"den": cs ** 2 * ((self.mu_c - self.mu) ** 2 + self.v) + (ks * s) ** 2,
               "num": mean_delta ** 2 + (kt * a - ks) ** 2 * var_s + kt ** 2 * (t ** 2 - (a * s) ** 2)}
        m_t = self.mu + self.drift * t
        features = {"a": (m_t * ct + kt * (1 - t) * self.mu, kt ** 2 * var_t),          # (E phi, Var phi)
                    "c": (self.mu_c * ct + kt * (1 - t) * self.mu, kt ** 2 * var_t),
                    "g": ((m_t - self.mu_c) * ct, 0.0)}
        for phi, (mean, var) in features.items():
            out[f"ip_{phi}"] = mean_delta * mean
            out[f"nn_{phi}"] = mean ** 2 + var
        return out


def _selftest_gaussian() -> None:
    import torch

    print("1. Gaussian teacher: pair_statistics matches the closed forms of d_total and of the moments r_phi")
    gen = torch.Generator().manual_seed(0)
    for drift in (0.0, 0.5):
        gauss = GaussianAnswers(mu_c=0.3, mu_u=-0.2, v=0.6, w=4.0, drift=drift)
        X = gauss.mu + math.sqrt(gauss.v) * torch.randn(2, 64, 64, 64, generator=gen, dtype=torch.float64)
        for s in (0.5, 0.767, 0.9674):
            levels = level_row(s, 0.99)
            xi = torch.randn(X.shape, generator=gen, dtype=torch.float64)
            z = torch.randn(X.shape, generator=gen, dtype=torch.float64)
            stats = pair_statistics(X, s, levels, gauss, xi, z)
            den = sum(stats["den"])
            for j in (1, 4, 8, len(levels) - 1):
                want = gauss.expected(s, levels[j])
                got = sum(row[j] for row in stats["num"]) / (den + EPS * len(stats["den"]))
                closed = want["num"] / (want["den"] + EPS)
                _ok(abs(got / closed - 1) < 0.02,
                    f"drift {drift} s {s} t {levels[j]:.4f}: d_total {got:.4f} vs closed form {closed:.4f} (2%)")
                for phi in FEATURES:
                    ip, nn = (sum(row[j] for row in stats[f"{key}_{phi}"]) for key in ("ip", "nn"))
                    r = ip / math.sqrt(den * nn) if den * nn > 0 else 0.0
                    scale = want["den"] * want[f"nn_{phi}"]
                    r_closed = want[f"ip_{phi}"] / math.sqrt(scale) if scale > 0 else 0.0
                    _ok(abs(r - r_closed) < 0.002 + 0.02 * abs(r_closed),
                        f"drift {drift} s {s} t {levels[j]:.4f}: r_{phi} {r:+.4f} vs closed form {r_closed:+.4f}")
                    _ok(r * r <= got * (1 + 1e-6), f"drift {drift} s {s} t {levels[j]:.4f}: r_{phi}^2 <= d_total")
            _ok(all(row[0] == 0.0 for key in LEVEL_KEYS for row in stats[key]),
                f"drift {drift} s {s}: every statistic is 0 at the landing point")


def _selftest_toy() -> None:
    import tempfile

    import torch

    from ..algo.calibration import step_laws
    from ..config.core import loads_yaml
    from ..config.load import REPO, compose
    from ..models.toy import ToyAdapter

    print("2. the toy adapter end to end: plan, shard rows, aggregate, readings, the run-time law, the run files")
    adapter = ToyAdapter(dtype=torch.float32, tokens=64, channels=4, seed=0)
    teacher = adapter.build_model(seed=0).eval()
    NS = SimpleNamespace
    config = NS(run=NS(data=NS(geometry=None)),          # the slice of a composed config the measurement reads
                method=NS(num_steps=4, schedule="native", engine=NS(cfg=4.0),
                          generator=NS(terms=(NS(kind="coupled", query=NS(margin=0.01)),))))
    plan = make_plan([f"toy prompt {i}" for i in range(40)] + ["", "toy prompt 3"], [{}] * 42, count=12, seed=1,
                     exclude={"toy prompt 5"})
    _ok(len(plan) == 12 and all(c["prompt"] != "toy prompt 5" for c in plan)
        and len({c["prompt"] for c in plan}) == 12, "plan: 12 distinct prompts, the excluded one absent")
    plan_text = "".join(json.dumps(case) + "\n" for case in plan)
    header = {"model": "toy", "variant": None, "cfg": 4.0, "num_steps": 4, "schedule": "native", "teacher_steps": 12,
              "sampler": "euler", "plan_sha256": sha256_bytes(plan_text.encode()), "noises": 2}
    rows = []
    for case in plan:
        rows.extend(measure_case(adapter, teacher, config, case, device=torch.device("cpu"), steps=12, noises=2,
                                 micro=3, header=header))
    _ok(len(rows) == 12 * 3 * 2, f"{len(rows)} rows (12 prompts x 3 landing points x 2 pairs)")
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "plan.jsonl").write_text(plan_text)
        (Path(tmp) / "rank-0.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows[6:]))
        try:
            check_coverage(Path(tmp), load_rows(Path(tmp)))
            refused = False
        except SystemExit:
            refused = True
        _ok(refused, "aggregate refuses rows that miss a planned prompt")
        (Path(tmp) / "rank-1.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows[:6]))
        check_coverage(Path(tmp), load_rows(Path(tmp)))
        table = summarise(load_rows(Path(tmp)))
    grid = adapter.student_grid(4, "native")
    _ok(all(abs(a - float(b)) < 1e-6 for a, b in zip(table["landing"], grid[1:-1])),
        "landing = the toy grid's waypoints")
    _ok(all(row[0] == 0.0 and all(v >= 0 for v in row) for row in table["d"]), "d(s) = 0, d >= 0")
    _ok(all(v <= w * (1 + 1e-6) for d_row, t_row in zip(table["d"], table["d_total"]) for v, w in zip(d_row, t_row)),
        "d <= d_total (Cauchy-Schwarz)")
    for name, ci_name in (("d", "ci95"), ("d_total", "ci95_total")):
        _ok(all(lo <= v <= hi for k in range(3) for v, (lo, hi) in zip(table[name][k][1:], table[ci_name][k][1:])),
            f"every {name} inside its bootstrap CI")
    _ok(table["d"] == drift_bound(table["moment_r"], table["moment_r_ci95"])[0], "d = max_phi r_phi^2")
    reads = table_readings(table)
    _ok(len(reads) == 3 and all(0.0 < r["effective_support_fraction"] <= 1.0 + 1e-9 for r in reads),
        "readings per step, effective support in (0, 1]")
    print_table(table)
    upper = float(torch.tensor(1.0) - 0.01)
    laws = step_laws(as_calibration(table), grid, lo=0.02, upper=upper)
    _ok(len(laws) == 3, "the table yields the run-time laws on its grid")
    saved = json.loads(table_text(table))
    _ok(saved["landing"] == [float("%.8g" % v) for v in table["landing"]]
        and len(step_laws(as_calibration(saved), grid, lo=0.02, upper=upper)) == 3,
        "the saved table (configs/calibration/<model>.json) yields the laws too")
    for path in sorted((REPO / "configs" / "runs").rglob("*.yaml")):
        named = [query["calibration"] for query in calibration_holders(loads_yaml(path.read_text()))]
        if not named:
            continue
        shipped = json.loads((REPO / "configs" / "calibration" / f"{named[0]}.json").read_text())
        query = read_query(compose(path)[0].method)
        _ok(list(query.calibration.landing) == shipped["landing"] and list(query.calibration.d[0]) == shipped["d"][0]
            and len(table_readings(shipped)) == len(shipped["landing"]),
            f"{path.relative_to(REPO).as_posix()} reads configs/calibration/{named[0]}.json")


def selftest(args) -> None:
    _selftest_gaussian()
    _selftest_toy()
    print("selftest ok")


# ============================================================================ CLI
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m edmd.cli.calibrate", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    plan = sub.add_parser("plan", help="choose the prompts (CPU)")
    plan.add_argument("--run-file", required=True)
    plan.add_argument("--out", required=True)
    plan.add_argument("--prompts", type=int, default=PROMPTS_DEFAULT)
    plan.add_argument("--seed", type=int, default=PLAN_SEED)
    plan.add_argument("--exclude", action="append", default=[],
                      help="a .txt / .jsonl file of prompts the plan must not use (repeatable)")
    shard = sub.add_parser("shard", help="measure one device's share of the plan")
    shard.add_argument("--run-file", required=True)
    shard.add_argument("--plan", required=True)
    shard.add_argument("--device", required=True)
    shard.add_argument("--rank", type=int, required=True)
    shard.add_argument("--world", type=int, required=True)
    shard.add_argument("--out", required=True)
    shard.add_argument("--noises", type=int, default=2, help="noise pairs (xi, z) per landing point and prompt")
    shard.add_argument("--teacher-steps", type=int, default=0, help="0 = the adapter's default sampler")
    shard.add_argument("--micro-batch", type=int, default=0, help="0 = 4 (Qwen-Image-2.1: 1)")
    agg = sub.add_parser("aggregate", help="the table from the shard rows")
    agg.add_argument("dir")
    agg.add_argument("--json", default=None)
    rd = sub.add_parser("readings", help="print tables and the read law they give each step")
    rd.add_argument("tables", nargs="+")
    sub.add_parser("selftest", help="CPU self-test")
    args = parser.parse_args(argv)
    {"plan": run_plan, "shard": run_shard, "aggregate": run_aggregate, "readings": run_readings,
     "selftest": selftest}[args.cmd](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

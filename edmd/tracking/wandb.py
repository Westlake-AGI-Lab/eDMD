"""W&B for eDMD runs: one run per output directory, in the model's project unless ``run.wandb.project`` names another.

Run identity (project, name = the output directory name, config, tags, notes, group) is set once, by the run's first
writer. Every later writer (each training start, each ``edmd.cli.evaluate publish``) resumes by id with
``resume="must"`` and never passes identity again: re-passing the name would revert a name edited in the UI.

Online mode needs W&B credentials, and a training process cannot answer wandb's login prompt: without credentials a
writer warns and logs offline instead (``offline_without_credentials``).

The periodic evaluation (``edmd.cli.coordinate``) is published by ``publish_evaluations`` (``edmd.cli.evaluate
publish --run``), a short-lived writer that rebuilds every point from the results on disk -- the raw student's
``<run>/eval/<eval name>/checkpoint-<N>/`` and, with ``run.eval_ema``, the EMA lane's ``checkpoint-<N>-ema/`` -- as
summary keys only (``tracking.media``), so publishing again after a re-score rewrites them in place.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

RUN_STATE = "wandb_run.json"
RUN_ID_FILE = "wandb_run_id.txt"
#: run.eval_ema: the EMA scores are published only as this table, no curves, no images
RESULT_TABLE_KEY = "result/raw_vs_ema"
GALLERY_KEY = "eval/images_inline"
HISTORY_NOTE_KEY = "eval/history_note"
HISTORY_NOTE = ("Evaluation charts and the image gallery read summary keys only (eval/curve_<step>_<metric>, "
                "eval/curve_latest_*, the gallery's summary table); this run writes no eval/* history rows, so "
                "any such row is not authoritative. Publishing again (after a re-score) overwrites the summary in "
                "place.")
MAX_TAG_LENGTH = 64
_SENSITIVE_PARTS = ("api_key", "apikey", "password", "secret", "access_key", "private_key", "api_token",
                    "auth_token", "access_token", "bearer", "credential")


class WandbError(RuntimeError):
    pass


def sanitize(value: Any, key: str = "") -> Any:
    """JSON-safe copy with likely credentials redacted (configs carry none; this is a second fence)."""
    lowered = key.lower()
    if lowered in ("token", "key") or any(part in lowered for part in _SENSITIVE_PARTS):
        return "<redacted>"
    if isinstance(value, Mapping):
        return {str(child): sanitize(item, str(child)) for child, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


# ============================================================================ settings
MODES = ("online", "offline", "disabled")


def effective_mode(wb: Any) -> str:
    """``run.wandb.mode``, or ``$WANDB_MODE`` when it is set."""
    mode = os.environ.get("WANDB_MODE", "").strip().lower() or str(wb.mode)
    if mode not in MODES:
        raise WandbError(f"WANDB_MODE={mode!r}; use one of {MODES}")
    return mode


@dataclass(frozen=True)
class WandbSettings:
    enabled: bool
    mode: str
    project: str
    entity: str
    base_url: str
    name: str
    group: str
    notes: str
    tags: tuple[str, ...]
    directory: Path

    @classmethod
    def from_config(cls, config: Any, output_dir: Path) -> "WandbSettings":
        output_dir = Path(output_dir)
        wb = config.run.wandb
        project = str(wb.project or "").strip() or str(config.model.wandb_project or "").strip()
        mode = effective_mode(wb)
        enabled = bool(wb.enabled) and mode != "disabled"
        if enabled and not project:
            raise WandbError("set run.wandb.project or model.wandb_project (e.g. edmd_flux2_klein_4b)")
        from ..platform import detect

        tags: list[str] = []
        eval_name = config.eval.name if config.eval is not None else None
        # the platform tag (ascend / cuda / cpu) tells apart numbers that differ by the kernels' rounding
        for tag in (f"{config.model.name}-{config.model.variant}", str(config.method.name), eval_name,
                    *tuple(wb.tags), detect()):
            text = str(tag or "").strip()
            if text and text not in tags:
                if len(text) > MAX_TAG_LENGTH:
                    raise WandbError(f"W&B tags are at most {MAX_TAG_LENGTH} characters: {text!r}")
                tags.append(text)
        return cls(enabled=enabled, mode=mode, project=project, entity=str(wb.entity or ""),
                   base_url=str(wb.base_url or ""), name=output_dir.name, group=str(wb.group or ""),
                   notes=str(wb.notes or ""), tags=tuple(tags), directory=output_dir / "wandb")


def config_payload(config: Any) -> dict[str, Any]:
    """``wandb.config``: the resolved config by section."""
    from ..config.core import to_mapping

    return {section: sanitize(to_mapping(getattr(config, section))) for section in ("model", "method", "eval", "run")
            if getattr(config, section) is not None}


# ============================================================================ credentials
DEFAULT_BASE_URL = "https://api.wandb.ai"


def server_url(base_url: str = "") -> str:
    """The server ``wandb.init`` (0.24) logs to: ``base_url`` (``run.wandb.base_url``), else ``$WANDB_BASE_URL``, else
    the ``base_url`` of wandb's settings files (``$WANDB_CONFIG_DIR/settings``, default ``~/.config/wandb/settings``,
    where ``wandb login --host`` saves it; ``./wandb/settings`` overrides it), else https://api.wandb.ai."""
    url = str(base_url or os.environ.get("WANDB_BASE_URL") or "")
    if not url:
        import configparser

        system = Path(os.environ.get("WANDB_CONFIG_DIR") or "~/.config/wandb").expanduser() / "settings"
        workspace = Path(".wandb" if Path(".wandb").exists() else "wandb") / "settings"
        for path in (system, workspace):
            parser = configparser.ConfigParser()
            try:
                parser.read(path)
                url = parser.get("default", "base_url", fallback=url)
            except (OSError, ValueError, configparser.Error):
                pass
    return (url or DEFAULT_BASE_URL).rstrip("/")


def has_credentials(base_url: str = "") -> bool:
    """Whether ``wandb.init`` (0.24) finds credentials without prompting for them: ``$WANDB_API_KEY`` or
    ``$WANDB_IDENTITY_TOKEN_FILE``, else a password for the server's host (``server_url``) in the netrc file ``wandb
    login`` writes (``$NETRC``, else ``~/.netrc`` or ``~/_netrc``)."""
    if os.environ.get("WANDB_API_KEY") or os.environ.get("WANDB_IDENTITY_TOKEN_FILE"):
        return True
    import netrc
    from urllib.parse import urlsplit

    host = urlsplit(server_url(base_url)).netloc
    names = [os.environ["NETRC"]] if os.environ.get("NETRC") else ["~/.netrc", "~/_netrc"]
    path = next((Path(name).expanduser() for name in names if Path(name).expanduser().exists()), None)
    if not host or path is None:
        return False
    try:
        entry = netrc.netrc(str(path)).authenticators(host)
    except (OSError, ValueError, netrc.NetrcParseError):
        return False
    return bool(entry and entry[2])


def offline_without_credentials(settings: WandbSettings) -> WandbSettings:
    """``settings`` in offline mode, with one warning, when they are online but wandb would find no credentials
    (``has_credentials``): ``wandb.init`` would ask for a key, and a training process cannot answer."""
    if not settings.enabled or settings.mode != "online" or has_credentials(settings.base_url):
        return settings
    print(f"warning: no W&B credentials for {server_url(settings.base_url)} (set WANDB_API_KEY or run `wandb login`), "
          f"so W&B runs offline instead of online: training logs stay in {settings.directory} (`wandb sync` uploads "
          "them later) and evaluations are not published", flush=True)
    return replace(settings, mode="offline")


# ============================================================================ run state
def read_state(output_dir: Path) -> dict[str, Any] | None:
    path = Path(output_dir) / RUN_STATE
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _write_state(output_dir: Path, state: Mapping[str, Any]) -> None:
    from ..evaluation.protocol import write_json

    write_json(Path(output_dir) / RUN_STATE, dict(state))
    (Path(output_dir) / RUN_ID_FILE).write_text(str(state["id"]) + "\n", encoding="utf-8")


def _apply_environment(settings: WandbSettings) -> None:
    if settings.base_url:
        os.environ["WANDB_BASE_URL"] = settings.base_url
    os.environ.setdefault("WANDB__SERVICE_WAIT", "300")


def init_kwargs(settings: WandbSettings, run_id: str, *, creating: bool, wandb_module: Any,
                payload: Mapping[str, Any] | None = None, job_type: str = "train") -> dict[str, Any]:
    """Keyword arguments of ``wandb.init``. Identity arguments are passed on creation only."""
    kwargs: dict[str, Any] = {
        "project": settings.project, "id": run_id, "mode": settings.mode, "dir": str(settings.directory),
        "settings": wandb_module.Settings(console="off", save_code=False),
    }
    if settings.entity:
        kwargs["entity"] = settings.entity
    if creating:
        # "allow": an interrupted creation (run made, state not yet marked) re-attaches instead of duplicating.
        kwargs.update(resume="allow", name=settings.name, tags=list(settings.tags), notes=settings.notes or None,
                      group=settings.group or None, job_type=job_type, config=dict(payload or {}))
    else:
        kwargs["resume"] = "must"
    return kwargs


def open_run(config: Any, output_dir: Path, *, job_type: str = "train", wandb_module: Any = None) -> Any:
    """Create the run (first writer) or resume it by id (every later writer). None when W&B is off; offline when it is
    online without credentials (``offline_without_credentials``)."""
    output_dir = Path(output_dir)
    settings = WandbSettings.from_config(config, output_dir)
    if not settings.enabled:
        return None
    settings = offline_without_credentials(settings)
    if wandb_module is None:
        import wandb as wandb_module
    _apply_environment(settings)
    state = read_state(output_dir)
    if state is not None and (state.get("project") != settings.project or state.get("entity") != settings.entity):
        raise WandbError(f"{output_dir / RUN_STATE} belongs to {state.get('entity')}/{state.get('project')}")
    creating = not (state and state.get("created"))
    if creating and settings.name != config.run.name:
        print(f"note: the W&B run is named after the output directory, {settings.name!r} (run.name is "
              f"{config.run.name!r})", flush=True)
    if state is None:
        state = {"id": uuid.uuid4().hex[:16], "project": settings.project, "entity": settings.entity,
                 "base_url": settings.base_url, "created": False}
        _write_state(output_dir, state)   # the id exists before the run does: a retry reuses it
    settings.directory.mkdir(parents=True, exist_ok=True)
    payload = config_payload(config) if creating else None
    run = wandb_module.init(**init_kwargs(settings, state["id"], creating=creating, wandb_module=wandb_module,
                                          payload=payload, job_type=job_type))
    if creating:
        _write_state(output_dir, dict(state, created=True,
                                      created_at=datetime.now(timezone.utc).isoformat(timespec="seconds")))
    return run


# ============================================================================ training tracker
class Tracker:
    """Rank 0's W&B writer for training scalars (a no-op elsewhere or when W&B is off)."""

    def __init__(self, run: Any = None) -> None:
        self.run = run
        #: steps below this are already in the W&B history (a resume from an earlier checkpoint) and are not re-logged
        self.history_end = 0

    @property
    def enabled(self) -> bool:
        return self.run is not None

    def log(self, metrics: Mapping[str, Any], step: int) -> None:
        if self.run is not None and int(step) >= self.history_end:
            self.run.log(dict(metrics), step=int(step))

    def log_pending(self, metrics: Mapping[str, Any], step: int) -> None:
        """Merge ``metrics`` into the row the next ``log`` at ``step`` commits (``commit=False``), so the training
        process stays the only history writer."""
        if self.run is not None and int(step) >= self.history_end:
            self.run.log(dict(metrics), step=int(step), commit=False)

    def summary(self, values: Mapping[str, Any]) -> None:
        if self.run is not None:
            for key, value in values.items():
                self.run.summary[key] = value

    def check_resume(self, checkpoint_step: int) -> None:
        """A run interrupted between checkpoints has W&B rows past its newest checkpoint: the retrained steps are not
        logged again (W&B would drop them anyway); logging continues at W&B's next step."""
        if self.run is None:
            return
        next_step = int(getattr(self.run, "step", 0) or 0)
        if next_step > int(checkpoint_step) + 1:
            self.history_end = next_step
            print(f"W&B history already holds steps up to {next_step - 1}; steps {int(checkpoint_step) + 1}-"
                  f"{next_step - 1} are retrained from checkpoint {checkpoint_step} without logging them again",
                  flush=True)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()
            self.run = None


def open_training_tracker(config: Any, output_dir: Path, *, is_main: bool, wandb_module: Any = None) -> Tracker:
    """What the training loop calls at every start: rank 0 creates or resumes the run."""
    if not is_main:
        return Tracker(None)
    return Tracker(open_run(config, output_dir, wandb_module=wandb_module))


# ============================================================================ evaluation publisher
LANES = {"raw": "", "ema": "-ema"}


def reported_metrics(spec: Any) -> tuple[str, ...]:
    """The metrics an eval config scores: ``metrics``, then ``extra_metrics``."""
    return tuple(spec.metrics) + tuple(spec.extra_metrics)


def load_results(config: Any, output_dir: Path, lane: str = "raw") -> dict[int, tuple[Path, dict]]:
    """``{step: (directory, result)}`` of the periodic evaluation's complete results in ``lane`` (``raw``:
    ``<run>/eval/<eval name>/checkpoint-<N>``, ``ema``: ``checkpoint-<N>-ema``); a result is complete once every
    metric of the eval config is scored."""
    from ..evaluation.scoring import missing_metrics

    root = Path(output_dir) / "eval" / config.eval.name
    pattern = re.compile(r"checkpoint-(\d+)" + re.escape(LANES[lane]))
    metrics = reported_metrics(config.eval)
    results = {}
    for path in root.glob("checkpoint-*/result.json"):
        match = pattern.fullmatch(path.parent.name)
        if match and not missing_metrics(path.parent, metrics):
            step = int(match.group(1))
            results[step] = (path.parent, dict(json.loads(path.read_text(encoding="utf-8")), step=step))
    return dict(sorted(results.items()))


def summary_update(results: Mapping[int, tuple[Path, dict]], *, metrics: Sequence[str], eval_interval: int,
                   max_step: int, eval_name: str) -> dict[str, Any]:
    """Every point's curve keys plus the run-table scalars of the latest checkpoint."""
    from .media import LATEST_STEP_KEY, build_curve_summary

    staged: dict[str, Any] = {LATEST_STEP_KEY: -1}
    update: dict[str, Any] = {}
    for step in sorted(results):
        delta = build_curve_summary(results[step][1], staged, metrics=metrics, eval_interval=eval_interval,
                                    max_step=max_step)
        staged.update(delta)
        update.update(delta)
    latest = results[max(results)][1]
    update.update({f"eval/{metric}": float(latest["results"][metric]["mean"]) for metric in metrics})
    update.update({"eval/checkpoint_step": max(results), "eval/num_images": int(latest["num_images"]),
                   "eval/name": eval_name, "eval/published_steps": sorted(results), HISTORY_NOTE_KEY: HISTORY_NOTE})
    return update


def raw_vs_ema_rows(raw: Mapping[int, tuple[Path, dict]], ema: Mapping[int, tuple[Path, dict]],
                    metrics: Sequence[str]) -> tuple[list[str], list[list]]:
    """``result/raw_vs_ema``: one row per evaluated checkpoint, ``step`` then ``<metric>_raw`` / ``<metric>_ema`` for
    every metric (None while that side is missing)."""
    columns = ["step"] + [f"{metric}_{kind}" for metric in metrics for kind in ("raw", "ema")]
    rows = []
    for step in sorted(set(raw) | set(ema)):
        row: list = [int(step)]
        for metric in metrics:
            for side in (raw, ema):
                entry = side.get(step)
                row.append(None if entry is None else float(entry[1]["results"][metric]["mean"]))
        rows.append(row)
    return columns, rows


def raw_vs_ema_history_table(config: Any, output_dir: Path, *, wandb_module: Any = None) -> Any:
    """The ``result/raw_vs_ema`` table for the training process's resume row (None without ``run.eval_ema`` or a
    result). W&B shows a table panel only for a history key; the panel reads the summary copy the publisher updates."""
    if config.eval is None or not config.run.eval_ema:
        return None
    columns, rows = raw_vs_ema_rows(load_results(config, output_dir, "raw"), load_results(config, output_dir, "ema"),
                                    reported_metrics(config.eval))
    if not rows:
        return None
    if wandb_module is None:
        import wandb as wandb_module
    return wandb_module.Table(columns=columns, data=rows)


def stale_curve_keys(keys: Sequence[str], present_steps: Sequence[int], metrics: Sequence[str]) -> list[str]:
    pattern = re.compile(r"^eval/curve_(\d+)_(" + "|".join(map(re.escape, metrics)) + r")$")
    stale = []
    for key in keys:
        match = pattern.match(str(key))
        if match and int(match.group(1)) not in set(present_steps):
            stale.append(str(key))
    return sorted(stale)


def retired_metric_keys(keys: Sequence[str], metrics: Sequence[str]) -> list[str]:
    """Summary keys of metrics the run no longer reports (``eval/<m>``, ``eval/curve_<step>_<m>``,
    ``eval/curve_latest_<m>``): publishing removes them, so a dropped metric leaves no stale scores behind."""
    from ..evaluation.protocol import METRICS as ALL_METRICS

    retired = [metric for metric in ALL_METRICS if metric not in set(metrics)]
    if not retired:
        return []
    names = "|".join(map(re.escape, retired))
    pattern = re.compile(r"^eval/(?:curve_(?:\d+|latest)_)?(" + names + r")$")
    return sorted(str(key) for key in keys if pattern.match(str(key)))


def publish_evaluations(output_dir: Path, *, wandb_module: Any = None) -> dict[str, Any]:
    """Resume (or create) the run's W&B run and rewrite every evaluation point from disk -- the curves' summary keys,
    the panels, the gallery and, with ``run.eval_ema``, the raw-vs-EMA table -- then finish. Curve keys of points
    without a complete result are removed, so W&B equals the ``result.json`` files."""
    from ..config.load import compose
    from .media import configure_native_backfill_panels, paged_history_chart, publish_summary_chart

    output_dir = Path(output_dir)
    config, _ = compose(output_dir / "config.yaml")
    if config.eval is None or config.run.eval_every <= 0:
        return {"published": False, "reason": "the run has no periodic evaluation (eval config, run.eval_every)"}
    raw = load_results(config, output_dir, "raw")
    if not raw:
        return {"published": False, "reason": f"no complete result under {output_dir / 'eval' / config.eval.name}"}
    settings = WandbSettings.from_config(config, output_dir)
    if not settings.enabled:
        return {"published": False, "reason": "W&B disabled for this run"}
    settings = offline_without_credentials(settings)
    if settings.mode != "online":
        return {"published": False, "reason": f"W&B mode {settings.mode}: evaluations are published online only"}
    if wandb_module is None:
        import wandb as wandb_module
    run = open_run(config, output_dir, wandb_module=wandb_module)
    report: dict[str, Any] = {"published": True, "run_id": getattr(run, "id", None), "steps": sorted(raw)}
    try:
        metrics = reported_metrics(config.eval)
        eval_interval, max_step = int(config.run.eval_every), int(config.method.updates.max_steps)
        update = summary_update(raw, metrics=metrics, eval_interval=eval_interval, max_step=max_step,
                                eval_name=config.eval.name)
        keys = list(run.summary.keys())
        stale = stale_curve_keys(keys, list(raw), metrics) + retired_metric_keys(keys, metrics)
        for key in stale:
            del run.summary[key]
        report["removed"] = stale
        run.summary.update(update)
        configure_native_backfill_panels(run, metrics, eval_interval=eval_interval, max_step=max_step)
        if config.eval.inline_chart and config.eval.preview_count > 0:
            try:
                chart = paged_history_chart(run, output_dir, int(config.eval.preview_count), config.eval.inline_chart,
                                            points={step: directory for step, (directory, _) in raw.items()},
                                            wandb_module=wandb_module)
                publish_summary_chart(run, GALLERY_KEY, chart)
                report["gallery"] = len(raw)
            except Exception as error:  # noqa: BLE001 - the preview must not block the scores
                report["gallery_error"] = f"{type(error).__name__}: {error}"
                print(f"W&B gallery not published ({report['gallery_error']}); the scores are.", flush=True)
        if config.run.eval_ema:
            columns, rows = raw_vs_ema_rows(raw, load_results(config, output_dir, "ema"), metrics)
            run.summary[RESULT_TABLE_KEY] = wandb_module.Table(columns=columns, data=rows)
            report["raw_vs_ema_rows"] = len(rows)
    finally:
        run.finish()
    return report


__all__ = ["DEFAULT_BASE_URL", "GALLERY_KEY", "HISTORY_NOTE", "HISTORY_NOTE_KEY", "LANES", "RESULT_TABLE_KEY",
           "RUN_ID_FILE", "RUN_STATE", "Tracker", "WandbError", "WandbSettings", "config_payload", "effective_mode",
           "has_credentials", "init_kwargs", "load_results", "offline_without_credentials", "open_run",
           "open_training_tracker", "publish_evaluations", "raw_vs_ema_history_table", "raw_vs_ema_rows",
           "read_state", "reported_metrics", "retired_metric_keys", "sanitize", "server_url", "stale_curve_keys",
           "summary_update"]

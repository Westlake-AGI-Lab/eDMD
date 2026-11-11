"""Flat composition of a run: model + method + run (+ eval), then strict validation.

A run file names its components and carries only run-level settings plus dotted overrides::

    model: flux2_klein_4b            # configs/models/<name>.yaml, or a path
    method: edmd                     # configs/methods/<name>.yaml
    eval: drawbench1k                # configs/eval/<name>.yaml (optional: the periodic evaluation's config)
    run: {name: ..., output_root: ..., seed: ..., data: {...}, wandb: {...}}
    overrides: {method.updates.student.lr: 6.0e-6}

There is no inheritance chain: each component has a fixed role, every override path must name an existing parent
(and schema validation rejects unknown leaf names). A calibrated query names its teacher's table
(``calibration: <name>``: ``configs/calibration/<name>.json``), whose numbers composing reads in. Training writes the
resolved mapping to ``<run dir>/config.yaml``, which composes again as it stands.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .core import ConfigError, expand_path, from_mapping, load_yaml, loads_yaml, set_dotted
from .schema import Config, ModelSpec

REPO = Path(__file__).resolve().parents[2]
SECTIONS = ("model", "method", "eval", "run")
RUN_FILE_KEYS = {"model", "method", "eval", "run", "overrides", "description"}
CALIBRATION_KEYS = ("landing", "levels", "d", "cfg")


def component_path(kind: str, ref: str) -> Path:
    """A component name (``configs/<kind>/<name>.yaml``) or a ``.yaml`` path: from the working directory when it is
    there, else from the repository root."""
    candidate = Path(ref)
    if candidate.suffix in (".yaml", ".yml"):
        if candidate.is_absolute() or candidate.is_file():
            return candidate.absolute()
        return REPO / candidate
    return REPO / "configs" / kind / f"{ref}.yaml"


def _load_component(kind: str, ref: str) -> dict:
    path = component_path(kind, ref)
    if not path.is_file():
        raise ConfigError(f"{kind} component {ref!r} not found at {path}")
    value = load_yaml(path)
    if not isinstance(value, dict):
        raise ConfigError(f"{path} must hold a mapping")
    return value


def calibration_table(ref: str, model: Mapping[str, Any]) -> dict:
    """``calibration: <name>``: the numbers of ``configs/calibration/<name>.json`` (or of a ``.json`` path), which must
    have been measured for the run's teacher."""
    path = Path(ref).expanduser() if ref.endswith(".json") else REPO / "configs" / "calibration" / f"{ref}.json"
    if not path.is_absolute() and not path.is_file():
        path = REPO / path
    if not path.is_file():
        raise ConfigError(f"query.calibration {ref!r}: no table at {path}")
    table = json.loads(path.read_text())
    measured, teacher = (table.get("model"), str(table.get("variant"))), (model.get("name"), str(model.get("variant")))
    if measured != teacher:
        raise ConfigError(f"query.calibration {ref!r} was measured for {'/'.join(measured)}, the run's teacher is "
                          f"{'/'.join(teacher)}")
    return {key: table[key] for key in CALIBRATION_KEYS}


def resolve_calibration(resolved: dict) -> None:
    """A ``query.calibration`` that names a table gets the table's numbers (a run directory's ``config.yaml`` holds
    them already)."""

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if isinstance(node.get("calibration"), str):
                node["calibration"] = calibration_table(node["calibration"], resolved.get("model") or {})
            for child in node.values():
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)

    walk(resolved.get("method"))


def apply_overrides(resolved: dict, overrides: Mapping[str, Any] | None, *, where: str) -> None:
    for dotted, value in (overrides or {}).items():
        if not isinstance(dotted, str) or dotted.split(".")[0] not in SECTIONS:
            raise ConfigError(f"{where}: override {dotted!r} must start with one of {SECTIONS}")
        set_dotted(resolved, dotted, copy.deepcopy(value), where=f"{where}:{dotted}", create_leaf=True)


def parse_assignments(items: Sequence[str] | None) -> dict[str, Any]:
    """``--set`` values: ``["method.updates.max_steps=500", "run.wandb.mode=offline"]`` -> ``{dotted: value}``, each
    value read as YAML (``500``, ``false``, ``[1, 2]``, ``null``)."""
    assignments: dict[str, Any] = {}
    for item in items or ():
        key, separator, text = str(item).partition("=")
        if not separator or not key.strip():
            raise ConfigError(f"--set {item!r}: expected key=value, e.g. method.updates.max_steps=500")
        assignments[key.strip()] = loads_yaml(text)
    return assignments


def compose(run: Mapping[str, Any] | str | Path, overrides: Mapping[str, Any] | None = None) -> tuple[Config, dict]:
    """Return the validated ``Config`` and the resolved plain mapping (what training writes to ``config.yaml``).
    ``overrides`` (``--set``) are applied last, after the run file's own.

    An already resolved config -- ``model`` and ``method`` as mappings, as a run directory's ``config.yaml`` holds
    them -- is validated as it stands.
    """
    raw = load_yaml(run) if isinstance(run, (str, Path)) else dict(run)
    if not isinstance(raw, dict):
        raise ConfigError("a run file must hold a mapping")
    if isinstance(raw.get("model"), Mapping) and isinstance(raw.get("method"), Mapping):
        unknown = sorted(set(raw) - set(SECTIONS))
        if unknown:
            raise ConfigError(f"a resolved config has the sections {SECTIONS}, not {unknown}")
        resolved = copy.deepcopy(raw)
    else:
        mixed = [key for key in ("model", "method", "eval") if isinstance(raw.get(key), Mapping)]
        if mixed:
            raise ConfigError(f"run file: {mixed} are mappings; a run file names its components (a resolved config "
                              "has model and method as mappings)")
        unknown = sorted(set(raw) - RUN_FILE_KEYS)
        if unknown:
            raise ConfigError(f"run file: unknown key(s) {unknown}; allowed: {sorted(RUN_FILE_KEYS)}")
        for key in ("model", "method", "run"):
            if key not in raw:
                raise ConfigError(f"run file: missing {key!r}")
        resolved = {"model": _load_component("models", raw["model"]),
                    "method": _load_component("methods", raw["method"]),
                    "run": copy.deepcopy(raw["run"])}
        if raw.get("eval"):
            resolved["eval"] = _load_component("eval", raw["eval"])
        apply_overrides(resolved, raw.get("overrides"), where="run overrides")
    apply_overrides(resolved, overrides, where="--set")
    resolve_calibration(resolved)
    return from_mapping(Config, resolved, "config"), resolved


def checkpoint_model(record: Mapping[str, Any], *, config: str | Path | None = None, run_dir: Path | None = None,
                     model_path: str | Path | None = None) -> ModelSpec:
    """The model a checkpoint with inference record ``record`` renders with: ``config`` (a run file or a run's
    ``config.yaml``), else ``run_dir/config.yaml``, else -- released weights -- the ``configs/models`` file of the
    record's adapter and variant with the record's adapter options and resolution. ``model_path`` replaces
    ``model.path``."""
    if config is not None:
        model = compose(config)[1]["model"]
    elif run_dir is not None and (Path(run_dir) / "config.yaml").is_file():
        model = load_yaml(Path(run_dir) / "config.yaml")["model"]       # only the model section is read
    else:
        wanted = (record["adapter"], str(record["variant"]))
        found = [load_yaml(path) for path in sorted((REPO / "configs" / "models").glob("*.yaml"))]
        found = [value for value in found if isinstance(value, dict)
                 and (value.get("name"), str(value.get("variant"))) == wanted]
        if not found:
            raise ConfigError(f"no configs/models file for {wanted[0]}/{wanted[1]}; pass --config")
        model = dict(found[0], options=dict(record.get("adapter_options") or found[0].get("options") or {}))
        if record.get("resolution"):
            model["resolution"] = int(record["resolution"])
    model = copy.deepcopy(dict(model))
    if model_path is not None:
        model["path"] = str(Path(model_path).expanduser().absolute())
    elif "$" in os.path.expandvars(str(model["path"])):
        raise ConfigError(f"model.path {model['path']!r}: set EDMD_MODEL_ROOT or pass --model-path")
    spec = from_mapping(ModelSpec, model, "model")
    if (spec.name, str(spec.variant)) != (record["adapter"], str(record["variant"])):
        raise ConfigError(f"the checkpoint is {record['adapter']}/{record['variant']}, the config "
                          f"{spec.name}/{spec.variant}")
    return spec


def run_output_dir(config: Config) -> Path:
    """``run.output_root / run.name`` with ``~`` and environment variables expanded; a relative root is taken from the
    repository root."""
    root = expand_path(config.run.output_root)
    return (root if root.is_absolute() else REPO / root) / config.run.name


__all__ = ["REPO", "apply_overrides", "calibration_table", "checkpoint_model", "component_path", "compose",
           "parse_assignments", "resolve_calibration", "run_output_dir"]

"""Strict, frozen configuration dataclasses and a YAML loader that reads ``1e-6`` as a float.

Rules:

* **unknown keys raise**, at every level: a misspelled key must not silently run the default;
* **no silent defaults for algorithm keys**: a dataclass field without a default is required, so a missing key is an
  error rather than a code default;
* **types are checked** (bool is not an int, ints are accepted for floats, ``None`` only for Optional);
* **scientific notation**: PyYAML follows YAML 1.1 and reads ``1e-06`` as a *string*; our loader adds the YAML 1.2
  float pattern.
"""

from __future__ import annotations

import dataclasses
import math
import os
import re
import types
import typing
from pathlib import Path
from typing import Any, Mapping, get_args, get_origin

import yaml


class ConfigError(ValueError):
    pass


class _Loader(yaml.SafeLoader):
    pass


_FLOAT_12 = re.compile(
    r"""^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?\.(?:inf|Inf|INF)
    |\.(?:nan|NaN|NAN))$""",
    re.X,
)
_Loader.add_implicit_resolver("tag:yaml.org,2002:float", _FLOAT_12, list("-+0123456789."))


def load_yaml(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as handle:
        return yaml.load(handle, Loader=_Loader)  # noqa: S506 - SafeLoader subclass


def loads_yaml(text: str) -> Any:
    return yaml.load(text, Loader=_Loader)  # noqa: S506


def dump_yaml(value: Any) -> str:
    return yaml.safe_dump(value, sort_keys=False, allow_unicode=True)


def _type_name(tp: Any) -> str:
    return getattr(tp, "__name__", str(tp))


def _coerce(value: Any, tp: Any, where: str) -> Any:
    origin = get_origin(tp)
    if tp is Any:
        return value
    if origin is typing.Union or (hasattr(types, "UnionType") and isinstance(tp, types.UnionType)):
        args = get_args(tp)
        if value is None:
            if type(None) in args:
                return None
            raise ConfigError(f"{where}: null is not allowed")
        errors = []
        for arg in args:
            if arg is type(None):
                continue
            try:
                return _coerce(value, arg, where)
            except ConfigError as exc:
                errors.append(str(exc))
        raise ConfigError(f"{where}: {value!r} matches none of {[_type_name(a) for a in args]} ({'; '.join(errors)})")
    if origin is typing.Literal:
        allowed = get_args(tp)
        if value not in allowed:
            raise ConfigError(f"{where}: {value!r} is not one of {list(allowed)}")
        return value
    if value is None:
        raise ConfigError(f"{where}: null is not allowed")
    if dataclasses.is_dataclass(tp):
        return from_mapping(tp, value, where)
    if origin in (tuple, list):
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{where}: expected a list, got {type(value).__name__}")
        args = get_args(tp)
        if origin is tuple and len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce(item, args[0], f"{where}[{i}]") for i, item in enumerate(value))
        if origin is tuple and args:
            if len(args) != len(value):
                raise ConfigError(f"{where}: expected {len(args)} items, got {len(value)}")
            return tuple(_coerce(item, arg, f"{where}[{i}]") for i, (item, arg) in enumerate(zip(value, args)))
        inner = args[0] if args else Any
        items = [_coerce(item, inner, f"{where}[{i}]") for i, item in enumerate(value)]
        return tuple(items) if origin is tuple else items
    if origin in (dict, Mapping) or tp in (dict, Mapping):
        if not isinstance(value, Mapping):
            raise ConfigError(f"{where}: expected a mapping")
        args = get_args(tp)
        vt = args[1] if len(args) == 2 else Any
        return {str(k): _coerce(v, vt, f"{where}.{k}") for k, v in value.items()}
    if tp is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{where}: expected true/false, got {value!r}")
        return value
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{where}: expected an integer, got {value!r}")
        return value
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{where}: expected a number, got {value!r}")
        value = float(value)
        if math.isnan(value):
            raise ConfigError(f"{where}: NaN is not allowed")
        return value
    if tp is str:
        if not isinstance(value, str):
            raise ConfigError(f"{where}: expected a string, got {value!r}")
        return value
    raise ConfigError(f"{where}: unsupported field type {tp!r}")


def from_mapping(cls: type, mapping: Any, where: str = "config") -> Any:
    """Build dataclass ``cls`` from ``mapping``: unknown keys and missing required keys raise."""
    if not isinstance(mapping, Mapping):
        raise ConfigError(f"{where}: expected a mapping, got {type(mapping).__name__}")
    hints = typing.get_type_hints(cls)
    fields = {field.name: field for field in dataclasses.fields(cls)}
    unknown = sorted(set(mapping) - set(fields))
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) {unknown}; allowed: {sorted(fields)}")
    values = {}
    for name, field in fields.items():
        if name in mapping:
            values[name] = _coerce(mapping[name], hints[name], f"{where}.{name}")
        elif field.default is dataclasses.MISSING and field.default_factory is dataclasses.MISSING:  # type: ignore[misc]
            raise ConfigError(f"{where}: missing required key {name!r}")
    try:
        return cls(**values)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{where}: {exc}") from exc


def to_mapping(value: Any) -> Any:
    """Frozen dataclass -> plain YAML-able mapping (tuples become lists)."""
    if dataclasses.is_dataclass(value):
        return {field.name: to_mapping(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, (list, tuple)):
        return [to_mapping(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): to_mapping(item) for key, item in value.items()}
    return value


def set_dotted(mapping: dict, dotted: str, value: Any, *, where: str = "override", create_leaf: bool = False) -> None:
    """Set ``a.b.c`` in a nested plain mapping. Every parent must exist; the leaf too, unless ``create_leaf`` (a field
    the component left at its default; schema validation still rejects names that are not fields)."""
    parts = dotted.split(".")
    node = mapping
    for i, part in enumerate(parts[:-1]):
        if not isinstance(node, dict) or part not in node:
            raise ConfigError(f"{where}: {'.'.join(parts[:i + 1])} does not exist")
        node = node[part]
    if not isinstance(node, dict) or (parts[-1] not in node and not create_leaf):
        raise ConfigError(f"{where}: {dotted} does not exist")
    node[parts[-1]] = value


def expand_path(value: str | Path) -> Path:
    """A configured path with ``~`` and ``$VAR`` / ``${VAR}`` expanded; a relative one is taken from the repository root
    (child processes run in their own working directory). An unset variable is an error rather than a literal
    ``$VAR`` directory."""
    text = os.path.expanduser(os.path.expandvars(str(value)))
    if "$" in text:
        raise ConfigError(f"{value!r}: environment variable not set (expanded to {text!r})")
    path = Path(text)
    if not text or path.is_absolute():
        return path
    return Path(__file__).resolve().parents[2] / path


__all__ = ["ConfigError", "dump_yaml", "expand_path", "from_mapping", "load_yaml", "loads_yaml", "set_dotted",
           "to_mapping"]

"""The accelerator a process computes on: an Ascend NPU (torch_npu, HCCL), a CUDA GPU (NCCL) or the CPU (gloo; CPU-only
checks). Everything vendor-specific goes through here, so no other module names ``torch.npu`` or ``torch.cuda``.

Selection (first match): ``backend`` / ``EDMD_BACKEND`` (``hccl`` | ``nccl`` | ``gloo``); ``EDMD_ACCELERATOR``
(``npu`` | ``cuda`` | ``cpu``); ``EDMD_PLATFORM`` (``ascend`` | ``cuda`` | ``cpu``, ``edmd.platform``'s override); an
NPU if ``torch_npu`` sees one; a GPU if ``torch.cuda`` does; the CPU. A requested accelerator that is absent raises.
"""

from __future__ import annotations

import os
from typing import Any

import torch

BACKENDS = {"npu": "hccl", "cuda": "nccl", "cpu": "gloo"}
BACKEND_DEVICES = {backend: kind for kind, backend in BACKENDS.items()}
#: ``EDMD_PLATFORM`` value -> device type
PLATFORM_DEVICES = {"ascend": "npu", "cuda": "cuda", "cpu": "cpu"}
_AVAILABLE: dict[str, bool] = {}


def npu_available() -> bool:
    if "npu" not in _AVAILABLE:
        try:
            import torch_npu  # noqa: F401 - registers the NPU backend

            _AVAILABLE["npu"] = bool(torch.npu.is_available())
        except ImportError:
            _AVAILABLE["npu"] = False
    return _AVAILABLE["npu"]


def cuda_available() -> bool:
    if "cuda" not in _AVAILABLE:
        _AVAILABLE["cuda"] = bool(torch.cuda.is_available())
    return _AVAILABLE["cuda"]


def accelerator_type(backend: str | None = None) -> str:
    """``npu`` | ``cuda`` | ``cpu`` for this process (module docstring)."""
    backend = backend or os.environ.get("EDMD_BACKEND") or None
    if backend is not None:
        if backend not in BACKEND_DEVICES:
            raise ValueError(f"unknown backend {backend!r}; one of {sorted(BACKEND_DEVICES)}")
        kind = BACKEND_DEVICES[backend]
    else:
        kind = (os.environ.get("EDMD_ACCELERATOR") or "").strip().lower() or None
        if kind is not None and kind not in BACKENDS:
            raise ValueError(f"EDMD_ACCELERATOR={kind!r}; use npu, cuda or cpu")
        platform = (os.environ.get("EDMD_PLATFORM") or "").strip().lower()
        if kind is None and platform:
            if platform not in PLATFORM_DEVICES:
                raise ValueError(f"EDMD_PLATFORM={platform!r}; use ascend, cuda or cpu")
            kind = PLATFORM_DEVICES[platform]
    if kind is None:
        return "npu" if npu_available() else "cuda" if cuda_available() else "cpu"
    if kind == "npu" and not npu_available():
        raise RuntimeError("an NPU was requested (backend hccl / EDMD_ACCELERATOR=npu) but torch_npu sees none")
    if kind == "cuda" and not cuda_available():
        raise RuntimeError("a GPU was requested (backend nccl / EDMD_ACCELERATOR=cuda) but torch.cuda sees none")
    return kind


def module(kind: str | torch.device) -> Any:
    """``torch.npu`` / ``torch.cuda`` for an accelerator (its memory and RNG API), ``None`` for the CPU."""
    kind = torch.device(kind).type if isinstance(kind, torch.device) else str(kind).split(":")[0]
    if kind == "npu":
        import torch_npu  # noqa: F401

        return torch.npu
    if kind == "cuda":
        return torch.cuda
    return None


def set_device(device: torch.device | str) -> None:
    device = torch.device(device)
    accelerator = module(device)
    if accelerator is not None:
        accelerator.set_device(device)


def manual_seed_all(seed: int, device: torch.device) -> None:
    """Seed every generator of ``device``'s accelerator; a CPU process touches none, so it never initializes a card."""
    accelerator = module(device)
    if accelerator is not None:
        accelerator.manual_seed_all(int(seed))


#: key of the device generator's state in ``runtime_state_rank-R.pt``
RNG_KEYS = {"npu": "npu_rng", "cuda": "cuda_rng"}


def rng_state(device: torch.device) -> dict[str, Any]:
    """The device generator's state keyed as :data:`RNG_KEYS` (empty for the CPU)."""
    accelerator = module(device)
    if accelerator is None:
        return {}
    return {RNG_KEYS[device.type]: accelerator.get_rng_state(device)}


def restore_rng_state(state: dict[str, Any], device: torch.device) -> None:
    accelerator = module(device)
    if accelerator is None:
        return
    key = RNG_KEYS[device.type]
    if state.get(key) is None:
        raise ValueError(f"the continuation state has no {key} for this {device.type} device")
    accelerator.set_rng_state(state[key], device)


def describe(device: torch.device) -> dict[str, Any]:
    """What ``runtime.json`` and generation records say about the device."""
    record: dict[str, Any] = {"type": device.type, "torch": torch.__version__}
    if device.type == "cuda":
        record.update(cuda=torch.version.cuda, name=torch.cuda.get_device_name(device),
                      cudnn=torch.backends.cudnn.version())
    elif device.type == "npu":
        try:
            import torch_npu

            record.update(torch_npu=getattr(torch_npu, "__version__", None), name=torch.npu.get_device_name(device))
        except Exception:  # noqa: BLE001 - descriptive only
            pass
    return record


__all__ = ["BACKENDS", "BACKEND_DEVICES", "RNG_KEYS", "accelerator_type", "cuda_available", "describe",
           "manual_seed_all", "module", "npu_available", "restore_rng_state", "rng_state", "set_device"]

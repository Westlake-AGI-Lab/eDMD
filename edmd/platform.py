"""Which kind of machine this is -- Ascend (NPUs), CUDA (NVIDIA GPUs) or CPU -- and what follows from it: card count,
device names and the collective backend. A config says *what* runs; this module says *where*.

Detection, first match wins: ``EDMD_PLATFORM`` (``ascend`` | ``cuda`` | ``cpu``); an Ascend driver or
``/dev/davinci*``; an NVIDIA driver, ``/dev/nvidia<N>`` or a GPU listed by ``nvidia-smi -L``; otherwise ``cpu``.

Standard library only: it is imported before torch.
"""

from __future__ import annotations

import functools
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

PLATFORM_ENV = "EDMD_PLATFORM"


@dataclass(frozen=True)
class Platform:
    name: str                              # ascend | cuda | cpu
    device_type: str                       # npu | cuda | cpu (torch device type)
    backend: str                           # hccl | nccl | gloo
    default_cards: int                     # cards per machine when they cannot be counted
    visible_devices_env: str = ""          # the runtime's device mask variable

    def cards(self) -> int:
        """Accelerators this process may use: the visible-devices mask if set (an empty mask hides every card), else the
        device nodes, else the default."""
        if self.device_type == "cpu":
            return 1
        mask = os.environ.get(self.visible_devices_env)
        if mask is not None:
            return len([item for item in mask.split(",") if item.strip() != ""])
        pattern = {"npu": r"davinci(\d+)", "cuda": r"nvidia(\d+)"}[self.device_type]
        try:
            count = sum(1 for entry in os.listdir("/dev") if re.fullmatch(pattern, entry))
        except OSError:
            count = 0
        return count or self.default_cards

    def device(self, index: int = 0) -> str:
        return "cpu" if self.device_type == "cpu" else f"{self.device_type}:{int(index)}"

    def devices(self) -> list[str]:
        """Every visible card (``cpu`` on a CPU machine)."""
        return [self.device(index) for index in range(self.cards())]


ASCEND = Platform(name="ascend", device_type="npu", backend="hccl", default_cards=16,
                  visible_devices_env="ASCEND_RT_VISIBLE_DEVICES")
CUDA = Platform(name="cuda", device_type="cuda", backend="nccl", default_cards=8, visible_devices_env="CUDA_VISIBLE_DEVICES")
CPU = Platform(name="cpu", device_type="cpu", backend="gloo", default_cards=1)


def detect() -> str:
    """``ascend`` | ``cuda`` | ``cpu`` for this machine (see the module docstring)."""
    forced = os.environ.get(PLATFORM_ENV, "").strip().lower()
    if forced:
        if forced not in ("ascend", "cuda", "cpu"):
            raise ValueError(f"{PLATFORM_ENV}={forced!r}; use ascend, cuda or cpu")
        return forced
    try:
        devices = os.listdir("/dev")
    except OSError:
        devices = []
    if Path("/usr/local/Ascend/driver").is_dir() or any(re.fullmatch(r"davinci\d+", name) for name in devices):
        return "ascend"
    if (Path("/proc/driver/nvidia/version").is_file() or any(re.fullmatch(r"nvidia\d+", name) for name in devices)
            or _nvidia_smi_lists_gpu()):
        return "cuda"
    return "cpu"


@functools.lru_cache(maxsize=1)
def _nvidia_smi_lists_gpu() -> bool:
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        listing = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return listing.lstrip().startswith("GPU ")


def by_name(name: str) -> Platform:
    platforms = {"ascend": ASCEND, "cuda": CUDA, "cpu": CPU}
    if name not in platforms:
        raise ValueError(f"unknown platform {name!r}")
    return platforms[name]


def current() -> Platform:
    return by_name(detect())


__all__ = ["ASCEND", "CPU", "CUDA", "PLATFORM_ENV", "Platform", "by_name", "current", "detect"]

"""Evaluation devices: device names, whole-batch sharding and one process per device.

* ``batch_shard_indices`` hands whole contiguous batches to shards round-robin, so a sample lands in the same batch
  whatever the device count, and the per-sample numbers do not depend on it.
* ``run_on_devices`` runs ``target(device, *args)`` once per device: in this process for one device, else in one
  spawned process per device; the first failure stops the others.
"""

from __future__ import annotations

import multiprocessing
import re
from collections.abc import Callable, Sequence
from multiprocessing.connection import wait


def normalize_device(value: str) -> str:
    value = str(value).strip()
    if value in {"cpu", "cpu:0"}:
        return "cpu"
    match = re.fullmatch(r"(npu|cuda)(?::([0-9]+))?", value)
    if not match:
        raise ValueError(f"invalid device {value!r}; use cpu, npu:N or cuda:N")
    return f"{match[1]}:{int(match[2] or 0)}"


def plan_devices(devices: Sequence[str] | None = None) -> list[str]:
    """``devices`` (distinct, one accelerator type), or every visible card of this machine."""
    if not devices:
        from ..platform import current

        return current().devices()
    planned = [normalize_device(value) for value in devices]
    if len(set(planned)) != len(planned):
        raise ValueError(f"duplicate devices in {list(devices)}")
    if len({device.split(":", 1)[0] for device in planned}) != 1:
        raise ValueError(f"the devices must be of one type, got {planned}")
    return planned


def batch_shard_indices(count: int, batch_size: int, shard_index: int, num_shards: int) -> list[int]:
    """Assign whole original contiguous batches round-robin, retaining the tail."""
    if batch_size < 1 or num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError(f"invalid shard {shard_index} of {num_shards} (batch size {batch_size})")
    indices = []
    for start in range(shard_index * batch_size, count, num_shards * batch_size):
        indices.extend(range(start, min(start + batch_size, count)))
    return indices


def run_on_devices(target: Callable[..., None], devices: Sequence[str], args: Sequence[tuple]) -> None:
    """``target(devices[r], *args[r])`` for every ``r``: in this process for one device, else in spawned processes
    (each loads its own models); a failure terminates the others and raises."""
    if len(devices) != len(args):
        raise ValueError("one argument tuple per device")
    if len(devices) == 1:
        target(devices[0], *args[0])
        return
    context = multiprocessing.get_context("spawn")
    processes = [context.Process(target=target, args=(device, *arguments), name=f"edmd-{device}")
                 for device, arguments in zip(devices, args)]
    for process in processes:
        process.start()
    try:
        running = list(processes)
        while running:
            wait([process.sentinel for process in running])
            for process in [process for process in running if process.exitcode is not None]:
                running.remove(process)
                if process.exitcode != 0:
                    raise RuntimeError(f"the process on {process.name[5:]} failed (exit code {process.exitcode}); "
                                       "its traceback is above")
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join()


__all__ = ["batch_shard_indices", "normalize_device", "plan_devices", "run_on_devices"]

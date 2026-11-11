"""Process group, FSDP2 sharding and collectives, on Ascend (HCCL), CUDA (NCCL) or CPU (gloo, CPU checks only).

The adapter names its FSDP units explicitly (an implicit ``_no_split_modules`` can silently leave blocks unsharded).
``MixedPrecisionPolicy(param bf16, reduce fp32, cast_forward_inputs=False)``: inputs are not cast, so each adapter
keeps its model's timestep precision. Nothing here names a vendor (``edmd.runtime.device`` picks the accelerator).
"""

from __future__ import annotations

import datetime
import os
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.distributed as dist
from torch import Tensor, nn


def init_distributed(timeout_minutes: int = 120, *, backend: str | None = None) -> tuple[int, int, torch.device]:
    """Initialize from torchrun's environment; return ``(rank, world_size, device)``. Started without torchrun
    (``python -m edmd.cli.train``), it is one process on one card. The backend follows
    ``edmd.runtime.device.accelerator_type``.
    """
    from . import device as device_lib

    if "RANK" not in os.environ:     # FSDP2 needs a process group even for a single process
        import socket

        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        os.environ.update(RANK="0", LOCAL_RANK="0", WORLD_SIZE="1", MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    kind = device_lib.accelerator_type(backend)
    if kind == "cpu":
        device = torch.device("cpu")
        chosen = "gloo"
        pin_cpu_defaults()
    else:
        device = torch.device(kind, local_rank)
        device_lib.set_device(device)
        chosen = device_lib.BACKENDS[kind]
    if not dist.is_initialized() and int(os.environ.get("WORLD_SIZE", "1")) >= 1 and "MASTER_ADDR" in os.environ:
        dist.init_process_group(backend=chosen, timeout=datetime.timedelta(minutes=timeout_minutes))
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    return rank, world, device


def pin_cpu_defaults() -> None:
    """Pin gradient-checkpoint recompute to the CPU, so a CPU process never initializes an NPU; torch_npu (when
    installed) is imported first, because importing it makes the NPU the default."""
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        pass
    try:
        from torch.utils.checkpoint import DefaultDeviceType

        DefaultDeviceType.set_device_type("cpu")
    except ImportError:   # pragma: no cover - older torch
        pass


def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def broadcast_object(value: Any, src: int = 0) -> Any:
    if not dist.is_initialized():
        return value
    payload = [value]
    dist.broadcast_object_list(payload, src=src)
    return payload[0]


def broadcast_tensor(value: Tensor, src: int = 0) -> Tensor:
    """In-place broadcast (DMD2 shares one exit step across all ranks)."""
    if dist.is_initialized():
        dist.broadcast(value, src=src)
    return value


def mixed_precision_policy(param_dtype: torch.dtype = torch.bfloat16):
    from torch.distributed.fsdp import MixedPrecisionPolicy

    return MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=torch.float32, cast_forward_inputs=False)


def shard_module(
    model: nn.Module,
    units: Sequence[nn.Module],
    *,
    reshard_after_forward: bool = True,
    param_dtype: torch.dtype = torch.bfloat16,
) -> nn.Module:
    """``fully_shard`` each unit, then the root. ``units`` must be non-empty and inside ``model``."""
    from torch.distributed.fsdp import fully_shard

    if not units:
        raise ValueError("shard_module needs the explicit list of FSDP units (the adapter's shard_units).")
    owned = {id(module) for module in model.modules()}
    policy = mixed_precision_policy(param_dtype)
    mesh = cpu_mesh()
    for unit in units:
        if id(unit) not in owned:
            raise ValueError("An FSDP unit is not a submodule of the model being sharded.")
        fully_shard(unit, mesh=mesh, mp_policy=policy, reshard_after_forward=reshard_after_forward)
    fully_shard(model, mesh=mesh, mp_policy=policy, reshard_after_forward=reshard_after_forward)
    return model


def cpu_mesh():
    """An explicit CPU device mesh under gloo, else None (FSDP2's default): with torch_npu installed, the default
    mesh would be an NPU mesh even in a gloo process group."""
    if not dist.is_initialized() or dist.get_backend() != "gloo":
        return None
    from torch.distributed.device_mesh import init_device_mesh

    pin_cpu_defaults()
    return init_device_mesh("cpu", (dist.get_world_size(),))


def materialize_and_load(
    model: nn.Module,
    full_state_dict: Mapping[str, Tensor] | None,
    *,
    device: torch.device,
    reset_derived: Any = None,
) -> None:
    """Allocate a sharded meta model on ``device`` and fill it from rank 0's CPU state dict (strict).

    ``reset_derived(model, device)`` runs after ``to_empty`` and before the load, to rebuild computed tensors.
    """
    from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

    model.to_empty(device=device)
    if reset_derived is not None:
        reset_derived(model, device)
    state = dict(full_state_dict) if (full_state_dict is not None and is_main_process()) else {}
    if is_main_process() and not state:
        raise ValueError("Rank 0 must provide the full state dict.")
    result = set_model_state_dict(
        model, state,
        options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True, strict=True),
    )
    missing = list(getattr(result, "missing_keys", []) or [])
    unexpected = list(getattr(result, "unexpected_keys", []) or [])
    if missing or unexpected:
        raise RuntimeError(f"State dict mismatch after load: missing={missing[:5]} unexpected={unexpected[:5]}")
    leftover = [name for name, tensor in list(model.named_parameters()) + list(model.named_buffers())
                if getattr(tensor, "is_meta", False)]
    if leftover:
        raise RuntimeError(f"Tensors left on the meta device after loading: {leftover[:5]}")


def full_tensor(value: Tensor) -> Tensor:
    """Gather a DTensor (collective on every rank) or return the tensor itself."""
    return value.full_tensor() if hasattr(value, "full_tensor") else value


def trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def clip_grad_norm(parameters: Iterable[nn.Parameter], max_norm: float | None) -> Tensor:
    """Global pre-clip L2 norm over (sharded) gradients; clips in place unless ``max_norm`` is None or <= 0."""
    params = [parameter for parameter in parameters if parameter.grad is not None]
    if not params:
        raise RuntimeError("No gradients to clip.")
    limit = float(max_norm) if (max_norm is not None and max_norm > 0) else float("inf")
    # CPU: foreach off (torch_npu's patched foreach probe would initialize an NPU); devices: torch's default
    foreach = False if all(parameter.grad.device.type == "cpu" for parameter in params) else None
    norm = torch.nn.utils.clip_grad_norm_(params, max_norm=limit, foreach=foreach)
    return full_tensor(norm).detach().float()


def reduce_mean(values: Mapping[str, float | Tensor], device: torch.device) -> dict[str, float]:
    if not values:
        return {}
    keys = sorted(values)
    stacked = torch.tensor([float(_scalar(values[key])) for key in keys], device=device, dtype=torch.float32)
    if dist.is_initialized():
        dist.all_reduce(stacked, op=dist.ReduceOp.SUM)
        stacked /= dist.get_world_size()
    return {key: float(stacked[index].item()) for index, key in enumerate(keys)}


def reduce_max(values: Mapping[str, float | Tensor], device: torch.device) -> dict[str, float]:
    if not values:
        return {}
    keys = sorted(values)
    stacked = torch.tensor([float(_scalar(values[key])) for key in keys], device=device, dtype=torch.float32)
    if dist.is_initialized():
        dist.all_reduce(stacked, op=dist.ReduceOp.MAX)
    return {key: float(stacked[index].item()) for index, key in enumerate(keys)}


def _scalar(value: float | Tensor) -> float:
    if isinstance(value, Tensor):
        return float(value.detach().float().mean().item())
    return float(value)


__all__ = [
    "barrier", "broadcast_object", "broadcast_tensor", "clip_grad_norm", "cpu_mesh", "full_tensor",
    "init_distributed", "is_main_process", "materialize_and_load", "mixed_precision_policy", "pin_cpu_defaults",
    "rank", "reduce_max", "reduce_mean", "shard_module", "trainable_parameters", "world_size",
]

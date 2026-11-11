"""Every random draw of the algorithm goes through a named ``Draws`` object (``"gen.dm.z"``,
``"rollout.step_noise.2"``, ...), so another source can stand in for the torch RNG, e.g. to align with a reference
implementation that draws in another order. ``TorchDraws`` calls the torch RNG in call order (per-rank seeded by the
loop).
"""

from __future__ import annotations

import torch
from torch import Tensor


class Draws:
    """The interface: a tensor per draw name (a name is drawn at most once per stream)."""

    def randn(self, name: str, like: Tensor) -> Tensor:  # pragma: no cover - interface
        raise NotImplementedError

    def rand(self, name: str, shape: tuple[int, ...], *, device: torch.device, dtype: torch.dtype = torch.float32) -> Tensor:  # pragma: no cover
        raise NotImplementedError

    def randint(self, name: str, high: int, shape: tuple[int, ...], *, device: torch.device) -> Tensor:  # pragma: no cover
        raise NotImplementedError


class TorchDraws(Draws):
    """The torch RNG (global, or an explicit generator for per-image evaluation seeds).

    With a generator, every draw happens on the generator's own device and is then moved, so a value depends on the
    seed only, not on the latents' device.
    """

    def __init__(self, generator: torch.Generator | None = None) -> None:
        self.generator = generator

    def randn(self, name: str, like: Tensor) -> Tensor:
        if self.generator is None:
            return torch.randn_like(like)
        value = torch.randn(like.shape, generator=self.generator, device=self.generator.device, dtype=like.dtype)
        return value.to(like.device)

    def rand(self, name: str, shape: tuple[int, ...], *, device: torch.device, dtype: torch.dtype = torch.float32) -> Tensor:
        if self.generator is None:
            return torch.rand(shape, device=device, dtype=dtype)
        return torch.rand(shape, generator=self.generator, device=self.generator.device, dtype=dtype).to(device)

    def randint(self, name: str, high: int, shape: tuple[int, ...], *, device: torch.device) -> Tensor:
        if self.generator is None:
            return torch.randint(0, int(high), shape, device=device)
        return torch.randint(0, int(high), shape, generator=self.generator, device=self.generator.device).to(device)


__all__ = ["Draws", "TorchDraws"]

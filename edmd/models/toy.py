"""A tiny denoiser and adapter for CPU-only checks.

``ToyDenoiser`` maps packed tokens ``[B, N, C]``, a per-sample time and a conditioning vector to a velocity; every
layer is an ``nn.Linear`` so LoRA injection, merging and FSDP units can be exercised. It runs in any dtype.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor, nn

from .base import Geometry, ModelAdapter, register


class ToyBlock(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden, hidden)

    def forward(self, h: Tensor) -> Tensor:
        return h + torch.tanh(self.proj(h))


class ToyDenoiser(nn.Module):
    def __init__(self, channels: int = 4, hidden: int = 16, cond_dim: int = 8, blocks: int = 2) -> None:
        super().__init__()
        self.inp = nn.Linear(channels, hidden)
        self.time = nn.Linear(4, hidden)
        self.cond = nn.Linear(cond_dim, hidden)
        self.blocks = nn.ModuleList([ToyBlock(hidden) for _ in range(blocks)])
        self.out = nn.Linear(hidden, channels)

    def forward(self, x: Tensor, t: Tensor, cond: Tensor) -> Tensor:
        t = t.to(x.dtype).reshape(-1)
        emb = torch.stack([t, t.square(), torch.sin(math.pi * t), torch.cos(math.pi * t)], dim=-1)
        h = self.inp(x) + self.time(emb)[:, None, :] + self.cond(cond.to(x.dtype))[:, None, :]
        for block in self.blocks:
            h = block(h)
        return self.out(torch.tanh(h))


@dataclass
class ToyCond:
    embeds: Tensor

    @property
    def batch_size(self) -> int:
        return int(self.embeds.shape[0])


@register
class ToyAdapter(ModelAdapter):
    name = "toy"
    default_lora_targets = ("blocks.*.proj", "out")

    def __init__(self, spec: Any = None, *, tokens: int = 6, channels: int = 4, cond_dim: int = 8,
                 dtype: torch.dtype = torch.float32, shift: float = 3.0, seed: int = 0) -> None:
        super().__init__(spec)
        self.tokens, self.channels, self.cond_dim, self.dtype, self.shift = tokens, channels, cond_dim, dtype, shift
        self._base_tokens, self._base_shift = tokens, shift
        # options.compute_dtype (e.g. bfloat16) mimics the real adapters: an fp32 meta model, bf16 FSDP compute
        options = dict(getattr(spec, "options", None) or {})
        unknown = sorted(set(options) - {"compute_dtype"})
        if unknown:
            raise ValueError(f"unknown toy options {unknown}")
        self.compute_dtype = getattr(torch, options["compute_dtype"]) if "compute_dtype" in options else dtype
        generator = torch.Generator().manual_seed(seed)
        self._prompt_table = torch.randn(64, cond_dim, generator=generator, dtype=torch.float64)
        self._base_seed = seed

    def geometry(self) -> Geometry:
        return getattr(self, "_geometry", Geometry(64, 64))

    def geometry_table(self) -> dict[str, dict[str, Geometry]]:
        """Two tiers for multi-resolution checks: the token count scales with the area and the shift with it."""
        return {"small": {"1:1": Geometry(64, 64), "3:2": Geometry(64, 96)},
                "large": {"1:1": Geometry(96, 96), "3:2": Geometry(96, 144), "2:3": Geometry(144, 96)}}

    def at(self, geometry: Geometry) -> "ToyAdapter":
        if geometry == self.geometry():
            return self
        base = Geometry(64, 64)
        view = copy.copy(self)
        view._geometry = geometry
        scale = geometry.height * geometry.width / (base.height * base.width)
        view.tokens = max(1, round(self._base_tokens * scale))
        view.shift = self._base_shift * scale ** 0.5      # a token-dependent shift, like Qwen's exp(mu(tokens))
        return view

    def native_shift_factor(self, num_steps: int) -> float:
        return float(self.shift)

    def build_on_meta(self) -> nn.Module:
        with torch.device("meta"):
            return ToyDenoiser(self.channels, 16, self.cond_dim).to(self.dtype)

    def build_model(self, *, seed: int | None = None) -> ToyDenoiser:
        torch.manual_seed(self._base_seed if seed is None else seed)
        return ToyDenoiser(self.channels, 16, self.cond_dim).to(self.dtype)

    def fsdp_units(self, model: nn.Module) -> list[nn.Module]:
        return list(model.blocks)

    def read_base_state(self, expected: set[str], *, dtype: torch.dtype) -> dict[str, Tensor]:
        state = self.build_model().state_dict()
        return {key: value.to(dtype) for key, value in state.items() if key in expected}

    def load_conditioning(self, device: torch.device, *, shard: bool = False) -> None:
        return None

    def encode(self, prompts: Sequence[str]) -> ToyCond:
        rows = [self._prompt_table[sum(map(ord, prompt)) % 64] for prompt in prompts]
        return ToyCond(torch.stack(rows).to(self.dtype))

    def encode_negative(self, batch_size: int) -> ToyCond:
        return ToyCond(torch.zeros(batch_size, self.cond_dim, dtype=self.dtype))

    def predict_velocity(self, model: nn.Module, x: Tensor, sigma: Tensor, cond: ToyCond) -> Tensor:
        if self.compute_dtype == self.dtype:
            return model(x, sigma, cond.embeds.to(x.device))
        out = model(x.to(self.compute_dtype), sigma, cond.embeds.to(device=x.device, dtype=self.compute_dtype))
        return out.to(x.dtype)

    def sample_noise(self, batch_size: int, *, device: torch.device, generator: Any = None) -> Tensor:
        return torch.randn(batch_size, self.tokens, self.channels, generator=generator, device=device, dtype=self.dtype)

    def load_decoder(self, device: torch.device) -> None:
        return None

    def decode(self, latents: Tensor) -> Tensor:
        """``[B, 3, H, W]`` pixels at this view's geometry: each token's channel mean in [0, 1] as a vertical stripe."""
        values = latents.float().mean(dim=-1).clamp(-1, 1) * 0.5 + 0.5                   # [B, tokens]
        geometry = self.geometry()
        stripes = torch.nn.functional.interpolate(values[:, None, None, :], size=(geometry.height, geometry.width),
                                                  mode="nearest")
        return stripes.expand(-1, 3, -1, -1)

    def load_inference_denoiser(self, weights: Path | None, device: torch.device, *, role_train: str = "lora") -> nn.Module:
        return self.build_model().to(device)


__all__ = ["ToyAdapter", "ToyBlock", "ToyCond", "ToyDenoiser"]

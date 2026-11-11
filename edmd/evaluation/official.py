"""Released few-step checkpoints (the SD3 TDM and CDM baselines) rendered with their own samplers.

Prompts, seeds, initial noise, encode / decode and PNG rounding stay the evaluation's; only the sampler the release
documents changes:

* ``sigmas`` (CDM): Euler on the grid the model's own scheduler makes from the raw sigmas, with full transformer
  weights from a diffusers directory;
* ``scheduler`` (TDM): a diffusers scheduler's own denoising loop;
* ``lora`` (TDM): a diffusers / PEFT LoRA file, attached unmerged.

The recipe helpers import neither torch nor diffusers; ``official_renderer`` runs in the device processes of
``edmd.evaluation.generate``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .generate import image_generator, render_case, to_uint8

SIGMAS_SAMPLER = "euler"
LORA_PREFIX = "transformer."
#: what ``unmerged_compute`` means here: the adapter cast to the base layer's (compute) dtype, as diffusers + peft load it
LORA_NUMERICS = "unmerged_compute"


# ============================================================================ recipe (no torch)
def parse_sigmas(text: str | Sequence[float]) -> list[float]:
    values = [float(item) for item in (text.split(",") if isinstance(text, str) else text) if str(item).strip()]
    if not values or any(not math.isfinite(v) or not 0.0 < v <= 1.0 for v in values) or any(
            a <= b for a, b in zip(values, values[1:])):
        raise ValueError(f"sigmas must strictly decrease within (0, 1], got {values}")
    return values


def load_scheduler_config(path: Path, overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """A diffusers scheduler config file plus overrides (TDM: Sana's config with ``flow_shift = 6``)."""
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not str(config.get("_class_name", "")).endswith("Scheduler"):
        raise ValueError(f"{path} is not a diffusers scheduler config (no *Scheduler _class_name)")
    for key, value in dict(overrides or {}).items():
        if key.startswith("_"):
            raise ValueError(f"refusing to override the private key {key!r}")
        config[key] = value
    return config


def parse_overrides(items: Sequence[str] | None) -> dict[str, Any]:
    """``KEY=VALUE`` (VALUE as JSON, else a string): ``flow_shift=6`` -> ``{"flow_shift": 6}``."""
    overrides: dict[str, Any] = {}
    for item in items or ():
        key, sep, raw = item.partition("=")
        if not sep or not key.strip():
            raise ValueError(f"expected KEY=VALUE, got {item!r}")
        try:
            overrides[key.strip()] = json.loads(raw)
        except json.JSONDecodeError:
            overrides[key.strip()] = raw
    return overrides


def lora_spec(path: Path, weight: float) -> dict[str, Any]:
    """The LoRA part of an official recipe (the file and the adapter weight of ``set_adapters``)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if not math.isfinite(float(weight)) or float(weight) == 0.0:
        raise ValueError(f"LoRA weight must be finite and non-zero, got {weight}")
    return {"path": str(path.absolute()), "weight": float(weight)}


def official_recipe(*, steps: int, sigmas: Sequence[float] | None = None, scheduler: Mapping[str, Any] | None = None,
                    lora: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The checked sampler of a released checkpoint: exactly one of ``sigmas`` / ``scheduler``."""
    if (sigmas is None) == (scheduler is None):
        raise ValueError("an official recipe needs exactly one sampler: sigmas (Euler on the model's scheduler grid) "
                         "or a diffusers scheduler config")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    recipe: dict[str, Any] = {"sampler": SIGMAS_SAMPLER if sigmas is not None else str(scheduler["_class_name"]),
                              "num_steps": int(steps), "sigmas": None, "scheduler": None,
                              "lora": None if lora is None else dict(lora)}
    if sigmas is not None:
        values = parse_sigmas(sigmas)
        if len(values) != steps:
            raise ValueError(f"{len(values)} sigmas for {steps} steps")
        recipe["sigmas"] = values
    else:
        recipe["scheduler"] = dict(scheduler)
    return recipe


# ============================================================================ device side (torch, diffusers)
def sigma_grid(adapter: Any, sigmas: Sequence[float]):
    """``[K+1]`` float32 CPU grid of ``pipe(..., sigmas=sigmas)`` (the model's own scheduler, terminal 0 included),
    checked to have the timesteps the Euler sampler feeds (``sigma * 1000``)."""
    import diffusers
    import torch

    config = dict(adapter.scheduler_config)
    scheduler = getattr(diffusers, config.get("_class_name", "FlowMatchEulerDiscreteScheduler")).from_config(config)
    scheduler.set_timesteps(sigmas=list(sigmas), device="cpu")
    grid = scheduler.sigmas.to(dtype=torch.float32, device="cpu").clone()
    expected = (scheduler.sigmas[:-1] * scheduler.config.num_train_timesteps).to(torch.float32)
    if grid.numel() != len(sigmas) + 1 or float(grid[-1]) != 0.0 or not torch.equal(
            scheduler.timesteps.to(device="cpu", dtype=torch.float32), expected):
        raise ValueError(f"{type(scheduler).__name__} does not give a sigma grid with timesteps = sigma * 1000")
    return grid


def make_scheduler(config: Mapping[str, Any]):
    import diffusers

    return getattr(diffusers, str(config["_class_name"])).from_config(dict(config))


def sd3_velocity_at(adapter: Any, model: Any, x: Any, timestep: Any, cond: Any):
    """``SD3MediumAdapter.predict_velocity`` at the scheduler's own timestep (DPM-Solver's are int64, not
    ``sigma * 1000``); ``v`` as float32."""
    if getattr(adapter, "name", None) != "sd3_medium":
        raise NotImplementedError("the scheduler sampler drives the SD3 transformer call only")
    adapter._check_latents(x)
    batch = int(x.shape[0])
    if cond.batch_size != batch:
        raise ValueError(f"latents have batch {batch}, the conditioning has {cond.batch_size}.")
    output = model(
        hidden_states=x.to(adapter.compute_dtype),
        timestep=timestep.to(x.device).expand(batch),
        encoder_hidden_states=cond.prompt_embeds.to(device=x.device, dtype=adapter.compute_dtype),
        pooled_projections=cond.pooled_prompt_embeds.to(device=x.device, dtype=adapter.compute_dtype),
        joint_attention_kwargs=None,
        return_dict=False,
    )[0]
    return output.float()


def sample_with_scheduler(adapter: Any, model: Any, cond: Any, latents: Any, *, scheduler_config: Mapping[str, Any],
                          num_steps: int):
    """``StableDiffusion3Pipeline``'s guidance-free denoising loop on ``latents``; returns ``(latents, scheduler)``."""
    scheduler = make_scheduler(scheduler_config)
    scheduler.set_timesteps(int(num_steps), device=latents.device)
    dtype = adapter.sampling_state_dtype
    for t in scheduler.timesteps:
        noise_pred = sd3_velocity_at(adapter, model, latents, t, cond).to(dtype)
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
    return latents, scheduler


def render_case_scheduler(adapter: Any, model: Any, case: Mapping[str, Any], *, scheduler_config: Mapping[str, Any],
                          num_steps: int, device: Any, height: int | None = None, width: int | None = None):
    """One image as ``generate.render_case`` renders it, with :func:`sample_with_scheduler` as the sampler; returns
    ``(pixels uint8, scheduler)``."""
    generator = image_generator(adapter, case["seed"], device)
    with adapter.sampling_context(device):
        cond = adapter.encode([case["render"]])
        noise = adapter.sample_noise(1, device=device, generator=generator).to(adapter.sampling_state_dtype)
        latents, scheduler = sample_with_scheduler(adapter, model, cond, noise, scheduler_config=scheduler_config,
                                                   num_steps=num_steps)
        pixels = adapter.decode(latents)
    return to_uint8(pixels, height=height, width=width), scheduler


def transformer_init_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    """The ``SD3Transformer2DModel.__init__`` kwargs a config builds, class defaults filled in, so that CDM's turbo
    config (which spells out keys the base config predates) compares equal to the base's."""
    from ..models.sd3 import components

    return components.normalized_init_config(config)


def load_full_transformer(adapter: Any, path: Path, device: Any):
    """The base SD3 transformer with every parameter (and any buffer the files hold) replaced by the tensors of a
    diffusers transformer directory or pipeline root, cast to the compute dtype; returns ``(model, provenance)``."""
    from ..models.sd3 import components

    directory = components.diffusers_transformer_dir(path)
    if directory is None:
        raise ValueError(f"{path} is not a diffusers SD3 transformer directory or a pipeline root with one")
    theirs = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    if transformer_init_kwargs(theirs) != transformer_init_kwargs(adapter.transformer_config):
        raise ValueError(f"{directory} is not the base architecture: {transformer_init_kwargs(theirs)} != "
                         f"{transformer_init_kwargs(adapter.transformer_config)}")
    model = adapter.build_on_meta()
    expected = set(model.state_dict().keys())
    parameters = {name for name, _ in model.named_parameters()}
    base, _ = components.read_transformer_state(adapter.model_path, expected, dtype=adapter.compute_dtype)
    files, variant = components.weight_files(directory)
    tensors, provenance = components.read_state(files, expected, dtype=adapter.compute_dtype, where=str(directory),
                                                variant=variant, optional=expected - parameters)
    components.assign_state(model, {**base, **tensors})
    model.requires_grad_(False)
    model.eval()
    return model.to(device), {**provenance, "source": str(directory), "from_file": len(tensors),
                              "buffers_from_base": sorted(expected - set(tensors))}


def read_peft_lora(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """``{<module>.lora_A|lora_B.weight: tensor}`` of a diffusers transformer LoRA, plus ``{"rank", "modules"}``; only
    plain ``transformer.`` weights of one rank without alphas / metadata (from which diffusers would set
    ``lora_alpha``) are accepted."""
    from safetensors import safe_open

    tensors: dict[str, Any] = {}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
        for key in handle.keys():
            if not key.startswith(LORA_PREFIX) or not key.endswith((".lora_A.weight", ".lora_B.weight")):
                raise ValueError(f"{path}: unsupported LoRA key {key!r} (expected {LORA_PREFIX}<module>.lora_A|B.weight)")
            tensors[key[len(LORA_PREFIX):]] = handle.get_tensor(key)
    extra = sorted(set(metadata) - {"format"})
    if extra:
        raise ValueError(f"{path}: LoRA metadata {extra} (alphas / adapter config) is not supported")
    modules = sorted({key.rsplit(".lora_", 1)[0] for key in tensors})
    ranks = set()
    for module in modules:
        a, b = tensors.get(f"{module}.lora_A.weight"), tensors.get(f"{module}.lora_B.weight")
        if a is None or b is None or a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]:
            raise ValueError(f"{path}: {module} has no matching lora_A / lora_B pair")
        ranks.add(int(b.shape[1]))
    if len(ranks) != 1:
        raise ValueError(f"{path}: LoRA ranks {sorted(ranks)} differ across modules (rank_pattern is not supported)")
    return tensors, {"rank": ranks.pop(), "modules": modules}


def attach_peft_lora(model: Any, path: Path, *, weight: float, compute_dtype: Any) -> dict[str, Any]:
    """``load_lora_weights`` + ``set_adapters([name], [weight])`` on a materialized ``compute_dtype`` transformer."""
    from ..models.sd3.inference_lora import attach_unmerged
    from ..runtime.lora import LoRALinear, LoraSpec

    tensors, info = read_peft_lora(Path(path))
    rank = int(info["rank"])
    alpha = float(rank)                      # diffusers' lora_alpha when the file stores no alphas
    spec = LoraSpec(rank=rank, alpha=alpha, dropout=0.0, init="gaussian", target_modules=tuple(info["modules"]))
    names = attach_unmerged(model, spec, tensors, mode=LORA_NUMERICS, compute_dtype=compute_dtype)
    scaling = float(weight) * alpha / rank   # as peft scales an adapter
    count = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_scaling = scaling
            count += 1
    if count != len(info["modules"]) or sorted(names) != info["modules"]:
        raise RuntimeError(f"attached {count} LoRA modules, the file has {len(info['modules'])}")
    return {"rank": rank, "alpha": alpha, "scaling": scaling, "modules": len(info["modules"]),
            "adapter_dtype": str(compute_dtype).replace("torch.", "")}


def official_renderer(adapter: Any, recipe: Mapping[str, Any], weights: Path | None, device: Any):
    """``case -> uint8 [H, W, 3]`` for a released checkpoint (``weights``: a diffusers transformer directory or an
    edmd weights file; None: the base) sampled by ``recipe`` (``official_recipe``), guidance-free."""
    import torch

    steps = int(recipe["num_steps"])
    if recipe["sigmas"] is not None:
        grid = sigma_grid(adapter, recipe["sigmas"])
    else:                                    # the scheduler's own grid (every image re-creates the scheduler)
        probe = make_scheduler(recipe["scheduler"])
        probe.set_timesteps(steps, device=device)
        grid = probe.sigmas.to(dtype=torch.float32, device="cpu").clone()
    adapter.load_conditioning(device)
    adapter.load_decoder(device)
    if weights is not None and weights.is_dir():          # a release's diffusers transformer (CDM's turbo snapshot)
        model, _ = load_full_transformer(adapter, weights, device)
    else:
        model = adapter.load_inference_denoiser(weights, device, role_train="full")
    if recipe.get("lora"):
        attach_peft_lora(model, Path(recipe["lora"]["path"]), weight=float(recipe["lora"]["weight"]),
                         compute_dtype=adapter.compute_dtype)
    model.eval()
    model.requires_grad_(False)

    def render(case: Mapping[str, Any]):
        height, width = case.get("height"), case.get("width")
        if recipe["sigmas"] is not None:
            return render_case(adapter, model, case, grid=grid, sampler=SIGMAS_SAMPLER, cfg=1.0, device=device,
                               height=height, width=width)
        array, scheduler = render_case_scheduler(adapter, model, case, scheduler_config=recipe["scheduler"],
                                                 num_steps=steps, device=device, height=height, width=width)
        if not torch.equal(scheduler.sigmas.to(dtype=torch.float32, device="cpu"), grid):
            raise RuntimeError("the scheduler's grid changed between images")
        return array

    return render


__all__ = ["LORA_NUMERICS", "SIGMAS_SAMPLER", "attach_peft_lora", "load_full_transformer", "load_scheduler_config",
           "lora_spec", "make_scheduler", "official_recipe", "official_renderer", "parse_overrides", "parse_sigmas",
           "read_peft_lora", "render_case_scheduler", "sample_with_scheduler", "sd3_velocity_at", "sigma_grid",
           "transformer_init_kwargs"]

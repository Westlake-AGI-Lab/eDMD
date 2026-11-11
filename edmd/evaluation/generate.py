"""Render an evaluation's images, one process per device; every image is decided by its seed alone.

``generate`` fills one image directory ``<out>/<label>/``::

    images/prompt_00000.png ...   image i
    prompts.jsonl                 per image: index, seed, render prompt, score prompt, height, width (wh_ratio)
    info.json                     the source (checkpoint and role, or the teacher's settings), its sampler and grid,
                                  the step and the eval config (``edmd.cli.evaluate score`` scores its metrics)

Images already on disk are kept, so a rerun renders only the missing ones; a directory that holds another source's
images or other prompts is refused unless ``force``, which renders everything again. Rendering any image removes the
directory's scores. Image ``i``:

* one generator seeded ``seed + i`` (``adapter.eval_generator``) draws the initial noise and every fresh noise of a
  stochastic sampler;
* a checkpoint renders with its own sampler on its recorded grid (``weights/inference.json``), even when the adapter
  now computes another one (that only warns), without guidance; a teacher row is ``steps``-step Euler with CFG
  ``cfg`` on the pipeline's own grid (``official``: a released checkpoint with its own sampler);
* ``adapter.decode`` -> uint8 (``to_uint8``) -> PNG.

Under ``geometry: aspect`` image ``i`` renders at its ``wh_ratio`` in the eval's area tier through ``adapter.at``: the
recorded grid holds at the training geometry, any other geometry samples on its view's own grid.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .parallel import plan_devices, run_on_devices
from .protocol import INFO_FILE, PROMPTS_FILE, RESULT_FILE, Protocol, image_name, read_json, write_json


# ============================================================================ parent side
def case_geometry(adapter: Any, tier: str, ratio: str | None) -> Any:
    """A case's geometry: its ``wh_ratio`` in the area tier (a ratio the tier lacks is 1:1, as in training)."""
    table = adapter.geometry_table()
    if tier not in table:
        raise ValueError(f"eval.geometry_tier {tier!r} is not one of {adapter.name}'s tiers {sorted(table)}")
    return table[tier].get(ratio or "1:1", table[tier]["1:1"])


def plan_cases(protocol: Protocol, adapter: Any, size: tuple[int, int] | None) -> list[dict[str, Any]]:
    """The protocol's cases with the size each image renders at: the eval's per-case geometry, else ``size`` (a
    checkpoint's recorded height and width), else the adapter's own."""
    cases = protocol.cases()
    for case in cases:
        if protocol.tier is not None:
            geometry = case_geometry(adapter, protocol.tier, case.get("wh_ratio"))
            case["height"], case["width"] = int(geometry.height), int(geometry.width)
        else:
            geometry = adapter.geometry()
            case["height"], case["width"] = size or (int(geometry.height), int(geometry.width))
    return cases


def _same_source(directory: Path, source: Mapping[str, Any], prompts: str) -> bool:
    """Whether ``directory`` is empty or holds images of ``source`` for exactly these cases."""
    info, cases = directory / INFO_FILE, directory / PROMPTS_FILE
    if not info.is_file() and not cases.is_file():
        return not (directory / "images").is_dir() or not any((directory / "images").iterdir())
    return (info.is_file() and read_json(info).get("source") == source
            and cases.is_file() and cases.read_text(encoding="utf-8") == prompts)


def generate(directory: Path, *, job: Mapping[str, Any], source: Mapping[str, Any], protocol: Protocol,
             info: Mapping[str, Any], devices: Sequence[str] | None = None, force: bool = False) -> Path:
    """Render ``protocol``'s images of ``job`` into ``directory`` (module docstring); returns the directory.

    ``job``: what the device processes load and how they sample (``model`` as a mapping, ``weights``, ``role_train``,
    ``sampler``, ``grid``, ``cfg``, ``size`` and the released-checkpoint recipe ``official``). ``source`` names the
    images (it is compared on a rerun); ``info`` goes into ``info.json`` with it."""
    from ..config.core import from_mapping
    from ..config.schema import ModelSpec
    from ..models.base import build_adapter

    directory = Path(directory)
    adapter = build_adapter(from_mapping(ModelSpec, job["model"], "model"))    # configs only, no weights
    size = tuple(job["size"]) if job.get("size") else None
    cases = plan_cases(protocol, adapter, size)
    prompts = "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases)
    if not _same_source(directory, source, prompts):
        if not force:
            raise SystemExit(f"{directory} holds images of another source or other prompts: pass --force to render "
                             "it again, or choose another --out / --label")
        print(f"--force: removing the images of {directory}", flush=True)
    if force:
        for stale in ("images", "scores"):
            shutil.rmtree(directory / stale, ignore_errors=True)
        (directory / RESULT_FILE).unlink(missing_ok=True)
    (directory / "images").mkdir(parents=True, exist_ok=True)
    (directory / PROMPTS_FILE).write_text(prompts, encoding="utf-8")
    write_json(directory / INFO_FILE, {**dict(info), "source": dict(source)})
    pending = [case for case in cases if not (directory / image_name(case["index"])).is_file()]
    if not pending:
        print(f"{directory}: all {len(cases)} images exist", flush=True)
        return directory
    shutil.rmtree(directory / "scores", ignore_errors=True)        # scores of other images
    (directory / RESULT_FILE).unlink(missing_ok=True)
    devices = plan_devices(devices)[:len(pending)]
    print(f"{directory}: rendering {len(pending)} of {len(cases)} images on {', '.join(devices)}", flush=True)
    started = time.time()
    run_on_devices(render_images, devices,
                   [(dict(job, aspect=protocol.tier is not None), pending[rank::len(devices)], str(directory))
                    for rank in range(len(devices))])
    missing = [case["index"] for case in cases if not (directory / image_name(case["index"])).is_file()]
    if missing:
        raise RuntimeError(f"{directory}: {len(missing)} images were not rendered, e.g. {missing[:5]}")
    print(f"{directory}: {len(pending)} images in {time.time() - started:.0f}s", flush=True)
    return directory


# ============================================================================ device side (imports torch)
def image_generator(adapter: Any, seed: int, device: Any):
    return adapter.eval_generator(int(seed), device)


def to_uint8(pixels: Any, *, height: int | None = None, width: int | None = None):
    """[1,3,H,W] in [0,1] -> [H,W,3] uint8, rounded as diffusers' ``numpy_to_pil``."""
    import numpy as np

    array = pixels.detach().float().cpu()
    if array.ndim != 4 or array.shape[0] != 1 or array.shape[1] != 3:
        raise ValueError(f"adapter.decode must return [1, 3, H, W] pixels, got {tuple(array.shape)}")
    if height is not None and (array.shape[2], array.shape[3]) != (height, width):
        raise ValueError(f"decoded {tuple(array.shape[2:])}, expected ({height}, {width})")
    low, high = float(array.min()), float(array.max())
    if not (0.0 <= low and high <= 1.0):
        raise ValueError(f"adapter.decode must return pixels in [0, 1], got [{low}, {high}]")
    values = array.permute(0, 2, 3, 1).numpy()
    return (values * 255).round().astype(np.uint8)[0]


def png_bytes(array: Any) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def write_atomic(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def resolve_grid(adapter: Any, spec: Mapping[str, Any]):
    """The float32 grid a job samples on: the recorded values (with a warning when the adapter now computes others),
    or the pipeline's."""
    import torch

    if spec["kind"] == "recorded":
        grid = torch.tensor([float.fromhex(value) for value in spec["float32_hex"]], dtype=torch.float32)
        recomputed = adapter.student_grid(int(spec["num_steps"]), str(spec["schedule"])).detach().float().cpu()
        if recomputed.shape != grid.shape or not torch.equal(recomputed, grid):
            print(f"warning: the adapter now computes the grid {recomputed.tolist()} for this checkpoint (a diffusers "
                  f"upgrade?), but it was trained and recorded with {grid.tolist()}; sampling on the recorded grid",
                  flush=True)
        return grid
    if spec["kind"] == "pipeline":
        maker = getattr(adapter, "pipeline_grid", None)
        steps = int(spec["num_steps"])
        grid = maker(steps) if maker is not None else adapter.student_grid(steps, "native")
        return grid.detach().float().cpu()
    raise ValueError(f"unknown grid kind {spec['kind']!r}")


def make_velocity(adapter: Any, model: Any, cond: Any, *, cfg: float, negative: Any = None):
    """``(x, sigma) -> v`` for one image through the adapter's sampling velocity (guided under CFG); velocities and
    the CFG combination are in ``adapter.sampling_state_dtype``, as in a pipeline whose transformer returns bf16."""
    dtype = adapter.sampling_state_dtype
    if cfg == 1.0:
        conditional = adapter.sampling_velocity(model, cond)
        return lambda x, t: conditional(x, t).to(dtype)
    pair = adapter.sampling_velocity_pair(model, cond, negative)   # SD3: the pipeline's one [uncond; cond] forward

    def guided(x, t):
        v_cond, v_uncond = pair(x, t)
        return adapter.guide(v_cond.to(dtype), v_uncond.to(dtype), cfg)

    return guided


def sample_teacher(adapter: Any, model: Any, cond: Any, noise: Any, grid: Any, *, cfg: float,
                   negative: Any = None, generator: Any = None):
    """The teacher's Euler sampler on ``grid`` with CFG ``cfg`` (calibration and teacher rows)."""
    from ..algo.rollout import sample

    velocity = make_velocity(adapter, model, cond, cfg=cfg, negative=negative)
    return sample(velocity, noise, grid.to(noise.device), "euler", generator=generator)


def render_case(adapter: Any, model: Any, case: Mapping[str, Any], *, grid: Any, sampler: str, cfg: float,
                device: Any, negative: Any = None, height: int | None = None, width: int | None = None):
    """One image: generator(seed) -> noise -> the sampler -> decode -> uint8 ``[H, W, 3]``."""
    from ..algo.rollout import sample

    generator = image_generator(adapter, case["seed"], device)   # fresh draws happen on its device (TorchDraws)
    with adapter.sampling_context(device):   # SD3: bf16 autocast over encode, rollout and decode, like CDM's eval
        cond = adapter.encode([case["render"]])
        noise = adapter.sample_noise(1, device=device, generator=generator).to(adapter.sampling_state_dtype)
        velocity = make_velocity(adapter, model, cond, cfg=cfg, negative=negative)
        latents = sample(velocity, noise, grid.to(noise.device), sampler, generator=generator)
        pixels = adapter.decode(latents)
    return to_uint8(pixels, height=height, width=width)


def renderer(adapter: Any, job: Mapping[str, Any], weights: Path | None, device: Any):
    """``case -> uint8 [H, W, 3]``: the job's denoiser with its sampler and grid on ``device``."""
    from ..models.base import Geometry

    grid = resolve_grid(adapter, job["grid"])
    adapter.load_conditioning(device)
    adapter.load_decoder(device)
    model = adapter.load_inference_denoiser(weights, device, role_train=job["role_train"])
    model.eval()
    model.requires_grad_(False)
    cfg = float(job["cfg"])
    with adapter.sampling_context(device):
        negative = adapter.encode_negative(1) if cfg != 1.0 else None
    views: dict[tuple[int, int], tuple[Any, Any, Any]] = {}      # (height, width) -> (view, grid, negative)

    def render(case: Mapping[str, Any]):
        height, width = int(case["height"]), int(case["width"])
        if not job.get("aspect"):
            view, view_grid, view_negative = adapter, grid, negative
        else:
            if (height, width) not in views:
                view = adapter.at(Geometry(height=height, width=width))
                if view is adapter:
                    view_grid = grid                 # the recorded (or pipeline) grid, resolved above
                elif job["grid"]["kind"] == "recorded":
                    view_grid = view.student_grid(int(job["grid"]["num_steps"]),
                                                  str(job["grid"]["schedule"])).detach().float().cpu()
                else:
                    view_grid = resolve_grid(view, job["grid"])
                with view.sampling_context(device):
                    views[(height, width)] = (view, view_grid, view.encode_negative(1) if cfg != 1.0 else None)
            view, view_grid, view_negative = views[(height, width)]
        return render_case(view, model, case, grid=view_grid, sampler=job["sampler"], cfg=cfg, device=device,
                           negative=view_negative, height=height, width=width)

    return render


def render_images(device_name: str, job: Mapping[str, Any], cases: Sequence[Mapping[str, Any]],
                  directory: str) -> None:
    """Render ``cases`` on one device into ``directory/images`` (one process per device)."""
    import torch

    from ..config.core import from_mapping
    from ..config.schema import ModelSpec
    from ..models.base import build_adapter
    from ..runtime import device as device_lib

    device = torch.device(device_name)
    device_lib.set_device(device)          # npu / cuda (a CPU device needs nothing)
    adapter = build_adapter(from_mapping(ModelSpec, job["model"], "model"))
    weights = Path(job["weights"]) if job.get("weights") else None
    if job.get("official"):                # a released checkpoint with its own sampler
        from .official import official_renderer

        render = official_renderer(adapter, job["official"], weights, device)
    else:
        render = renderer(adapter, job, weights, device)
    with torch.no_grad():
        for case in cases:
            tick = time.time()
            write_atomic(Path(directory) / image_name(case["index"]), png_bytes(render(case)))
            print(f"[{device_name}] {image_name(case['index'])} seed={case['seed']} {time.time() - tick:.1f}s",
                  flush=True)


__all__ = ["case_geometry", "generate", "image_generator", "make_velocity", "plan_cases", "png_bytes", "render_case",
           "render_images", "renderer", "resolve_grid", "sample_teacher", "to_uint8", "write_atomic"]

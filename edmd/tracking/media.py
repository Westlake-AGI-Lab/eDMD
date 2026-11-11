"""W&B presentation of the periodic evaluation, built from summary keys only (no history rows): native score panels
and the paged image gallery.

* Score panels: ``build_curve_summary`` writes ``eval/curve_<step>_<metric>`` and ``eval/curve_latest_*`` summary
  scalars; ``configure_native_backfill_panels`` registers a native ``Run History Line Plot`` per ``eval/<metric>``
  that reads them on the checkpoint-step axis. Because the panels read the summary, publishing again (e.g. after a
  re-score) rewrites the curves in place, which panels over history rows cannot do.
* Image gallery (optional, ``eval.inline_chart``): each checkpoint's first ``preview_count`` images become one
  immutable JSON page of base64 JPEGs (``run.save``); a summary table lists the pages, and a custom Vega preset
  (``paged_image_grid_spec``, registered once per entity by ``register_preset``) loads only the selected page.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import re
from collections.abc import Mapping, Sequence
from numbers import Real
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit

#: the preset's name under an entity: ``eval.inline_chart: <entity>/edmd-eval-gallery`` once registered
CHART_NAME = "edmd-eval-gallery"
#: (``fixed_reference`` is always false; the registered preset reads the column)
COLUMNS = ["checkpoint", "page_url", "page_filename", "page_origin", "preview_count", "fixed_reference"]
EMPTY_URL = "data:application/json,%7B%22data%22%3A%5B%5D%7D"
LATEST_STEP_KEY = "eval/curve_latest_step"


# ============================================================================ gallery pages
def preview_rows(directory: Path, step: int, count: int) -> list:
    """The first ``count`` images of an image directory: ``[case, step, seed, score prompt, JPEG data URI]``."""
    from PIL import Image

    from ..evaluation.protocol import image_name, read_cases

    rows = []
    for case in read_cases(directory)[:count]:
        buffer = io.BytesIO()
        with Image.open(Path(directory) / image_name(case["index"])) as image:
            # Full resolution; the JPEG is only the browser preview (scoring reads the original PNGs).
            image.convert("RGB").save(buffer, format="JPEG", quality=95, subsampling=0)
        uri = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
        rows.append([case["index"], int(step), case["seed"], case["score"], uri])
    return rows


def history_preview_rows(points: Mapping[int, Path], count: int) -> list:
    """Every point's preview rows in step order (``points``: step -> image directory), which must show the same
    prompts and seeds."""
    if not points:
        raise ValueError("No completed evaluation results found")
    rows, cases = [], None
    for step, directory in sorted(points.items()):
        page = preview_rows(directory, step, count)
        shown = [(row[3], row[2]) for row in page]
        if cases is not None and shown != cases:
            raise ValueError("Evaluation previews must use the same cases, order, and seeds")
        cases = shown
        rows += [row + [False] for row in page]
    return rows


def paged_history_chart(run: Any, output_dir: Path, count: int, chart_id: str, *, points: Mapping[int, Path],
                        wandb_module: Any):
    """Upload immutable per-checkpoint pages (``run.save``, no history) and return the small manifest chart."""
    if count < 1:
        raise ValueError("Paged gallery preview count must be positive")
    rows = history_preview_rows(points, count)
    pages: dict[int, list] = {}
    for row in rows:
        pages.setdefault(row[1], []).append(row)
    parsed = urlsplit(str(run.url))
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("Paged galleries require an HTTPS W&B run URL")
    origin = f"{parsed.scheme}://{parsed.netloc}"
    run_path = "/".join(quote(str(value), safe="") for value in (run.entity, run.project, run.id))
    # Keep published pages separate from model checkpoints and full PNG inputs.
    base = Path(output_dir) / "wandb_gallery" / str(run.id)
    directory = base / "eval_previews"
    directory.mkdir(parents=True, exist_ok=True)
    manifest = []
    for step, page_rows in sorted(pages.items()):
        content = json.dumps({"columns": ["case", "checkpoint", "seed", "prompt", "image", "fixed_reference"],
                              "data": page_rows}, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        digest = hashlib.sha256(content).hexdigest()
        filename = f"checkpoint-{step:06d}-{digest[:20]}.json"
        path = directory / filename
        if path.exists():
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError(f"Existing preview page is damaged: {path}")
        else:
            temporary = path.with_suffix(".tmp")
            temporary.write_bytes(content)
            temporary.replace(path)
        run.save(str(path), base_path=str(base), policy="now")
        # A stable endpoint that returns a fresh signed URL at read time (the raw object-store URL is private: 403).
        url = f"{origin}/files/{run_path}/eval_previews/{quote(filename, safe='')}?redirect=false"
        manifest.append([step, url, filename, origin, len(page_rows), False])
    chart = wandb_module.plot_table(chart_id, wandb_module.Table(columns=COLUMNS, data=manifest),
                                    fields={key: key for key in COLUMNS}, split_table=True)

    class CumulativeChartSpec(type(chart.spec)):
        @property
        def config_value(self):
            config = super().config_value
            # Explicit, as config merges keep an existing run's older keys; a summary table has no step to select.
            config["panel_config"]["showStepSelector"] = False
            return config

    chart.spec = CumulativeChartSpec(**vars(chart.spec))
    return chart


def publish_summary_chart(run: Any, key: str, chart: Any) -> None:
    """Publish a custom chart and its table without a history row: the chart config is written explicitly (normally
    ``Run.log()`` does it), the table goes into the summary. Not ``log(commit=False)``: ``finish`` flushes that row."""
    chart.set_key(key)
    run._config_callback(val=chart.spec.config_value, key=chart.spec.config_key)
    run.summary[chart.spec.table_key] = chart.table


def paged_image_grid_spec(*, storage_host: str = "") -> dict:
    """The gallery preset's Vega spec: a checkpoint slider (one page load on mouseup) and previous / next buttons.

    ``storage_host``: for a W&B server whose signed file URLs point at an object store the browser cannot reach (or
    with an invalid certificate), URLs on that host are fetched through the W&B origin, which must proxy them.
    """
    if storage_host and not re.fullmatch(r"[A-Za-z0-9.-]+(:[0-9]+)?", storage_host):
        raise ValueError(f"storage_host must be a host name (optionally :port), got {storage_host!r}")
    signed_url = "datum.url" if not storage_host else (
        f"replace(replace(datum.url, 'http://{storage_host}', selected_page._origin), "
        f"'https://{storage_host}', selected_page._origin)")
    checkpoint = "datum['${field:checkpoint}']"
    pointer_index = (
        "clamp(round((clamp(x(), slider_left, slider_left + slider_width) - slider_left)"
        " / slider_width * max(0, length(data('checkpoints')) - 1)),"
        " 0, max(0, length(data('checkpoints')) - 1))"
    )
    drag_start = "@checkpoint_handle:mousedown, @checkpoint_hit:mousedown"
    drag_move = ("[@checkpoint_handle:mousedown, window:mouseup] > window:mousemove!, "
                 "[@checkpoint_hit:mousedown, window:mouseup] > window:mousemove!")
    text_color = {"value": "#616161"}
    return {
        "$schema": "https://vega.github.io/schema/vega/v5.json",
        "autosize": {"type": "fit", "contains": "padding"}, "padding": 5,
        "signals": [
            {"name": "focus_case", "value": -1, "on": [
                {"events": "@preview:click", "update": "focus_case < 0 ? datum.case : -1"},
                {"events": "@instruction:click", "update": "-1"}]},
            {"name": "run_count", "update": "max(1, length(data('runs')))"},
            {"name": "grid_columns", "update": "focus_case >= 0 ? 1 : (run_count === 1 ? 4 : 2)"},
            {"name": "slider_left", "update": "min(108, width * 0.35)"},
            {"name": "slider_width", "update": "max(1, width - slider_left - 84)"},
            {"name": "dragging", "value": False, "on": [
                {"events": drag_start, "update": "true"},
                {"events": "window:mouseup", "update": "false"}]},
            {"name": "drag_index", "value": 2147483647, "on": [
                {"events": drag_start, "update": pointer_index},
                {"events": drag_move, "update": pointer_index}]},
            {"name": "checkpoint_index", "value": 2147483647, "on": [
                {"events": {"signal": "dragging"}, "update": "dragging ? checkpoint_index : drag_index"},
                {"events": "@checkpoint_previous:click", "update": "previous_enabled ? selected_index - 1 : checkpoint_index"},
                {"events": "@checkpoint_next:click", "update": "next_enabled ? selected_index + 1 : checkpoint_index"}]},
            {"name": "selected_index", "update": "clamp(checkpoint_index, 0, max(0, length(data('checkpoints')) - 1))"},
            {"name": "previous_enabled", "update": "selected_index > 0"},
            {"name": "next_enabled", "update": "selected_index < length(data('checkpoints')) - 1"},
            {"name": "selected_checkpoint", "update": "length(data('checkpoints')) ? data('checkpoints')[selected_index]._checkpoint : 0"},
            {"name": "handle_index", "update": "clamp(dragging ? drag_index : checkpoint_index, 0, max(0, length(data('checkpoints')) - 1))"},
            {"name": "slider_position", "update": "slider_left + slider_width * (length(data('checkpoints')) > 1 ? handle_index / (length(data('checkpoints')) - 1) : 0)"},
        ],
        "data": [
            {"name": "wandb"},
            {"name": "manifest", "source": "wandb", "transform": [
                {"type": "formula", "as": "_run_key", "expr": "isValid(datum.id) ? toString(datum.id) : toString(datum.name)"},
                {"type": "formula", "as": "_run_name", "expr": "isValid(datum.name) ? datum.name : datum._run_key"},
                {"type": "formula", "as": "_checkpoint", "expr": checkpoint},
                {"type": "formula", "as": "_page_url", "expr": "datum['${field:page_url}']"},
                {"type": "formula", "as": "_filename", "expr": "datum['${field:page_filename}']"},
                {"type": "formula", "as": "_origin", "expr": "datum['${field:page_origin}']"},
                {"type": "formula", "as": "_count", "expr": "datum['${field:preview_count}']"},
                {"type": "formula", "as": "_is_reference", "expr": "datum['${field:fixed_reference}'] === true"},
            ]},
            {"name": "checkpoints", "source": "manifest", "transform": [
                {"type": "filter", "expr": "!datum._is_reference"},
                {"type": "aggregate", "groupby": ["_checkpoint"]},
                {"type": "collect", "sort": {"field": "_checkpoint", "order": "ascending"}},
            ]},
            {"name": "runs", "source": "manifest", "transform": [
                {"type": "aggregate", "groupby": ["_run_key", "_run_name", "_is_reference"]},
                {"type": "collect", "sort": {"field": ["_run_name", "_run_key"]}},
            ]},
        ],
        "scales": [{"name": "run_columns", "type": "band", "domain": {"data": "runs", "field": "_run_key"},
                    "range": "width", "paddingInner": 0.025}],
        "marks": [
            {"type": "text", "name": "instruction", "encode": {"update": {
                "x": {"value": 0}, "y": {"value": 12}, "fontSize": {"value": 12},
                "fill": text_color, "limit": {"signal": "width"}, "cursor": {"value": "pointer"},
                "text": {"signal": "focus_case < 0 ? length(data('runs')) + ' run(s) · click a case to compare' : 'case ' + focus_case + ' · click to return to all images'"}}}},
            {"type": "text", "encode": {"update": {
                "x": {"value": 0}, "y": {"value": 39}, "fontSize": {"value": 12}, "fill": text_color,
                "limit": {"signal": "max(1, slider_left - 8)"},
                "text": {"signal": "length(data('checkpoints')) ? 'Step ' + data('checkpoints')[handle_index]._checkpoint : 'Reference'"}}}},
            {"type": "rect", "encode": {"update": {
                "x": {"signal": "slider_left"}, "y": {"value": 34}, "width": {"signal": "slider_width"},
                "height": {"value": 2}, "fill": {"value": "#cccccc"},
                "opacity": {"signal": "length(data('checkpoints')) > 1 ? 1 : 0"}}}},
            {"type": "rect", "name": "checkpoint_hit", "encode": {"update": {
                "x": {"signal": "slider_left"}, "y": {"value": 24}, "width": {"signal": "slider_width"},
                "height": {"value": 22}, "fill": {"value": "transparent"}, "cursor": {"value": "ew-resize"}}}},
            {"type": "symbol", "name": "checkpoint_handle", "encode": {"update": {
                "x": {"signal": "slider_position"}, "y": {"value": 35}, "shape": {"value": "circle"},
                "size": {"value": 130}, "fill": {"value": "#357edd"}, "cursor": {"value": "ew-resize"},
                "opacity": {"signal": "length(data('checkpoints')) > 1 ? 1 : 0"}}}},
            *[
                mark
                for direction, label, right in (("previous", "−", 68), ("next", "+", 34))
                for mark in (
                    {"type": "rect", "name": f"checkpoint_{direction}", "encode": {"update": {
                        "x": {"signal": f"width - {right}"}, "y": {"value": 23},
                        "width": {"value": 28}, "height": {"value": 24}, "cornerRadius": {"value": 4},
                        "fill": {"value": "#f3f4f6"}, "stroke": {"value": "#9ca3af"},
                        "cursor": {"signal": f"{direction}_enabled ? 'pointer' : 'default'"},
                        "opacity": {"signal": f"{direction}_enabled ? 1 : 0.35"},
                        "tooltip": {"value": f"{direction.title()} checkpoint"}}}},
                    {"type": "text", "interactive": False, "encode": {"update": {
                        "x": {"signal": f"width - {right - 14}"}, "y": {"value": 35},
                        "align": {"value": "center"}, "baseline": {"value": "middle"},
                        "fontSize": {"value": 19}, "fill": {"value": "#374151"},
                        "opacity": {"signal": f"{direction}_enabled ? 1 : 0.35"},
                        "text": {"value": label}}}},
                )
            ],
            {"type": "group", "name": "run_gallery",
             "from": {"facet": {"name": "run_manifest", "data": "manifest", "groupby": ["_run_key", "_run_name", "_is_reference"]}},
             "encode": {"update": {
                 "x": {"scale": "run_columns", "field": "_run_key"}, "y": {"value": 60},
                 "width": {"signal": "bandwidth('run_columns')"}, "height": {"signal": "max(1, height - 60)"},
                 "stroke": {"value": "#dddddd"}}},
             "signals": [
                 {"name": "selected_page", "update": "length(data('page')) ? data('page')[0] : null"},
                 {"name": "page_count", "update": "selected_page ? selected_page._count : 8"},
                 {"name": "grid_rows", "update": "focus_case >= 0 ? 1 : max(1, ceil(page_count / grid_columns))"},
                 {"name": "cell_width", "update": "bandwidth('run_columns') / grid_columns"},
                 {"name": "cell_height", "update": "max(1, height - 104) / grid_rows"},
             ],
             "data": [
                 {"name": "page", "source": "run_manifest", "transform": [
                     {"type": "filter", "expr": "datum._is_reference || datum._checkpoint === selected_checkpoint"}]},
                 {"name": "signed", "url": {"signal": f"selected_page ? selected_page._page_url : '{EMPTY_URL}'"},
                  "format": {"type": "json"}, "transform": [
                      {"type": "filter", "expr": "selected_page && isString(datum.url) && indexof(datum.url, selected_page._filename) >= 0"},
                      {"type": "formula", "as": "_proxy_url", "expr": signed_url}]},
                 {"name": "pixels", "url": {"signal": f"length(data('signed')) ? data('signed')[0]._proxy_url : '{EMPTY_URL}'"},
                  "format": {"type": "json", "property": "data"}, "transform": [
                      {"type": "formula", "as": "case", "expr": "datum[0]"},
                      {"type": "formula", "as": "checkpoint", "expr": "datum[1]"},
                      {"type": "formula", "as": "seed", "expr": "datum[2]"},
                      {"type": "formula", "as": "prompt", "expr": "datum[3]"},
                      {"type": "formula", "as": "image", "expr": "datum[4]"}]},
                 {"name": "visible", "source": "pixels", "transform": [
                     {"type": "filter", "expr": "selected_page && (selected_page._is_reference || datum.checkpoint === selected_checkpoint)"},
                     {"type": "filter", "expr": "focus_case < 0 || datum.case === focus_case"}]},
             ],
             "marks": [
                 {"type": "text", "encode": {"update": {
                     "x": {"value": 4}, "y": {"value": 15}, "text": {"signal": "parent._run_name"},
                     "fontWeight": {"value": "bold"}, "fontSize": {"value": 12}, "fill": text_color,
                     "limit": {"signal": "max(1, bandwidth('run_columns') - 8)"}}}},
                 {"type": "text", "encode": {"update": {
                     "x": {"value": 4}, "y": {"value": 30}, "fontSize": {"value": 10}, "fill": text_color,
                     "text": {"signal": "(parent._is_reference ? 'reference' : 'checkpoint ' + selected_checkpoint) + ' · ' + parent._run_key"},
                     "limit": {"signal": "max(1, bandwidth('run_columns') - 8)"}}}},
                 {"type": "image", "name": "preview", "from": {"data": "visible"}, "encode": {"update": {
                     "url": {"field": "image"},
                     "x": {"signal": "(focus_case < 0 ? datum.case % grid_columns : 0) * cell_width + 4"},
                     "y": {"signal": "50 + (focus_case < 0 ? floor(datum.case / grid_columns) : 0) * cell_height"},
                     "width": {"signal": "max(1, cell_width - 8)"}, "height": {"signal": "max(1, cell_height - 26)"},
                     "cursor": {"value": "pointer"}, "tooltip": {"signal": "parent._run_name + '\\n' + datum.prompt"}}}},
                 {"type": "text", "from": {"data": "visible"}, "encode": {"update": {
                     "x": {"signal": "(focus_case < 0 ? datum.case % grid_columns : 0) * cell_width + 4"},
                     "y": {"signal": "44 + (focus_case < 0 ? floor(datum.case / grid_columns) : 0) * cell_height"},
                     "fontSize": {"value": 10}, "fill": text_color,
                     "text": {"signal": "'case ' + datum.case + ' · seed ' + datum.seed"},
                     "limit": {"signal": "max(1, cell_width - 8)"}}}},
                 {"type": "text", "encode": {"update": {
                     "x": {"value": 8}, "y": {"value": 75}, "fontSize": {"value": 11}, "fill": text_color,
                     "opacity": {"signal": "length(data('visible')) ? 0 : 1"},
                     "text": {"signal": "!selected_page ? 'No evaluation at this step' : 'Loading selected images…'"}}}},
             ]},
        ],
    }


# ============================================================================ summary-reading score panels
def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def eval_schedule(eval_interval: int, max_step: int) -> tuple[int, ...]:
    """0, interval, 2*interval, ... and the final step (the periodic evaluation's points)."""
    _positive_int(eval_interval, "eval_interval")
    _positive_int(max_step, "max_step")
    steps = tuple(range(0, max_step + 1, eval_interval))
    return steps if steps[-1] == max_step else (*steps, max_step)


def _metric(metric: str) -> str:
    from ..evaluation.protocol import METRICS

    if metric not in METRICS:
        raise ValueError(f"Unsupported evaluation metric: {metric!r}")
    return metric


def curve_key(step: int, metric: str) -> str:
    return f"eval/curve_{step}_{_metric(metric)}"


def latest_metric_key(metric: str) -> str:
    return f"eval/curve_latest_{_metric(metric)}"


def native_metric_panel_config(metric: str, *, eval_interval: int = 100, max_step: int = 2000) -> dict[str, Any]:
    """A native ``Run History Line Plot`` of the summary scalars on the checkpoint-step axis. Keys of future steps
    stay absent until scored, and the line stops at the latest completed checkpoint."""
    _metric(metric)
    steps = eval_schedule(eval_interval, max_step)
    latest_step = "${summary:" + LATEST_STEP_KEY + "}"
    latest_value = "${summary:" + latest_metric_key(metric) + "}"
    expression = "0 / 0"  # A missing or unsupported point must stay missing.
    for step in reversed(steps):
        value = "${summary:" + curve_key(step, metric) + "}"
        expression = f"(_step < {step + eval_interval + 1} ? {value} : {expression})"
    expression = f"(_step >= {latest_step} ? {latest_value} : {expression})"
    x_expression = (
        f"(_step >= {latest_step} ? {latest_step} : "
        f"max(0, floor((_step - 1) / {eval_interval}) * {eval_interval}))"
    )
    # W&B's full-fidelity renderer loads variables from the x expression only, then reuses them for y: reference the
    # score keys there through always-zero terms (absent keys leave x unchanged too).
    score_variables = [latest_value] + ["${summary:" + curve_key(step, metric) + "}" for step in steps]
    x_expression += "".join(f" + ({value} ? 0 : 0)" for value in score_variables)
    return {
        "metrics": ["_step"],
        "xAxis": "_step",
        "xExpression": x_expression,
        "expressions": [expression],
        "chartTitle": f"eval/{metric}",
        "groupBy": "None",
        "legendFields": ["run:displayName"],
        "legendTemplate": "${run:displayName}",
        "smoothingWeight": 0,
    }


def configure_native_backfill_panels(run: Any, metrics: Sequence[str], *, eval_interval: int = 100,
                                     max_step: int = 2000) -> None:
    """Register the summary-reading native panels (config only: no history, summary or step changes)."""
    metrics = tuple(metrics)
    if not metrics or len(set(metrics)) != len(metrics):
        raise ValueError("metrics must be a nonempty sequence without duplicates")
    configs = {metric: native_metric_panel_config(metric, eval_interval=eval_interval, max_step=max_step)
               for metric in metrics}
    for metric, config in configs.items():
        run._config_callback(key=("_wandb", "visualize", f"eval/{metric}"),
                             val={"panel_type": "Run History Line Plot", "panel_config": config})


def build_curve_summary(result: Mapping[str, Any], existing_summary: Mapping[str, Any], *, metrics: Sequence[str],
                        eval_interval: int = 100, max_step: int = 2000) -> dict[str, int | float]:
    """Check a ``result.json`` and return its summary update (an older checkpoint never moves ``latest`` back)."""
    steps = eval_schedule(eval_interval, max_step)
    metrics = tuple(metrics)
    if not metrics or len(set(metrics)) != len(metrics):
        raise ValueError("metrics must be a nonempty sequence without duplicates")
    for metric in metrics:
        _metric(metric)
    step = result.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step not in steps:
        raise ValueError(f"checkpoint step {step!r} is not one of the evaluation points {list(steps)}")
    count = result.get("num_images")
    _positive_int(count, "num_images")
    values: dict[str, float] = {}
    scored = result.get("results")
    if not isinstance(scored, Mapping):
        raise ValueError("Missing evaluation results")
    for metric in metrics:
        stats = scored.get(metric)
        mean = stats.get("mean") if isinstance(stats, Mapping) else None
        if not isinstance(stats, Mapping) or stats.get("n") != count:
            raise ValueError(f"Incomplete scored sample count for {metric}")
        if isinstance(mean, bool) or not isinstance(mean, Real) or not math.isfinite(mean):
            raise ValueError(f"Missing/nonfinite completed score for {metric}")
        values[metric] = float(mean)
    previous = existing_summary.get(LATEST_STEP_KEY, -1)
    if isinstance(previous, bool) or not isinstance(previous, int) or (previous != -1 and previous not in steps):
        raise ValueError("Existing latest checkpoint has an incompatible schedule")
    update: dict[str, int | float] = {curve_key(step, metric): value for metric, value in values.items()}
    if step >= previous:
        update[LATEST_STEP_KEY] = step
        update.update({latest_metric_key(metric): value for metric, value in values.items()})
    return update


def register_preset(entity: str, *, storage_host: str = "", wandb_module: Any = None) -> Any:
    """Register the private gallery preset as ``<entity>/<CHART_NAME>``; the periodic evaluation publishes the
    gallery once ``eval.inline_chart`` names it."""
    if wandb_module is None:
        import wandb as wandb_module
    return wandb_module.Api().create_custom_chart(entity=entity, name=CHART_NAME,
                                                  display_name="Fixed evaluation images", spec_type="vega2",
                                                  access="private",
                                                  spec=paged_image_grid_spec(storage_host=storage_host))


__all__ = ["CHART_NAME", "COLUMNS", "LATEST_STEP_KEY", "build_curve_summary",
           "configure_native_backfill_panels", "curve_key", "eval_schedule", "history_preview_rows",
           "latest_metric_key", "native_metric_panel_config", "paged_history_chart", "paged_image_grid_spec",
           "preview_rows", "publish_summary_chart", "register_preset"]

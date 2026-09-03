"""CPU-only FastAPI for comparing registered exact-span eval runs."""

from __future__ import annotations

from collections.abc import Callable
from html import escape
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel, ConfigDict

from uzbek_ner.evaldash.bands import F1_BANDS, band_for_f1, scale_payload
from uzbek_ner.evaldash.prometheus import render_prometheus
from uzbek_ner.evaldash.registry import get_run, load_runs, resolve_runs_dir
from uzbek_ner.evaldash.schema import EvalRun

SORT_KEYS: dict[str, Callable[[EvalRun], Any]] = {
    "micro.f1": lambda run: run.metrics.micro.f1,
    "micro.precision": lambda run: run.metrics.micro.precision,
    "micro.recall": lambda run: run.metrics.micro.recall,
    "macro.f1": lambda run: run.metrics.macro.f1,
    "created_at": lambda run: run.created_at,
    "run_id": lambda run: run.run_id,
    "model": lambda run: run.model,
}


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str = "ok"


class RunsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    runs: list[EvalRun]


def _runs_dir_from_app(request: Request) -> Path:
    override = getattr(request.app.state, "runs_dir", None)
    if isinstance(override, Path):
        return override
    return resolve_runs_dir()


def create_app(runs_dir: Path | None = None) -> FastAPI:
    """Application factory. Tests pass a fixture directory; uvicorn uses env/default."""

    application = FastAPI(
        title="Uzbek NER eval comparison",
        version="0.1.0",
        description=(
            "Local dashboard API over official exact-span metrics. "
            "Visual layer on top of MLflow — not a replacement for experiment logging."
        ),
    )
    application.state.runs_dir = runs_dir
    application.add_api_route("/", index, methods=["GET"], include_in_schema=False)
    application.add_api_route("/healthz", healthz, methods=["GET"], response_model=HealthResponse)
    application.add_api_route(
        "/api/v1/runs", list_runs, methods=["GET"], response_model=RunsResponse
    )
    application.add_api_route(
        "/api/v1/runs/{run_id}",
        get_run_detail,
        methods=["GET"],
        response_model=EvalRun,
    )
    application.add_api_route("/api/v1/scale", scale, methods=["GET"])
    application.add_api_route("/metrics", prometheus_metrics, methods=["GET"])
    return application


async def healthz() -> HealthResponse:
    return HealthResponse(status="ok")


async def list_runs(
    request: Request,
    sort: str = Query(default="micro.f1"),
    order: str = Query(default="desc"),
) -> RunsResponse:
    if sort not in SORT_KEYS:
        allowed = ", ".join(sorted(SORT_KEYS))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"unknown sort {sort!r}; allowed: {allowed}",
        )
    descending = order.lower() not in {"asc", "ascending"}
    runs = load_runs(_runs_dir_from_app(request))
    runs.sort(key=SORT_KEYS[sort], reverse=descending)
    return RunsResponse(runs=runs)


async def get_run_detail(request: Request, run_id: str) -> EvalRun:
    run = get_run(run_id, _runs_dir_from_app(request))
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"run {run_id!r} not found"
        )
    return run


async def scale() -> dict[str, object]:
    return scale_payload()


async def prometheus_metrics(request: Request) -> PlainTextResponse:
    body = render_prometheus(load_runs(_runs_dir_from_app(request)))
    return PlainTextResponse(
        body,
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


def _render_index(runs: list[EvalRun]) -> str:
    ranked = sorted(runs, key=lambda run: run.metrics.micro.f1, reverse=True)
    rows = []
    for run in ranked:
        micro = run.metrics.micro
        band = band_for_f1(micro.f1)
        org = run.metrics.by_label["ORG"].f1
        name = run.metrics.by_label["NAME"].f1
        geo = run.metrics.by_label["GEO"].f1
        f1_cell = f"<td class='f1' style='background:{band.color};color:#fff'>{micro.f1:.4f}</td>"
        diag = run.diagnostics
        if diag is None:
            bound_cell = "<td>—</td>"
            type_cell = "<td>—</td>"
            bucket_cell = "<td>—</td>"
            reading_cell = "<td class='muted'>нет разбора</td>"
        else:
            bound_cell = f"<td>{diag.boundary_exact.f1:.4f}</td>"
            type_cell = f"<td>{diag.type_given_boundary.accuracy:.3f}</td>"
            buckets = diag.buckets
            bucket_cell = (
                f"<td><code>ok {buckets.exact_match}</code> "
                f"<code>тип {buckets.type_mismatch}</code> "
                f"<code>границы {buckets.partial_same_type + buckets.partial_diff_type}</code> "
                f"<code>miss {buckets.missed}</code> "
                f"<code>fp {buckets.spurious}</code></td>"
            )
            reading_cell = f"<td>{escape(diag.reading)}</td>"
        rows.append(
            "<tr>"
            f"<td><a href='/api/v1/runs/{escape(run.run_id)}'>{escape(run.run_id)}</a></td>"
            f"<td>{escape(run.model)}</td>"
            f"<td>{escape(run.split)}</td>"
            f"{f1_cell}"
            f"<td>{escape(band.title)}</td>"
            f"<td>{micro.precision:.4f}</td>"
            f"<td>{micro.recall:.4f}</td>"
            f"<td>{org:.4f}</td>"
            f"<td>{name:.4f}</td>"
            f"<td>{geo:.4f}</td>"
            f"{bound_cell}"
            f"{type_cell}"
            f"{bucket_cell}"
            f"{reading_cell}"
            f"<td>{escape(run.created_at)}</td>"
            "</tr>"
        )
    body = (
        "\n".join(rows) if rows else "<tr><td colspan='15'>No runs in the registry yet.</td></tr>"
    )
    legend_items = []
    for band in F1_BANDS:
        hi = "1.00" if band.max_exclusive > 1 else f"{band.max_exclusive:.2f}"
        legend_items.append(
            "<div class='band'>"
            f"<span class='swatch' style='background:{band.color}'></span>"
            f"<strong>{escape(band.title)}</strong>"
            f" <code>{band.min_inclusive:.2f}-{hi}</code>"
            f"<p>{escape(band.meaning)}</p>"
            "</div>"
        )
    legend = "\n".join(legend_items)
    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="utf-8"/>
  <title>Uzbek NER eval comparison</title>
  <style>
    body {{ font-family: sans-serif; margin: 2rem; color: #111; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ccc; padding: 0.4rem 0.6rem; text-align: left; }}
    th {{ background: #f4f4f4; }}
    tbody tr:nth-child(even) {{ background: #fafafa; }}
    .muted {{ color: #888; }}
    td {{ font-variant-numeric: tabular-nums; }}
    code {{ background: #f4f4f4; padding: 0.1rem 0.3rem; }}
    .note {{ color: #444; max-width: 70rem; }}
    .scale {{ display: grid; gap: 0.75rem; max-width: 70rem; margin: 1.25rem 0 1.75rem; }}
    .band p {{ margin: 0.25rem 0 0; color: #333; }}
    .swatch {{
      display: inline-block; width: 0.85rem; height: 0.85rem;
      margin-right: 0.4rem; vertical-align: -0.1rem; border-radius: 2px;
    }}
  </style>
</head>
<body>
  <h1>Uzbek NER — official exact-span comparison</h1>
  <p class="note">
    Visual layer on top of MLflow. Scores use organizer matching
    (same hash and exact label/start/end) for ORG / NAME / GEO.
    Grafana: <a href="http://127.0.0.1:3000">http://127.0.0.1:3000</a>
    (admin/admin). Prometheus scrape: <a href="/metrics">/metrics</a>.
    Шкала: <a href="/api/v1/scale">/api/v1/scale</a>.
    Лидерборд — только strict exact-span; колонки границ/типа — диагностика.
  </p>
  <h2>Как читать micro-F1</h2>
  <p class="note">
    Это не CoNLL-English. Частичных совпадений нет: суффикс вне спана = полный промах.
    <strong>≈0.5 — рабочий бейзлайн, не продукт. ≈0.7 — уже хорошо для этой метрики.
    ≥0.85 — редко, сначала проверь скорер.</strong>
    Жёсткого порога у организаторов нет.
  </p>
  <div class="scale">
    {legend}
  </div>
  <p class="note">
    <strong>Диагностика (не метрика авторов):</strong>
    <em>границы F1</em> — совпал только <code>(start, end)</code>, тип не важен.
    Если он заметно выше strict F1, модель чует спаны, но путает ORG/NAME/GEO.
    <em>тип @ спан</em> — доля верных лейблов среди спанов с точными координатами.
    Корзины: ok = полный hit авторов; тип = те же границы, другой класс;
    границы = пересечение, но не exact; miss/fp = нет пересечения.
  </p>
  <p><a href="/api/v1/runs">JSON runs</a> · <a href="/healthz">healthz</a></p>
  <table>
    <thead>
      <tr>
        <th>run_id</th><th>model</th><th>split</th>
        <th>micro F1</th><th>шкала</th><th>P</th><th>R</th>
        <th>ORG F1</th><th>NAME F1</th><th>GEO F1</th>
        <th>границы F1</th><th>тип @ спан</th><th>корзины</th><th>разбор</th>
        <th>created_at</th>
      </tr>
    </thead>
    <tbody>
      {body}
    </tbody>
  </table>
</body>
</html>
"""


async def index(request: Request) -> HTMLResponse:
    runs = load_runs(_runs_dir_from_app(request))
    return HTMLResponse(_render_index(runs))


app = create_app()

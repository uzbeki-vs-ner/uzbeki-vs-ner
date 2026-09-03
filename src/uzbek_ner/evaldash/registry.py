"""Load / write eval run JSON files under ``outputs/eval/runs/``."""

from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import ValidationError

from uzbek_ner.evaldash.schema import EvalPaths, EvalRun, ExactSpanMetrics, SpanDiagnostics
from uzbek_ner.metrics.error_analysis import analyze_span_errors
from uzbek_ner.metrics.exact_span import (
    calculate_exact_span_metrics,
    load_gold_and_predictions,
    write_metrics,
)
from uzbek_ner.settings import REPO_ROOT

DEFAULT_RUNS_DIR = REPO_ROOT / "outputs" / "eval" / "runs"
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")


def resolve_runs_dir(override: Path | None = None) -> Path:
    """Directory of ``{run_id}.json`` files. Env: ``EVAL_RUNS_DIR``."""

    if override is not None:
        return override
    env = os.environ.get("EVAL_RUNS_DIR")
    if env:
        return Path(env)
    return DEFAULT_RUNS_DIR


def validate_run_id(run_id: str) -> str:
    if not _RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(f"invalid run_id {run_id!r}: use letters, digits, dot, underscore, hyphen")
    return run_id


def run_path(run_id: str, directory: Path | None = None) -> Path:
    return resolve_runs_dir(directory) / f"{validate_run_id(run_id)}.json"


def load_runs(directory: Path | None = None) -> list[EvalRun]:
    """Parse every ``*.json`` run file. Invalid files are skipped with a warning."""

    root = resolve_runs_dir(directory)
    if not root.is_dir():
        return []
    runs: list[EvalRun] = []
    for path in sorted(root.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            runs.append(EvalRun.model_validate(payload))
        except (OSError, json.JSONDecodeError, ValidationError) as exc:
            logger.warning("skipping eval run {}: {}", path, exc)
    return runs


def get_run(run_id: str, directory: Path | None = None) -> EvalRun | None:
    path = run_path(run_id, directory)
    if not path.is_file():
        return None
    try:
        return EvalRun.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as exc:
        logger.warning("invalid eval run {}: {}", path, exc)
        return None


def write_run(run: EvalRun, directory: Path | None = None) -> Path:
    """Serialize ``run`` to ``{runs_dir}/{run_id}.json``, overwriting if present."""

    path = run_path(run.run_id, directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(run.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _diagnostics_from_files(gold: Path | str, predictions: Path | str) -> SpanDiagnostics | None:
    gold_path = Path(gold)
    pred_path = Path(predictions)
    if not gold_path.is_file() or not pred_path.is_file():
        return None
    gold_records, pred_records = load_gold_and_predictions(gold_path, pred_path)
    return SpanDiagnostics.model_validate(analyze_span_errors(gold_records, pred_records))


def build_run(
    *,
    run_id: str,
    model: str,
    metrics: ExactSpanMetrics | dict[str, Any],
    checkpoint: str = "",
    split: str = "official_dev",
    hyperparams: dict[str, Any] | None = None,
    paths: EvalPaths | dict[str, str] | None = None,
    created_at: str | None = None,
    diagnostics: SpanDiagnostics | dict[str, Any] | None = None,
) -> EvalRun:
    payload = (
        metrics
        if isinstance(metrics, ExactSpanMetrics)
        else ExactSpanMetrics.model_validate(metrics)
    )
    path_model = paths if isinstance(paths, EvalPaths) else EvalPaths.model_validate(paths or {})
    diag_model: SpanDiagnostics | None
    if diagnostics is None:
        diag_model = None
    elif isinstance(diagnostics, SpanDiagnostics):
        diag_model = diagnostics
    else:
        diag_model = SpanDiagnostics.model_validate(diagnostics)
    return EvalRun(
        run_id=validate_run_id(run_id),
        model=model,
        checkpoint=checkpoint,
        created_at=created_at or datetime.now(UTC).isoformat(),
        split=split,
        hyperparams=hyperparams or {},
        metrics=payload,
        diagnostics=diag_model,
        paths=path_model,
    )


def ingest_metrics(
    metrics_path: Path,
    *,
    run_id: str,
    model: str,
    checkpoint: str = "",
    split: str = "official_dev",
    hyperparams: dict[str, Any] | None = None,
    gold: Path | str = "",
    predictions: Path | str = "",
    created_at: str | None = None,
    directory: Path | None = None,
) -> EvalRun:
    """Wrap an existing metrics JSON (from ``write_metrics``) and save a run file."""

    raw = json.loads(metrics_path.read_text(encoding="utf-8"))
    run = build_run(
        run_id=run_id,
        model=model,
        metrics=raw,
        checkpoint=checkpoint,
        split=split,
        hyperparams=hyperparams,
        created_at=created_at,
        diagnostics=_diagnostics_from_files(gold, predictions),
        paths=EvalPaths(
            gold=str(gold),
            predictions=str(predictions),
            metrics=str(metrics_path),
        ),
    )
    write_run(run, directory)
    return run


def evaluate_and_register(
    gold: Path,
    predictions: Path,
    *,
    run_id: str,
    model: str,
    metrics_out: Path,
    checkpoint: str = "",
    split: str = "official_dev",
    hyperparams: dict[str, Any] | None = None,
    created_at: str | None = None,
    directory: Path | None = None,
) -> EvalRun:
    """Score gold vs predictions with the official scorer, then register the run."""

    gold_records, pred_records = load_gold_and_predictions(gold, predictions)
    metrics = calculate_exact_span_metrics(gold_records, pred_records)
    write_metrics(metrics_out, metrics)
    run = build_run(
        run_id=run_id,
        model=model,
        metrics=metrics,
        checkpoint=checkpoint,
        split=split,
        hyperparams=hyperparams,
        created_at=created_at,
        diagnostics=analyze_span_errors(gold_records, pred_records),
        paths=EvalPaths(
            gold=str(gold),
            predictions=str(predictions),
            metrics=str(metrics_out),
        ),
    )
    write_run(run, directory)
    return run

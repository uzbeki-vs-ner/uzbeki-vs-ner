"""MLflow helpers — единый sqlite-backend для экспериментов."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import mlflow

from uzbek_ner.settings import Settings, get_settings

RESEARCH_EXPERIMENT = "uzbek_ner"
SMOKE_EXPERIMENT = "uzbek_ner_smoke"


def setup_experiment(
    experiment_name: str,
    *,
    run_name: str | None = None,
    tags: dict[str, Any] | None = None,
    settings: Settings | None = None,
) -> mlflow.ActiveRun:
    cfg = settings or get_settings()
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)
    return mlflow.start_run(run_name=run_name, tags=tags or {})


def log_params(params: dict[str, Any]) -> None:
    mlflow.log_params({k: str(v) for k, v in params.items()})


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    mlflow.log_metrics(metrics, step=step)


def log_json(data: dict[str, Any], artifact_dir: str, filename: str) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / filename
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        mlflow.log_artifact(str(path), artifact_dir)


def log_artifact(path: str | Path, artifact_dir: str | None = None) -> None:
    file_path = Path(path)
    if file_path.exists():
        mlflow.log_artifact(str(file_path), artifact_dir)

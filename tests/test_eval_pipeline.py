"""File-based evaluate stage (CPU) and register CLI — no GPU / torch."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from uzbek_ner.pipeline.evaluate import run_evaluate
from uzbek_ner.settings import REPO_ROOT, Settings

GOLD = REPO_ROOT / "tests" / "fixtures" / "official_metrics" / "gold.jsonl"
PRED = REPO_ROOT / "tests" / "fixtures" / "official_metrics" / "predictions.jsonl"


def _load_config() -> DictConfig:
    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base=None):
        return compose(config_name="default")


def test_evaluate_scores_fixture_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setenv("EVAL_RUNS_DIR", str(tmp_path / "runs"))
    cfg = _load_config()
    cfg.evaluate.gold = str(GOLD)
    cfg.evaluate.predictions = str(PRED)
    cfg.evaluate.metrics_out = str(tmp_path / "span_metrics.json")
    cfg.evaluate.run_id = "pipeline_fixture"
    settings = Settings(metrics_path=tmp_path / "metrics.json", checkpoints=tmp_path / "ckpt")
    dvc_path = run_evaluate(cfg, settings=settings)
    payload = json.loads(dvc_path.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert payload["micro_f1"] == 1.0
    run_file = tmp_path / "runs" / "pipeline_fixture.json"
    assert run_file.is_file()
    registered = json.loads(run_file.read_text(encoding="utf-8"))
    assert registered["metrics"]["micro"]["f1"] == 1.0


def test_evaluate_waiting_without_predictions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    cfg = _load_config()
    cfg.evaluate.gold = str(tmp_path / "missing_gold.jsonl")
    cfg.evaluate.predictions = str(tmp_path / "missing_pred.jsonl")
    settings = Settings(metrics_path=tmp_path / "metrics.json")
    dvc_path = run_evaluate(cfg, settings=settings)
    payload = json.loads(dvc_path.read_text(encoding="utf-8"))
    assert payload["status"] == "waiting_for_predictions"
    assert payload["micro_f1"] == 0.0


def test_register_script_from_gold_and_predictions(tmp_path: Path) -> None:
    script = REPO_ROOT / "scripts" / "register_eval_run.py"
    dest = tmp_path / "runs"
    metrics_out = tmp_path / "out.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--run-id",
            "cli_fixture",
            "--model",
            "fixture-model",
            "--gold",
            str(GOLD),
            "--predictions",
            str(PRED),
            "--metrics",
            str(metrics_out),
            "--runs-dir",
            str(dest),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    run_file = dest / "cli_fixture.json"
    assert run_file.is_file()
    payload = json.loads(run_file.read_text(encoding="utf-8"))
    assert payload["metrics"]["micro"]["f1"] == 1.0
    assert metrics_out.is_file()

"""Smoke tests for pipeline stubs (no dataset required)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from uzbek_ner.pipeline.prepare import run_prepare
from uzbek_ner.settings import REPO_ROOT, Settings


def _load_config() -> DictConfig:
    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base=None):
        return compose(config_name="default")


def _isolated_settings(tmp_path: Path, official: Path | None = None) -> Settings:
    return Settings(
        data_official=official if official is not None else tmp_path / "official",
        data_raw=official if official is not None else tmp_path / "official",
        data_processed=tmp_path / "processed",
        checkpoints=tmp_path / "checkpoints",
        models=tmp_path / "models",
        metrics_path=tmp_path / "metrics.json",
    )


def test_prepare_stub_writes_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    settings = _isolated_settings(tmp_path)
    cfg = _load_config()
    manifest = run_prepare(cfg, settings=settings)
    assert manifest.exists()
    assert manifest.name == "manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["status"] == "waiting_for_data"


def test_prepare_reads_official_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    official = tmp_path / "official"
    official.mkdir()
    (official / "train.jsonl").write_text("{}\n", encoding="utf-8")
    (official / "dev.jsonl").write_text("{}\n", encoding="utf-8")
    (official / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "splits": {"train": {"records": 2}, "dev": {"records": 1}},
                "schema": {"labels": ["ORG", "NAME", "GEO"]},
            }
        ),
        encoding="utf-8",
    )
    settings = _isolated_settings(tmp_path, official=official)
    manifest = run_prepare(_load_config(), settings=settings)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["status"] == "ready"
    assert payload["stats"]["train_records"] == 2
    assert payload["stats"]["dev_records"] == 1

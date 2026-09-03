"""Offline registry tests for eval-run JSON (no GPU, no torch)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from uzbek_ner.evaldash.prometheus import latest_run, render_prometheus
from uzbek_ner.evaldash.registry import (
    evaluate_and_register,
    ingest_metrics,
    load_runs,
    validate_run_id,
    write_run,
)
from uzbek_ner.evaldash.schema import EvalRun
from uzbek_ner.metrics.exact_span import evaluate_prediction_files

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_RUNS = REPO_ROOT / "tests" / "fixtures" / "eval" / "runs"
GOLD = REPO_ROOT / "tests" / "fixtures" / "official_metrics" / "gold.jsonl"
PRED = REPO_ROOT / "tests" / "fixtures" / "official_metrics" / "predictions.jsonl"


def test_load_fixture_runs_sorted_by_f1() -> None:
    runs = load_runs(FIXTURE_RUNS)
    assert {run.run_id for run in runs} == {"gazetteer_stub", "uztext_smoke", "uztext_full"}
    ranked = sorted(runs, key=lambda run: run.metrics.micro.f1, reverse=True)
    assert [run.run_id for run in ranked] == ["uztext_full", "uztext_smoke", "gazetteer_stub"]
    assert ranked[0].metrics.micro.f1 > ranked[1].metrics.micro.f1


def test_latest_run_is_smoke_not_best_f1() -> None:
    runs = load_runs(FIXTURE_RUNS)
    newest = latest_run(runs)
    assert newest is not None
    assert newest.run_id == "uztext_smoke"
    best = max(runs, key=lambda run: run.metrics.micro.f1)
    assert best.run_id == "uztext_full"


def test_validate_run_id_rejects_paths() -> None:
    with pytest.raises(ValueError, match="invalid run_id"):
        validate_run_id("../escape")
    with pytest.raises(ValueError, match="invalid run_id"):
        validate_run_id("a/b")
    assert validate_run_id("uztext_smoke") == "uztext_smoke"


def test_write_and_overwrite_run(tmp_path: Path) -> None:
    original = load_runs(FIXTURE_RUNS)[0]
    path = write_run(original, tmp_path)
    assert path.name == f"{original.run_id}.json"
    loaded = load_runs(tmp_path)
    assert len(loaded) == 1
    assert loaded[0].run_id == original.run_id

    updated = original.model_copy(deep=True)
    updated.metrics.micro.f1 = 0.99
    write_run(updated, tmp_path)
    again = load_runs(tmp_path)
    assert len(again) == 1
    assert again[0].metrics.micro.f1 == 0.99


def test_load_runs_skips_corrupt_json(tmp_path: Path) -> None:
    shutil.copy(FIXTURE_RUNS / "uztext_smoke.json", tmp_path / "uztext_smoke.json")
    (tmp_path / "broken.json").write_text("{not-json", encoding="utf-8")
    (tmp_path / "empty.json").write_text("{}\n", encoding="utf-8")
    runs = load_runs(tmp_path)
    assert [run.run_id for run in runs] == ["uztext_smoke"]


def test_ingest_metrics_wraps_scorer_output(tmp_path: Path) -> None:
    metrics = evaluate_prediction_files(GOLD, PRED)
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    run = ingest_metrics(
        metrics_path,
        run_id="fixture_perfect",
        model="fixture",
        gold=GOLD,
        predictions=PRED,
        directory=tmp_path / "runs",
    )
    assert run.metrics.micro.f1 == 1.0
    assert run.diagnostics is not None
    assert run.diagnostics.reading_id in {"healthy", "mixed"}
    assert (tmp_path / "runs" / "fixture_perfect.json").is_file()


def test_evaluate_and_register_uses_official_scorer(tmp_path: Path) -> None:
    run = evaluate_and_register(
        GOLD,
        PRED,
        run_id="from_files",
        model="fixture",
        metrics_out=tmp_path / "scored.json",
        directory=tmp_path / "runs",
    )
    assert run.metrics.records == 2
    assert run.metrics.micro.f1 == 1.0
    assert run.metrics.by_label["ORG"].f1 == 1.0
    assert run.diagnostics is not None
    assert run.diagnostics.buckets.exact_match == 3
    assert run.diagnostics.type_given_boundary.accuracy == 1.0
    saved = EvalRun.model_validate_json((tmp_path / "runs" / "from_files.json").read_text())
    assert saved.paths.metrics.endswith("scored.json")


def test_prometheus_labels_include_run_model_split() -> None:
    text = render_prometheus(load_runs(FIXTURE_RUNS))
    assert "ner_eval_f1{" in text
    assert 'run_id="uztext_full"' in text
    assert 'model="rifkat/uztext-3Gb-BPE-Roberta"' in text
    assert 'split="official_dev"' in text
    assert 'label="ORG"' in text
    assert 'label="micro"' in text
    assert "ner_eval_latest{" in text
    latest_lines = [
        line
        for line in text.splitlines()
        if line.startswith("ner_eval_latest{") and 'run_id="uztext_smoke"' in line
    ]
    assert latest_lines and latest_lines[0].endswith(" 1")
    assert text.endswith("\n")

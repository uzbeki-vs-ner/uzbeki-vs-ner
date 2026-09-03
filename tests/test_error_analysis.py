"""Type-vs-boundary diagnostics (not the organizer exact-span score)."""

from __future__ import annotations

import json
from pathlib import Path

from uzbek_ner.metrics.error_analysis import analyze_prediction_files, analyze_span_errors
from uzbek_ner.metrics.exact_span import evaluate_prediction_files, load_gold_and_predictions

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLD = REPO_ROOT / "tests" / "fixtures" / "official_metrics" / "gold.jsonl"
PRED = REPO_ROOT / "tests" / "fixtures" / "official_metrics" / "predictions.jsonl"


def _write_pair(tmp_path: Path, gold_rows: list[dict], pred_rows: list[dict]) -> tuple[Path, Path]:
    gold_path = tmp_path / "gold.jsonl"
    pred_path = tmp_path / "pred.jsonl"
    gold_path.write_text("".join(json.dumps(row) + "\n" for row in gold_rows), encoding="utf-8")
    pred_path.write_text("".join(json.dumps(row) + "\n" for row in pred_rows), encoding="utf-8")
    return gold_path, pred_path


def test_perfect_official_fixture_is_all_exact_match() -> None:
    gold, predictions = load_gold_and_predictions(GOLD, PRED)
    official = evaluate_prediction_files(GOLD, PRED)
    diag = analyze_span_errors(gold, predictions)
    assert official["micro"]["f1"] == 1.0
    assert diag["buckets"]["exact_match"] == official["micro"]["tp"]
    assert diag["buckets"]["type_mismatch"] == 0
    assert diag["type_given_boundary"]["accuracy"] == 1.0
    assert diag["boundary_exact"]["f1"] == 1.0


def test_swapped_labels_are_type_confusion_not_boundary(tmp_path: Path) -> None:
    gold_path, pred_path = _write_pair(
        tmp_path,
        [
            {
                "hash": "a",
                "text": "Ali Toshkent",
                "entities": [
                    {"label": "NAME", "start": 0, "end": 3},
                    {"label": "GEO", "start": 4, "end": 12},
                ],
            }
        ],
        [
            {
                "hash": "a",
                "entities": [
                    {"label": "GEO", "start": 0, "end": 3},
                    {"label": "NAME", "start": 4, "end": 12},
                ],
            }
        ],
    )
    official = evaluate_prediction_files(gold_path, pred_path)
    diag = analyze_prediction_files(gold_path, pred_path)
    assert official["micro"]["f1"] == 0.0
    assert diag["boundary_exact"]["f1"] == 1.0
    assert diag["type_given_boundary"]["accuracy"] == 0.0
    assert diag["buckets"]["type_mismatch"] == 2
    assert diag["buckets"]["exact_match"] == 0
    assert diag["type_given_boundary"]["confusion"]["NAME"]["GEO"] == 1
    assert diag["type_given_boundary"]["confusion"]["GEO"]["NAME"] == 1
    assert diag["reading_id"] == "type_confusion"


def test_suffix_trim_is_boundary_jitter(tmp_path: Path) -> None:
    gold_path, pred_path = _write_pair(
        tmp_path,
        [
            {
                "hash": "b",
                "text": "Toshkentda",
                "entities": [{"label": "GEO", "start": 0, "end": 10}],
            }
        ],
        [{"hash": "b", "entities": [{"label": "GEO", "start": 0, "end": 8}]}],
    )
    official = evaluate_prediction_files(gold_path, pred_path)
    diag = analyze_prediction_files(gold_path, pred_path)
    assert official["micro"]["tp"] == 0
    assert diag["boundary_exact"]["f1"] == 0.0
    assert diag["buckets"]["partial_same_type"] == 1
    assert diag["buckets"]["type_mismatch"] == 0
    assert diag["reading_id"] == "boundary_jitter"


def test_miss_and_spurious_are_detection_gap(tmp_path: Path) -> None:
    gold_path, pred_path = _write_pair(
        tmp_path,
        [
            {
                "hash": "c",
                "text": "Ali va Spendrups",
                "entities": [
                    {"label": "NAME", "start": 0, "end": 3},
                    {"label": "ORG", "start": 7, "end": 16},
                ],
            }
        ],
        [{"hash": "c", "entities": [{"label": "GEO", "start": 4, "end": 6}]}],
    )
    diag = analyze_prediction_files(gold_path, pred_path)
    assert diag["buckets"]["missed"] == 2
    assert diag["buckets"]["spurious"] == 1
    assert diag["buckets"]["exact_match"] == 0
    assert diag["reading_id"] == "detection_gap"

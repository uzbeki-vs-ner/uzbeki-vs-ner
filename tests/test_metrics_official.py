"""Ensure our exact-span scorer matches organizer scripts.

CI stays offline: fixture-only scoring always runs. Comparison against
the organizer `evaluate.py` is skipped when the official bundle is absent.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from uzbek_ner.metrics.exact_span import evaluate_prediction_files

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "official_metrics"
GOLD = FIXTURES / "gold.jsonl"
PRED = FIXTURES / "predictions.jsonl"
OFFICIAL_EVALUATE = (
    REPO_ROOT / "data" / "official" / "bundle" / "ner_uz_hackathon_participant" / "scripts"
)


def test_exact_span_metrics_on_fixtures() -> None:
    ours = evaluate_prediction_files(GOLD, PRED)
    assert ours["records"] == 2
    assert ours["micro"]["tp"] == 3
    assert ours["micro"]["fp"] == 0
    assert ours["micro"]["fn"] == 0
    assert ours["micro"]["f1"] == 1.0
    assert ours["by_label"]["ORG"]["f1"] == 1.0
    assert ours["by_label"]["NAME"]["f1"] == 1.0
    assert ours["by_label"]["GEO"]["f1"] == 1.0


@pytest.mark.skipif(
    not (OFFICIAL_EVALUATE / "evaluate.py").is_file(),
    reason="official evaluate.py is not checked out (CI stays offline)",
)
def test_exact_span_metrics_match_official_script(tmp_path: Path) -> None:
    ours = evaluate_prediction_files(GOLD, PRED)
    out_json = tmp_path / "official_metrics.json"
    proc = subprocess.run(
        [
            sys.executable,
            "evaluate.py",
            "--gold",
            str(GOLD.resolve()),
            "--predictions",
            str(PRED.resolve()),
            "--output",
            str(out_json),
        ],
        cwd=OFFICIAL_EVALUATE,
        check=True,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    official = json.loads(out_json.read_text(encoding="utf-8"))
    assert ours["micro"]["f1"] == official["micro"]["f1"]
    assert ours["by_label"]["ORG"]["f1"] == official["by_label"]["ORG"]["f1"]

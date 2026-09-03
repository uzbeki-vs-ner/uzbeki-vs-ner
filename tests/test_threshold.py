"""CPU tests for the entity-confidence gate (no GPU / no checkpoint)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from uzbek_ner.decode.threshold import gold_lookup, sweep_confidence_gate
from uzbek_ner.labels import TAG_TO_ID
from uzbek_ner.modeling.eval_cache import MergedDoc, pack_merged
from uzbek_ner.pipeline.calibrate import run_calibrate
from uzbek_ner.settings import REPO_ROOT, Settings

N_LABELS = 7


def _doc(
    record_hash: str,
    text: str,
    offsets: list[tuple[int, int]],
    tag: str,
    logit: float,
    gold: set[tuple[str, int, int]],
) -> MergedDoc:
    n = len(offsets)
    logits = np.zeros((n, N_LABELS), dtype=np.float32)
    logits[:, TAG_TO_ID[tag]] = logit
    gold_labels = np.zeros((n,), dtype=np.int64)
    return MergedDoc(
        record_hash=record_hash,
        text=text,
        offsets=np.asarray(offsets, dtype=np.int32),
        logits=logits,
        counts=np.ones((n,), dtype=np.int32),
        gold_labels=gold_labels,
        gold_keys=gold,
    )


def _records_from_docs(docs: list[MergedDoc]) -> list[dict[str, object]]:
    return [
        {
            "hash": doc.record_hash,
            "text": doc.text,
            "entities": [
                {"label": label, "start": start, "end": end} for label, start, end in doc.gold_keys
            ],
        }
        for doc in docs
    ]


def _load_config() -> DictConfig:
    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base=None):
        return compose(config_name="default")


def test_sweep_picks_tau_that_drops_low_conf_spurious() -> None:
    keep = _doc("keep", "Ali", [(0, 3)], "B-NAME", 8.0, {("NAME", 0, 3)})
    junk = _doc("junk", "zz", [(0, 2)], "B-ORG", 0.2, set())
    docs = [keep, junk]
    gold = gold_lookup(_records_from_docs(docs))
    sweep = sweep_confidence_gate(
        docs,
        gold,
        analysis_hashes=["keep", "junk"],
        held_out_hashes=["keep", "junk"],
        taus=(0.0, 0.5),
    )
    assert sweep["baseline_held_out"]["fp"] == 1
    assert sweep["selected_tau"] == 0.5
    assert sweep["held_out"]["fp"] == 0
    assert sweep["held_out"]["tp"] == 1
    assert sweep["held_out_delta_f1"] > 0


def test_calibrate_waits_without_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    cfg = _load_config()
    gold = tmp_path / "dev.jsonl"
    gold.write_text(
        json.dumps({"hash": "h", "text": "Ali", "entities": []}) + "\n",
        encoding="utf-8",
    )
    cfg.calibrate.gold = str(gold)
    cfg.calibrate.checkpoint = str(tmp_path / "missing_ckpt")
    cfg.calibrate.cache_dir = str(tmp_path / "cache")
    cfg.calibrate.output = str(tmp_path / "threshold.json")
    cfg.calibrate.metrics_out = str(tmp_path / "threshold_metrics.json")
    cfg.calibrate.fill_cache = False
    settings = Settings(metrics_path=tmp_path / "metrics.json")
    path = run_calibrate(cfg, settings=settings)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "waiting_for_cache"
    assert payload["selected_tau"] == 0.0


def test_calibrate_sweeps_from_merged_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    keep = _doc("keep", "Ali", [(0, 3)], "B-NAME", 8.0, {("NAME", 0, 3)})
    junk = _doc("junk", "zz", [(0, 2)], "B-ORG", 0.2, set())
    docs = [keep, junk]
    records = _records_from_docs(docs)
    gold_path = tmp_path / "dev.jsonl"
    gold_path.write_text(
        "".join(json.dumps(row) + "\n" for row in records),
        encoding="utf-8",
    )
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    packed = pack_merged(docs)
    np.savez(cache_dir / "merged_mean_logits.npz", **packed)
    (cache_dir / "index.json").write_text(
        json.dumps({"status": "complete", "files": {"merged": "merged_mean_logits.npz"}}),
        encoding="utf-8",
    )
    cfg = _load_config()
    cfg.calibrate.gold = str(gold_path)
    cfg.calibrate.cache_dir = str(cache_dir)
    cfg.calibrate.checkpoint = str(tmp_path / "missing_ckpt")
    cfg.calibrate.output = str(tmp_path / "threshold.json")
    cfg.calibrate.metrics_out = str(tmp_path / "threshold_metrics.json")
    cfg.calibrate.fill_cache = False
    cfg.calibrate.taus = [0.0, 0.5]
    cfg.calibrate.k = 2
    cfg.calibrate.seed = 42
    settings = Settings(metrics_path=tmp_path / "metrics.json")
    path = run_calibrate(cfg, settings=settings)
    metrics = json.loads(path.read_text(encoding="utf-8"))
    report = json.loads((tmp_path / "threshold.json").read_text(encoding="utf-8"))
    assert metrics["status"] == "ok"
    assert report["status"] == "ok"
    assert report["selected_tau"] in {0.0, 0.5}
    assert "grid" in report

#!/usr/bin/env python3
"""GLiNER span-score calibration and confidence-threshold sweep on official dev.

Unlike uztext token ECE (diluted by gold ``O``), this script reports **only**
span-level binary calibration: GLiNER ``score`` vs exact-span correctness.
Inference uses sliding-word windows (``gliner_windows`` defaults); τ is applied
in CPU post-processing so the score distribution is not re-cut by the model.

Protocol matches ``uzbek_ner.decode.threshold``: seed 42, 5-fold, fold 0 =
analysis (300 docs), rest = held-out (1200). τ is picked on analysis F1;
held-out F1 is the headline number.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from gliner import GLiNER
from loguru import logger

from uzbek_ner.decode.kfold import score_split
from uzbek_ner.decode.threshold import analysis_held_split, gold_lookup, micro_view
from uzbek_ner.gliner_windows import DEFAULT_MAX_WORDS, DEFAULT_STRIDE, predict_records_windowed
from uzbek_ner.io.jsonl import read_jsonl_records
from uzbek_ner.metrics.calibration import binary_calibration, confidence_reading
from uzbek_ner.metrics.exact_span import EntityKey
from uzbek_ner.settings import REPO_ROOT, get_settings

JsonObject = dict[str, Any]
N_BINS = 15


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "checkpoints" / "gliner_multi_v21",
    )
    parser.add_argument("--gold", type=Path, default=settings.official_dev)
    parser.add_argument(
        "--cache",
        type=Path,
        default=REPO_ROOT
        / "outputs"
        / "cache"
        / "gliner_multi_v21_official_dev"
        / "scored_spans.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs" / "eval" / "gliner_multi_v21_calibration.json",
    )
    parser.add_argument("--infer-threshold", type=float, default=0.0)
    parser.add_argument("--max-words", type=int, default=DEFAULT_MAX_WORDS)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--tau-step", type=float, default=0.01)
    parser.add_argument("--force-infer", action="store_true")
    parser.add_argument("--skip-infer", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def infer_scored_spans(
    model: GLiNER,
    records: list[JsonObject],
    *,
    infer_threshold: float,
    batch_size: int,
    max_words: int,
    stride: int,
) -> dict[str, list[JsonObject]]:
    logger.info(
        "infer {} docs infer_threshold={} max_words={} stride={} batch_size={}",
        len(records),
        infer_threshold,
        max_words,
        stride,
        batch_size,
    )
    with torch.inference_mode():
        raw, _meta = predict_records_windowed(
            model,
            records,
            max_words=max_words,
            stride=stride,
            threshold=infer_threshold,
            batch_size=batch_size,
        )
    scored: dict[str, list[JsonObject]] = {}
    for record in records:
        record_hash = str(record["hash"])
        rows = [
            {
                "label": str(row["label"]),
                "start": int(row["start"]),
                "end": int(row["end"]),
                "score": float(row["score"]),
            }
            for row in raw[record_hash]
        ]
        rows.sort(key=lambda item: (item["start"], item["end"], item["label"]))
        scored[record_hash] = rows
    return scored


def load_or_infer_scored(
    *,
    checkpoint: Path,
    records: list[JsonObject],
    cache_path: Path,
    infer_threshold: float,
    max_words: int,
    stride: int,
    batch_size: int,
    device: str,
    force: bool,
    skip_infer: bool,
) -> dict[str, list[JsonObject]]:
    meta_ok = False
    if cache_path.is_file() and not force:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        meta = payload.get("meta", {})
        meta_ok = (
            meta.get("checkpoint") == str(checkpoint.resolve())
            and float(meta.get("infer_threshold", -1)) == infer_threshold
            and int(meta.get("max_words", -1)) == max_words
            and int(meta.get("stride", -1)) == stride
            and meta.get("n_docs") == len(records)
        )
        if meta_ok:
            logger.info("load scored cache {}", cache_path)
            return {row["hash"]: row["spans"] for row in payload["docs"]}

    if skip_infer:
        raise FileNotFoundError(f"missing or stale cache at {cache_path} (--skip-infer)")

    if not checkpoint.is_dir():
        raise FileNotFoundError(f"missing checkpoint at {checkpoint}")

    model = GLiNER.from_pretrained(str(checkpoint), local_files_only=True)
    model.to(device)
    model.eval()
    scored = infer_scored_spans(
        model,
        records,
        infer_threshold=infer_threshold,
        batch_size=batch_size,
        max_words=max_words,
        stride=stride,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "meta": {
                    "checkpoint": str(checkpoint.resolve()),
                    "infer_threshold": infer_threshold,
                    "max_words": max_words,
                    "stride": stride,
                    "n_docs": len(records),
                    "created_at": datetime.now(UTC).isoformat(),
                },
                "docs": [{"hash": h, "spans": spans} for h, spans in scored.items()],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    logger.info("wrote scored cache {} spans={}", cache_path, sum(len(v) for v in scored.values()))
    return scored


def filter_scored(
    scored: dict[str, list[JsonObject]],
    *,
    tau: float,
) -> dict[str, set[EntityKey]]:
    return {
        record_hash: {
            (str(row["label"]), int(row["start"]), int(row["end"]))
            for row in spans
            if float(row["score"]) >= tau
        }
        for record_hash, spans in scored.items()
    }


def span_calibration(
    scored: dict[str, list[JsonObject]],
    gold: dict[str, JsonObject],
    hashes: set[str],
    *,
    tau: float = 0.0,
    n_bins: int = N_BINS,
) -> dict[str, Any]:
    confidences: list[float] = []
    correct: list[bool] = []
    for record_hash in hashes:
        gold_keys = gold[record_hash]["entities"]
        for row in scored[record_hash]:
            if float(row["score"]) < tau:
                continue
            key = (str(row["label"]), int(row["start"]), int(row["end"]))
            confidences.append(float(row["score"]))
            correct.append(key in gold_keys)
    metrics = binary_calibration(
        np.asarray(confidences, dtype=np.float64),
        np.asarray(correct, dtype=np.bool_),
        n_bins=n_bins,
    )
    reading = confidence_reading(metrics["confidence_minus_accuracy"])
    slim = {key: value for key, value in metrics.items() if key != "reliability"}
    slim["reading"] = reading
    return slim


def precision_bins(
    scored: dict[str, list[JsonObject]],
    gold: dict[str, JsonObject],
    hashes: set[str],
    *,
    width: float = 0.1,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n_bins_count = round(1.0 / width)
    for bin_index in range(n_bins_count):
        lo = round(bin_index * width, 10)
        hi = round(lo + width, 10)
        if bin_index == n_bins_count - 1:
            hi = 1.0 + 1e-9
        tp = fp = 0
        for record_hash in hashes:
            gold_keys = gold[record_hash]["entities"]
            for span in scored[record_hash]:
                score = float(span["score"])
                if score < lo or score >= hi:
                    continue
                key = (str(span["label"]), int(span["start"]), int(span["end"]))
                if key in gold_keys:
                    tp += 1
                else:
                    fp += 1
        n = tp + fp
        rows.append(
            {
                "lo": lo,
                "hi": hi if hi <= 1.0 else 1.0,
                "n": n,
                "tp": tp,
                "fp": fp,
                "precision": (tp / n) if n else None,
            }
        )
    return rows


def tau_grid(step: float) -> list[float]:
    if step <= 0 or step > 1:
        raise ValueError("tau-step must be in (0, 1]")
    count = round(1.0 / step)
    values = [round(index * step, 10) for index in range(count + 1)]
    if values[-1] != 1.0:
        values.append(1.0)
    return sorted(set(values))


def sweep_tau(
    scored: dict[str, list[JsonObject]],
    gold: dict[str, JsonObject],
    *,
    analysis_hashes: list[str],
    held_out_hashes: list[str],
    all_hashes: list[str],
    taus: list[float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tau in taus:
        keyed = filter_scored(scored, tau=tau)
        rows.append(
            {
                "tau": float(tau),
                "analysis": micro_view(score_split(gold, keyed, analysis_hashes)),
                "held_out": micro_view(score_split(gold, keyed, held_out_hashes)),
                "full_dev": micro_view(score_split(gold, keyed, all_hashes)),
                "held_out_predicted": sum(len(keyed[h]) for h in held_out_hashes),
            }
        )
    return rows


def pick_tau(grid: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = min(grid, key=lambda row: row["tau"])
    picked = max(grid, key=lambda row: (float(row["analysis"]["f1"]), -float(row["tau"])))
    return {
        "baseline_tau": float(baseline["tau"]),
        "selected_tau": float(picked["tau"]),
        "analysis": picked["analysis"],
        "held_out": picked["held_out"],
        "full_dev": picked["full_dev"],
        "baseline_held_out": baseline["held_out"],
        "held_out_delta_f1": float(picked["held_out"]["f1"]) - float(baseline["held_out"]["f1"]),
    }


def main() -> None:
    args = parse_args()
    records = read_jsonl_records(args.gold)
    gold = gold_lookup(records)
    hashes = [record["hash"] for record in records]
    analysis_hashes, held_out_hashes = analysis_held_split(hashes, k=args.k, seed=args.seed)
    analysis_set = set(analysis_hashes)
    held_set = set(held_out_hashes)
    all_set = set(hashes)

    scored = load_or_infer_scored(
        checkpoint=args.checkpoint,
        records=records,
        cache_path=args.cache,
        infer_threshold=args.infer_threshold,
        max_words=args.max_words,
        stride=args.stride,
        batch_size=args.batch_size,
        device=args.device,
        force=args.force_infer,
        skip_infer=args.skip_infer,
    )

    taus = tau_grid(args.tau_step)
    grid = sweep_tau(
        scored,
        gold,
        analysis_hashes=analysis_hashes,
        held_out_hashes=held_out_hashes,
        all_hashes=hashes,
        taus=taus,
    )
    selection = pick_tau(grid)

    calibration = {
        "note": (
            "Span-level only. GLiNER score vs exact (label,start,end). "
            "No token-O dilution. infer_threshold collects candidates; "
            "tau filters in post."
        ),
        "infer_threshold": args.infer_threshold,
        "full_dev": span_calibration(scored, gold, all_set, tau=0.0),
        "analysis": span_calibration(scored, gold, analysis_set, tau=0.0),
        "held_out": span_calibration(scored, gold, held_set, tau=0.0),
        "at_tau_0_5": span_calibration(scored, gold, all_set, tau=0.5),
        "at_selected_tau": span_calibration(
            scored, gold, all_set, tau=float(selection["selected_tau"])
        ),
        "precision_bins_full_dev": precision_bins(scored, gold, all_set),
    }

    payload = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "checkpoint": str(args.checkpoint.resolve()),
        "gold": str(args.gold.resolve()),
        "cache": str(args.cache.resolve()),
        "split": {
            "seed": args.seed,
            "k": args.k,
            "analysis_docs": len(analysis_hashes),
            "held_out_docs": len(held_out_hashes),
        },
        "n_spans_collected": sum(len(spans) for spans in scored.values()),
        "calibration": calibration,
        "threshold_sweep": {
            "tau_step": args.tau_step,
            "grid": grid,
            **selection,
        },
        "reference": {
            "tau_0_5_full_dev_f1": next(row["full_dev"]["f1"] for row in grid if row["tau"] == 0.5),
            "uztext_smoke_snap_held_out_f1": 0.668,
            "uztext_smoke_snap_tau_0_67_held_out_f1": 0.704,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(jsonable(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    logger.info(
        "span ECE full={:.4f} gap={:+.4f} {} | τ*={} held-out F1 {:.4f} (Δ {:+.4f} vs τ={})",
        calibration["full_dev"]["ece"],
        calibration["full_dev"]["confidence_minus_accuracy"],
        calibration["full_dev"]["reading"],
        selection["selected_tau"],
        selection["held_out"]["f1"],
        selection["held_out_delta_f1"],
        selection["baseline_tau"],
    )
    logger.info("wrote {}", args.output)


if __name__ == "__main__":
    main()

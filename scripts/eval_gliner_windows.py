#!/usr/bin/env python3
"""Evaluate GLiNER with sliding-word windows on official dev."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from gliner import GLiNER
from loguru import logger

from uzbek_ner.gliner_windows import (
    DEFAULT_MAX_WORDS,
    DEFAULT_STRIDE,
    entities_for_submission,
    predict_records_windowed,
    word_window_spans,
)
from uzbek_ner.io.jsonl import read_jsonl_records
from uzbek_ner.metrics.exact_span import evaluate_prediction_files, write_metrics
from uzbek_ner.settings import REPO_ROOT, get_settings

JsonObject = dict[str, Any]


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "checkpoints" / "gliner_multi_v21",
    )
    parser.add_argument("--gold", type=Path, default=settings.official_dev)
    parser.add_argument("--run-id", default="gliner_multi_v21_windows")
    parser.add_argument("--max-words", type=int, default=DEFAULT_MAX_WORDS)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--baseline-metrics", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def write_prediction_jsonl(
    path: Path,
    records: list[JsonObject],
    predictions: dict[str, list[JsonObject]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            entities = entities_for_submission(predictions[record["hash"]])
            stream.write(
                json.dumps(
                    {
                        "hash": record["hash"],
                        "text": record["text"],
                        "entities": entities,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def window_stats(records: list[JsonObject], *, max_words: int, stride: int) -> dict[str, Any]:
    per_doc = [
        len(word_window_spans(str(record["text"]), max_words=max_words, stride=stride))
        for record in records
    ]
    multi = sum(1 for count in per_doc if count > 1)
    return {
        "docs": len(records),
        "docs_multi_window": multi,
        "docs_single_window": len(records) - multi,
        "windows_total": sum(per_doc),
        "windows_mean": sum(per_doc) / len(per_doc) if per_doc else 0.0,
        "windows_max": max(per_doc) if per_doc else 0,
    }


def main() -> None:
    args = parse_args()
    records = read_jsonl_records(args.gold)
    stats = window_stats(records, max_words=args.max_words, stride=args.stride)
    logger.info("window stats: {}", stats)

    model = GLiNER.from_pretrained(str(args.checkpoint), local_files_only=True)
    model.to(args.device)
    model.eval()

    raw, windows_per_doc = predict_records_windowed(
        model,
        records,
        max_words=args.max_words,
        stride=args.stride,
        threshold=args.threshold,
        batch_size=args.batch_size,
    )
    predictions = raw

    pred_path = Path(f"outputs/official/{args.run_id}_dev_predictions.jsonl")
    write_prediction_jsonl(pred_path, records, predictions)
    metrics = evaluate_prediction_files(args.gold, pred_path)
    metrics_path = Path(f"outputs/official/{args.run_id}_dev_metrics.json")
    write_metrics(metrics_path, metrics)

    baseline_path = args.baseline_metrics or (
        REPO_ROOT / "outputs" / "official" / "gliner_multi_v21_dev_metrics.json"
    )
    baseline_f1 = None
    if baseline_path.is_file():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        baseline_f1 = float(baseline["micro"]["f1"])

    report = {
        "run_id": args.run_id,
        "checkpoint": str(args.checkpoint.resolve()),
        "gold": str(args.gold.resolve()),
        "max_words": args.max_words,
        "stride": args.stride,
        "threshold": args.threshold,
        "window_stats": stats,
        "metrics": metrics,
        "baseline_metrics": str(baseline_path) if baseline_path.is_file() else None,
        "baseline_micro_f1": baseline_f1,
        "delta_micro_f1": (
            float(metrics["micro"]["f1"]) - baseline_f1 if baseline_f1 is not None else None
        ),
        "paths": {
            "predictions": str(pred_path.resolve()),
            "metrics": str(metrics_path.resolve()),
        },
    }
    report_path = REPO_ROOT / "outputs" / "eval" / f"{args.run_id}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    logger.info(
        "windowed dev micro-F1 {:.4f} P={:.4f} R={:.4f} (baseline {:.4f}, Δ {:+.4f}) → {}",
        metrics["micro"]["f1"],
        metrics["micro"]["precision"],
        metrics["micro"]["recall"],
        baseline_f1 if baseline_f1 is not None else float("nan"),
        report["delta_micro_f1"] if report["delta_micro_f1"] is not None else float("nan"),
        metrics_path,
    )
    if windows_per_doc:
        logger.info(
            "sample windows/doc: min={} max={}",
            min(windows_per_doc.values()),
            max(windows_per_doc.values()),
        )


if __name__ == "__main__":
    main()

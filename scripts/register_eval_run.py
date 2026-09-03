#!/usr/bin/env python3
"""Ingest exact-span metrics into the local eval-run registry (Grafana / eval API)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from uzbek_ner.evaldash.registry import evaluate_and_register, ingest_metrics, resolve_runs_dir


def _load_hyperparams(raw: str | None, path: Path | None) -> dict[str, Any]:
    if path is not None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("--hyperparams-file must contain a JSON object")
        return payload
    if raw:
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("--hyperparams must be a JSON object")
        return payload
    return {}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="Registry key / filename stem")
    parser.add_argument("--model", required=True, help="HuggingFace id or local model name")
    parser.add_argument("--checkpoint", default="", help="Checkpoint directory")
    parser.add_argument("--split", default="official_dev")
    parser.add_argument("--gold", type=Path, default=None, help="Gold JSONL (official scorer)")
    parser.add_argument("--predictions", type=Path, default=None, help="Predictions JSONL")
    parser.add_argument(
        "--metrics",
        type=Path,
        default=None,
        help="Metrics JSON: input if scoring is skipped, output if gold+predictions are given",
    )
    parser.add_argument("--hyperparams", default=None, help="JSON object string")
    parser.add_argument("--hyperparams-file", type=Path, default=None)
    parser.add_argument("--created-at", default=None, help="ISO-8601 timestamp (default: now)")
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=None,
        help="Registry directory (default: EVAL_RUNS_DIR or outputs/eval/runs)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    hyperparams = _load_hyperparams(args.hyperparams, args.hyperparams_file)
    runs_dir = args.runs_dir
    gold = args.gold
    predictions = args.predictions
    metrics_path = args.metrics

    if gold is not None and predictions is not None:
        if metrics_path is None:
            metrics_path = predictions.with_name(predictions.stem + "_metrics.json")
        run = evaluate_and_register(
            gold,
            predictions,
            run_id=args.run_id,
            model=args.model,
            metrics_out=metrics_path,
            checkpoint=args.checkpoint,
            split=args.split,
            hyperparams=hyperparams,
            created_at=args.created_at,
            directory=runs_dir,
        )
    elif metrics_path is not None:
        if not metrics_path.is_file():
            print(f"metrics file not found: {metrics_path}", file=sys.stderr)
            return 1
        run = ingest_metrics(
            metrics_path,
            run_id=args.run_id,
            model=args.model,
            checkpoint=args.checkpoint,
            split=args.split,
            hyperparams=hyperparams,
            gold=gold or "",
            predictions=predictions or "",
            created_at=args.created_at,
            directory=runs_dir,
        )
    else:
        print("provide --gold and --predictions, or an existing --metrics JSON", file=sys.stderr)
        return 2

    dest = resolve_runs_dir(runs_dir) / f"{run.run_id}.json"
    print(
        f"registered {run.run_id} micro-F1={run.metrics.micro.f1:.4f} → {dest}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

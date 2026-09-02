#!/usr/bin/env python3
"""Run organizer tooling from repo root with stable paths."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from uzbek_ner.settings import get_settings


def _run(script: str, args: list[str]) -> int:
    settings = get_settings()
    scripts_dir = settings.official_scripts
    cmd = [sys.executable, str(scripts_dir / script), *args]
    return subprocess.call(cmd, cwd=scripts_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check-api", help="Validate HTTP service contract")
    check.add_argument("--url", default="http://localhost:8000")

    evaluate = sub.add_parser("evaluate", help="Score predictions JSONL vs official dev")
    evaluate.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Predictions JSONL (hash + entities)",
    )
    evaluate.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/official/dev_metrics.json"),
    )
    evaluate.add_argument(
        "--gold",
        type=Path,
        default=None,
        help="Gold JSONL (default: data/official/dev.jsonl)",
    )

    service = sub.add_parser("evaluate-service", help="Run dev through HTTP service and score")
    service.add_argument("--url", default="http://localhost:8000")
    service.add_argument(
        "--predictions",
        type=Path,
        default=Path("outputs/official/service_dev_predictions.jsonl"),
    )
    service.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/official/service_dev_metrics.json"),
    )
    service.add_argument("--batch-size", type=int, default=16)

    args = parser.parse_args()
    settings = get_settings()
    gold = (args.gold if hasattr(args, "gold") and args.gold else settings.official_dev).resolve()

    if args.command == "check-api":
        return _run("check_service.py", ["--url", args.url])

    if args.command == "evaluate":
        predictions = args.predictions.expanduser().resolve()
        output = args.output.expanduser().resolve()
        return _run(
            "evaluate.py",
            [
                "--gold",
                str(gold),
                "--predictions",
                str(predictions),
                "--output",
                str(output),
            ],
        )

    predictions = args.predictions.expanduser().resolve()
    output = args.output.expanduser().resolve()
    return _run(
        "evaluate_service.py",
        [
            "--url",
            args.url,
            "--gold",
            str(gold),
            "--predictions",
            str(predictions),
            "--output",
            str(output),
            "--batch-size",
            str(args.batch_size),
        ],
    )


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""K-fold decode tuning on official dev.

One fold is the analysis slice (error geometry + offset mode). The other folds
are scored with exact-span F1 and averaged. Analysis hashes never enter the
reported mean.

Gold-free policies only: identity, global/per-label/mid-token offset mode
fitted on the analysis fold, and linguistic word/suffix snap (no fit).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from loguru import logger

from uzbek_ner.decode.kfold import (
    apply_policy_to_split,
    fit_offset_mode,
    identity_policy,
    make_folds,
    make_global_mode_policy,
    make_midtoken_mode_policy,
    make_per_label_mode_policy,
    mean_std,
    score_split,
    word_snap_policy,
    word_suffix_snap_policy,
)
from uzbek_ner.metrics.exact_span import load_gold_and_predictions
from uzbek_ner.settings import get_settings


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, default=settings.official_dev)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("outputs/official/uztext_smoke_dev_predictions.jsonl"),
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/eval/decode_kfold.json"),
    )
    return parser.parse_args()


def load_raw_predictions(path: Path) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            rows[record["hash"]] = list(record["entities"])
    return rows


def main() -> None:
    args = parse_args()
    gold, keyed_predictions = load_gold_and_predictions(args.gold, args.predictions)
    raw = load_raw_predictions(args.predictions)
    hashes = sorted(gold)
    folds = make_folds(hashes, k=args.k, seed=args.seed)
    logger.info("k={} docs={} fold sizes {}", args.k, len(hashes), [len(fold) for fold in folds])

    policy_scores: dict[str, list[float]] = {
        "identity": [],
        "global_mode": [],
        "joint_mode": [],
        "per_label_mode": [],
        "midtoken_mode": [],
        "word_snap": [],
        "word_suffix_snap": [],
    }
    fold_reports: list[dict[str, Any]] = []

    for fold_index, analysis in enumerate(folds):
        held_out = [
            record_hash for i, fold in enumerate(folds) if i != fold_index for record_hash in fold
        ]
        fitted = fit_offset_mode(gold, keyed_predictions, analysis)
        policies = {
            "identity": identity_policy,
            "global_mode": make_global_mode_policy(int(fitted["d_start"]), int(fitted["d_end"])),
            "joint_mode": make_global_mode_policy(
                int(fitted["joint_d_start"]), int(fitted["joint_d_end"])
            ),
            "per_label_mode": make_per_label_mode_policy(fitted),
            "midtoken_mode": make_midtoken_mode_policy(
                int(fitted["d_start"]), int(fitted["d_end"])
            ),
            "word_snap": word_snap_policy,
            "word_suffix_snap": word_suffix_snap_policy,
        }
        fold_row: dict[str, Any] = {
            "analysis_fold": fold_index,
            "analysis_docs": len(analysis),
            "eval_docs": len(held_out),
            "fitted": fitted,
            "metrics": {},
        }
        for name, policy in policies.items():
            applied = apply_policy_to_split(gold, raw, held_out, policy)
            metrics = score_split(gold, applied, held_out)
            f1 = float(metrics["micro"]["f1"])
            policy_scores[name].append(f1)
            fold_row["metrics"][name] = {
                "micro_f1": f1,
                "micro_precision": float(metrics["micro"]["precision"]),
                "micro_recall": float(metrics["micro"]["recall"]),
            }
        fold_reports.append(fold_row)
        logger.info(
            "fold {} analysis={} eval={} marg=({},{}) joint=({},{}) n_pairs={} "
            "held-out F1 ident={:.4f} joint={:.4f} snap={:.4f}",
            fold_index,
            len(analysis),
            len(held_out),
            fitted["d_start"],
            fitted["d_end"],
            fitted["joint_d_start"],
            fitted["joint_d_end"],
            fitted["n_pairs"],
            fold_row["metrics"]["identity"]["micro_f1"],
            fold_row["metrics"]["joint_mode"]["micro_f1"],
            fold_row["metrics"]["word_suffix_snap"]["micro_f1"],
        )

    summary = {
        "schema_version": 1,
        "k": args.k,
        "seed": args.seed,
        "gold": str(args.gold),
        "predictions": str(args.predictions),
        "docs": len(hashes),
        "note": "Each fold is analysis-only; reported F1 is the complement. Exact-span official metric.",
        "policies": {},
        "folds": fold_reports,
    }
    for name, values in policy_scores.items():
        mean, std = mean_std(values)
        summary["policies"][name] = {
            "mean_f1": mean,
            "std_f1": std,
            "folds": values,
        }
        logger.info("policy {:<18} mean F1 {:.4f} ± {:.4f}", name, mean, std)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    logger.info("wrote {}", args.output)


if __name__ == "__main__":
    main()

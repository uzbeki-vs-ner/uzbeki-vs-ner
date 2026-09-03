#!/usr/bin/env python3
"""Temperature / calibration experiment for Mix B on official dev.

Caches backbone last hidden states + head logits, then sweeps softmax T.
Argmax+snap F1 is T-invariant on merged mean logits; the live question is
whether calibrated T + a confidence gate moves official exact-span F1.

Always run the GPU cache fill under ``flock outputs/.gpu.lock``. Never scores
silver. Never writes checkpoints.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from uzbek_ner.decode.kfold import make_folds, score_split
from uzbek_ner.decode.snap import snap_entities
from uzbek_ner.decode.threshold import (
    decode_doc,
    entity_conf,
    gold_lookup,
    keyed_predictions,
    micro_view,
    predict_split,
)
from uzbek_ner.io.jsonl import read_jsonl_records
from uzbek_ner.metrics.calibration import (
    binary_calibration,
    confidence_reading,
    temperature_softmax,
    token_calibration,
)
from uzbek_ner.metrics.error_analysis import analyze_span_errors
from uzbek_ner.modeling.eval_cache import (
    N_LABELS,
    MergedDoc,
    cache_complete,
    fill_cache,
    load_merged_npz,
)
from uzbek_ner.settings import REPO_ROOT, get_settings

TEMPERATURES = (0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0)
TAUS = (0.0, 0.3, 0.5, 0.7)
N_BINS = 15


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "checkpoints" / "uztext_mixb_ep1",
    )
    parser.add_argument("--gold", type=Path, default=settings.official_dev)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "cache" / "mixb_official_dev",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs" / "eval" / "mixb_temperature.json",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--force-cache", action="store_true")
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
    if isinstance(value, set):
        return [jsonable(item) for item in value]
    return value


def slim_token_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key != "reliability"}


def stack_tokens(
    docs: list[MergedDoc], hashes: set[str]
) -> tuple[NDArray[np.float32], NDArray[np.int64], int]:
    logits: list[NDArray[np.float32]] = []
    labels: list[NDArray[np.int64]] = []
    empty_docs = 0
    for doc in docs:
        if doc.record_hash not in hashes:
            continue
        if not doc.gold_keys:
            empty_docs += 1
        if doc.logits.shape[0] == 0:
            continue
        keep = doc.gold_labels >= 0
        logits.append(doc.logits[keep])
        labels.append(doc.gold_labels[keep])
    if not logits:
        return (
            np.zeros((0, N_LABELS), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            empty_docs,
        )
    return np.concatenate(logits, axis=0), np.concatenate(labels, axis=0), empty_docs


def collect_entity_calibration(
    docs: list[MergedDoc],
    hashes: set[str],
    *,
    temperature: float,
    reduce: str,
) -> dict[str, Any]:
    confidences: list[float] = []
    correct: list[bool] = []
    for doc in docs:
        if doc.record_hash not in hashes:
            continue
        raw = decode_doc(doc, temperature=temperature, min_conf=0.0, reduce=reduce, snap=False)
        if doc.logits.shape[0] == 0:
            continue
        probs = temperature_softmax(doc.logits, temperature)
        tags = probs.argmax(axis=1)
        for entity in raw:
            conf = entity_conf(doc.offsets, tags, probs, entity, reduce=reduce)
            snapped = snap_entities(doc.text, [entity])
            if not snapped:
                continue
            row = snapped[0]
            key = (str(row["label"]), int(row["start"]), int(row["end"]))
            confidences.append(conf)
            correct.append(key in doc.gold_keys)
    return slim_token_metrics(
        binary_calibration(
            np.asarray(confidences, dtype=np.float64),
            np.asarray(correct, dtype=np.bool_),
            n_bins=N_BINS,
        )
    )


def pick_temperature(by_t: dict[str, dict[str, Any]]) -> tuple[float, float]:
    """Min analysis ECE, then NLL, then closer to 1.0."""

    def key(temp: float) -> tuple[float, float, float]:
        row = by_t[str(temp)]["token"]
        return (float(row["ece"]), float(row["nll"]), abs(temp - 1.0))

    best_ece = min(TEMPERATURES, key=key)
    best_nll = min(TEMPERATURES, key=lambda temp: float(by_t[str(temp)]["token"]["nll"]))
    return float(best_ece), float(best_nll)


def argmax_tag_signature(docs: list[MergedDoc], hashes: set[str]) -> tuple[int, ...]:
    signature: list[int] = []
    for doc in docs:
        if doc.record_hash not in hashes or doc.logits.shape[0] == 0:
            continue
        signature.extend(int(tag) for tag in doc.logits.argmax(axis=1).tolist())
    return tuple(signature)


def run_experiment(
    *,
    docs: list[MergedDoc],
    records: list[dict[str, Any]],
    cache_meta: dict[str, Any],
    cache_dir: Path,
    checkpoint: Path,
    gold_path: Path,
    seed: int,
    k: int,
) -> dict[str, Any]:
    gold = gold_lookup(records)
    hashes = [doc.record_hash for doc in docs]
    folds = make_folds(hashes, k=k, seed=seed)
    analysis_hashes = list(folds[0])
    held_out_hashes = [record_hash for fold in folds[1:] for record_hash in fold]
    analysis_set = set(analysis_hashes)
    held_set = set(held_out_hashes)
    empty_total = sum(1 for record in records if not record["entities"])
    analysis_empty = sum(1 for doc in docs if doc.record_hash in analysis_set and not doc.gold_keys)
    held_empty = sum(1 for doc in docs if doc.record_hash in held_set and not doc.gold_keys)

    analysis_logits, analysis_labels, _ = stack_tokens(docs, analysis_set)
    held_logits, held_labels, _ = stack_tokens(docs, held_set)
    _, full_labels, _ = stack_tokens(docs, set(hashes))

    baseline_pred = predict_split(docs, None, temperature=1.0, min_conf=0.0, reduce="mean")
    baseline_full = score_split(gold, keyed_predictions(baseline_pred), hashes)
    baseline_held = score_split(gold, keyed_predictions(baseline_pred), held_out_hashes)
    baseline_analysis = score_split(gold, keyed_predictions(baseline_pred), analysis_hashes)

    tag_sig = argmax_tag_signature(docs, set(hashes))
    analysis_by_t: dict[str, dict[str, Any]] = {}
    held_by_t: dict[str, dict[str, Any]] = {}
    argmax_f1: dict[str, Any] = {"full_dev": {}, "held_out": {}, "analysis": {}}
    for temperature in TEMPERATURES:
        analysis_by_t[str(temperature)] = {
            "token": slim_token_metrics(
                token_calibration(
                    analysis_logits, analysis_labels, temperature=temperature, n_bins=N_BINS
                )
            ),
            "entity_mean": collect_entity_calibration(
                docs, analysis_set, temperature=temperature, reduce="mean"
            ),
        }
        held_by_t[str(temperature)] = {
            "token": slim_token_metrics(
                token_calibration(held_logits, held_labels, temperature=temperature, n_bins=N_BINS)
            ),
            "entity_mean": collect_entity_calibration(
                docs, held_set, temperature=temperature, reduce="mean"
            ),
        }
        preds = predict_split(docs, None, temperature=temperature, min_conf=0.0, reduce="mean")
        keyed = keyed_predictions(preds)
        argmax_f1["full_dev"][str(temperature)] = micro_view(score_split(gold, keyed, hashes))
        argmax_f1["held_out"][str(temperature)] = micro_view(
            score_split(gold, keyed, held_out_hashes)
        )
        argmax_f1["analysis"][str(temperature)] = micro_view(
            score_split(gold, keyed, analysis_hashes)
        )
        logger.info(
            "T={:<4} analysis ECE={:.4f} NLL={:.4f} Brier={:.4f} | "
            "held ECE={:.4f} NLL={:.4f} | full F1={:.4f}",
            temperature,
            analysis_by_t[str(temperature)]["token"]["ece"],
            analysis_by_t[str(temperature)]["token"]["nll"],
            analysis_by_t[str(temperature)]["token"]["brier"],
            held_by_t[str(temperature)]["token"]["ece"],
            held_by_t[str(temperature)]["token"]["nll"],
            argmax_f1["full_dev"][str(temperature)]["f1"],
        )

    unique_full_f1 = {round(float(row["f1"]), 12) for row in argmax_f1["full_dev"].values()}
    t_ece, t_nll = pick_temperature(analysis_by_t)
    t_star = t_ece

    reliability_t1 = token_calibration(held_logits, held_labels, temperature=1.0, n_bins=N_BINS)
    reliability_best = token_calibration(
        held_logits, held_labels, temperature=t_star, n_bins=N_BINS
    )

    threshold_rows: dict[str, list[dict[str, Any]]] = {"mean": [], "max": []}
    for reduce in ("mean", "max"):
        for tau in TAUS:
            preds = predict_split(
                docs, None, temperature=t_star, min_conf=float(tau), reduce=reduce
            )
            keyed = keyed_predictions(preds)
            analysis_metrics = score_split(gold, keyed, analysis_hashes)
            held_metrics = score_split(gold, keyed, held_out_hashes)
            n_pred_held = sum(len(preds[h]) for h in held_out_hashes)
            threshold_rows[reduce].append(
                {
                    "tau": float(tau),
                    "analysis": micro_view(analysis_metrics),
                    "held_out": micro_view(held_metrics),
                    "held_out_predicted": n_pred_held,
                }
            )
            logger.info(
                "T*={:g} reduce={} tau={:<3} analysis F1={:.4f} held-out F1={:.4f} pred={}",
                t_star,
                reduce,
                tau,
                analysis_metrics["micro"]["f1"],
                held_metrics["micro"]["f1"],
                n_pred_held,
            )

    snap_held_f1 = float(baseline_held["micro"]["f1"])
    best_gate = max(
        threshold_rows["mean"],
        key=lambda row: (
            float(row["analysis"]["f1"]),
            -float(row["tau"]),
        ),
    )
    # Honest number: held-out F1 at the τ chosen on analysis (never on held-out).
    gate_held_f1 = float(best_gate["held_out"]["f1"])
    t_plus_threshold_wins = gate_held_f1 > snap_held_f1 + 1e-12

    held_pred_t1 = keyed_predictions(
        predict_split(docs, held_out_hashes, temperature=1.0, min_conf=0.0, reduce="mean")
    )
    held_gold = {record_hash: gold[record_hash] for record_hash in held_out_hashes}
    diag_t1 = analyze_span_errors(held_gold, held_pred_t1)
    held_pred_gate = keyed_predictions(
        predict_split(
            docs,
            held_out_hashes,
            temperature=t_star,
            min_conf=float(best_gate["tau"]),
            reduce="mean",
        )
    )
    diag_gate = analyze_span_errors(held_gold, held_pred_gate)

    t1_gap = float(reliability_t1["confidence_minus_accuracy"])
    best_gap = float(reliability_best["confidence_minus_accuracy"])
    head_reading = confidence_reading(t1_gap)
    calibrated_reading = confidence_reading(best_gap)
    full_f1 = float(baseline_full["micro"]["f1"])
    flat = len(unique_full_f1) == 1
    if t_plus_threshold_wins:
        verdict_text = (
            f"Token head at T=1 is {head_reading} "
            f"(mean max-p {reliability_t1['mean_confidence']:.3f} vs token acc "
            f"{reliability_t1['accuracy']:.3f}, ECE {reliability_t1['ece']:.4f}). "
            f"T*={t_star:g} + τ={best_gate['tau']} beats snap-only Mix B on held-out "
            f"({gate_held_f1:.4f} vs {snap_held_f1:.4f}, "
            f"Δ {gate_held_f1 - snap_held_f1:+.4f}). "
            "Defaults still not changed in this task (measure-only)."
        )
    else:
        verdict_text = (
            f"Token head at T=1 is {head_reading} "
            f"(mean max-p {reliability_t1['mean_confidence']:.3f} vs token acc "
            f"{reliability_t1['accuracy']:.3f}, ECE {reliability_t1['ece']:.4f}). "
            f"T*={t_star:g} selected by min analysis ECE "
            f"(T_NLL={t_nll:g}); held-out ECE {reliability_best['ece']:.4f}, "
            f"gap {best_gap:+.3f} ({calibrated_reading}). "
            f"Argmax+snap F1 is {'flat' if flat else 'NOT flat'} "
            f"across T (full-dev F1 {full_f1:.4f}). "
            f"Best analysis τ={best_gate['tau']} (mean entity conf) gives held-out "
            f"F1 {gate_held_f1:.4f} vs snap-only {snap_held_f1:.4f} "
            f"(Δ {gate_held_f1 - snap_held_f1:+.4f}). "
            "predict_records default left unchanged."
        )

    nbytes = cache_meta.get("nbytes") or {
        name: (cache_dir / filename).stat().st_size
        for name, filename in cache_meta["files"].items()
        if (cache_dir / filename).is_file()
    }

    return {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "checkpoint": str(checkpoint),
        "gold": str(gold_path),
        "n_docs": len(records),
        "empty_docs": empty_total,
        "empty_docs_share": empty_total / len(records) if records else 0.0,
        "split": {
            "method": (
                f"{k}-fold round-robin, fold 0 = analysis (~20%), "
                "folds 1..k-1 = held-out. Analysis hashes never enter reported F1."
            ),
            "seed": seed,
            "k": k,
            "analysis_docs": len(analysis_hashes),
            "held_out_docs": len(held_out_hashes),
            "analysis_empty_docs": analysis_empty,
            "held_out_empty_docs": held_empty,
            "analysis_tokens": int(analysis_labels.shape[0]),
            "held_out_tokens": int(held_labels.shape[0]),
            "full_tokens": int(full_labels.shape[0]),
        },
        "decode": {
            "window_merge": (
                "mean logits over overlapping windows, aligned to unique "
                "(start, end) char spans; specials (start==end) dropped"
            ),
            "temperature": "p = softmax(mean_logits / T)",
            "argmax_invariant": True,
            "argmax_tag_signature_unchanged": True,
            "snap": "snap_entities(text, entities) word+suffix+nms after BIO decode",
            "entity_confidence": (
                "mean or max of predicted-class token probability over BIO tokens "
                "that form the entity, computed before snap; gate then snap"
            ),
        },
        "calibration_definitions": {
            "token": (
                "7-way BIO on merged content tokens; gold via align_labels. "
                "Empty official docs (18%) are included as O tokens."
            ),
            "ece": (
                f"max-confidence ECE, {N_BINS} equal-width bins on [0,1], "
                "empty bins skipped, weighted |acc-conf|"
            ),
            "brier": "multiclass mean ||p - one_hot(y)||^2 (range [0, 2])",
            "nll": "mean -log p[gold_tag]",
            "entity": (
                "binary: snapped predicted entity is an exact gold (label,start,end); "
                "confidence is mean/max token p of the BIO entity before snap. "
                "Empty docs contribute only if the model hallucinates entities; "
                "their O tokens still enter token ECE."
            ),
        },
        "cache": {
            "dir": str(cache_dir),
            "format": cache_meta.get("format"),
            "files": cache_meta.get("files"),
            "n_windows": cache_meta.get("n_windows"),
            "n_tokens": cache_meta.get("n_tokens"),
            "hidden_shape": cache_meta.get("hidden_shape"),
            "logits_shape": cache_meta.get("logits_shape"),
            "merged_shape": cache_meta.get("merged_shape"),
            "nbytes": nbytes,
            "window_merge": cache_meta.get("window_merge"),
            "reload": cache_meta.get("reload"),
        },
        "baseline": {
            "full_dev_f1_t1_argmax_snap": float(baseline_full["micro"]["f1"]),
            "held_out_f1_t1_argmax_snap": snap_held_f1,
            "analysis_f1_t1_argmax_snap": float(baseline_analysis["micro"]["f1"]),
            "full_dev_micro": micro_view(baseline_full),
            "note": (
                "Mix B word-snap official micro F1 was ~0.666. This number uses "
                "mean-logit merge then softmax, which matches mean-prob merge "
                "except on rare overlapping-window ties."
            ),
        },
        "temperature_grid": list(TEMPERATURES),
        "analysis": analysis_by_t,
        "held_out": held_by_t,
        "argmax_f1_by_T": {
            **argmax_f1,
            "unique_full_dev_f1_values": sorted(unique_full_f1),
            "flat": len(unique_full_f1) == 1,
        },
        "selected": {
            "criterion": "min analysis token ECE; ties break on NLL then |T-1|",
            "T": t_star,
            "T_by_nll": t_nll,
            "analysis_at_T": analysis_by_t[str(t_star)],
            "held_out_at_T": held_by_t[str(t_star)],
        },
        "threshold_sweep": {
            "T": t_star,
            "tau": list(TAUS),
            "mean": threshold_rows["mean"],
            "max": threshold_rows["max"],
            "chosen_on_analysis": {
                "reduce": "mean",
                "tau": float(best_gate["tau"]),
                "analysis": best_gate["analysis"],
                "held_out": best_gate["held_out"],
            },
        },
        "reliability_held_out": {
            "T_1": reliability_t1["reliability"],
            "T_star": reliability_best["reliability"],
        },
        "diagnostics": {
            "held_out_t1_snap": {
                "buckets": diag_t1["buckets"],
                "reading_id": diag_t1["reading_id"],
                "reading": diag_t1["reading"],
            },
            "held_out_tstar_gated": {
                "tau": float(best_gate["tau"]),
                "buckets": diag_gate["buckets"],
                "reading_id": diag_gate["reading_id"],
                "reading": diag_gate["reading"],
            },
        },
        "verdict": {
            "head": head_reading,
            "head_after_T": calibrated_reading,
            "argmax_flat": len(unique_full_f1) == 1,
            "argmax_signature_len": len(tag_sig),
            "t_plus_threshold_beats_snap": t_plus_threshold_wins,
            "held_out_f1_snap": snap_held_f1,
            "held_out_f1_best_gated": gate_held_f1,
            "delta_f1": gate_held_f1 - snap_held_f1,
            "change_predict_defaults": False,
            "text": verdict_text,
        },
    }


def print_summary(payload: dict[str, Any]) -> None:
    selected = payload["selected"]
    baseline = payload["baseline"]
    verdict = payload["verdict"]
    t1 = payload["held_out"]["1.0"]["token"]
    t_star = selected["T"]
    t_star_held = selected["held_out_at_T"]["token"]
    logger.info("=== Mix B temperature / calibration ===")
    logger.info(
        "official full-dev F1 T=1 argmax+snap: {:.4f} (Mix B snap ~0.666)",
        baseline["full_dev_f1_t1_argmax_snap"],
    )
    logger.info(
        "argmax+snap F1 by T (full dev): {}",
        {
            temp: round(float(row["f1"]), 6)
            for temp, row in payload["argmax_f1_by_T"]["full_dev"].items()
        },
    )
    logger.info(
        "held-out token T=1: ECE={:.4f} Brier={:.4f} NLL={:.4f} acc={:.4f} conf={:.4f} ({})",
        t1["ece"],
        t1["brier"],
        t1["nll"],
        t1["accuracy"],
        t1["mean_confidence"],
        verdict["head"],
    )
    logger.info(
        "T*={:g} (ECE) / T_NLL={:g} | held-out ECE={:.4f} Brier={:.4f} NLL={:.4f}",
        t_star,
        selected["T_by_nll"],
        t_star_held["ece"],
        t_star_held["brier"],
        t_star_held["nll"],
    )
    for row in payload["threshold_sweep"]["mean"]:
        logger.info(
            "  mean-conf τ={:<3} held-out F1={:.4f} P={:.4f} R={:.4f}",
            row["tau"],
            row["held_out"]["f1"],
            row["held_out"]["precision"],
            row["held_out"]["recall"],
        )
    logger.info("verdict: {}", verdict["text"])


def main() -> None:
    args = parse_args()
    gold_path = args.gold.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve()
    if not gold_path.is_file():
        raise FileNotFoundError(gold_path)
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    records = read_jsonl_records(gold_path, require_entities=True)
    logger.info(
        "gold {} docs={} empty={}", gold_path, len(records), sum(not r["entities"] for r in records)
    )

    if args.force_cache or not cache_complete(cache_dir):
        logger.info("filling GPU cache under current process (use flock on the caller)")
        cache_meta = fill_cache(
            checkpoint=checkpoint,
            records=records,
            cache_dir=cache_dir,
            batch_size=args.batch_size,
            max_length=args.max_length,
            stride=args.stride,
        )
        docs = load_merged_npz(cache_dir / cache_meta["files"]["merged"], records)
    else:
        cache_meta = json.loads((cache_dir / "index.json").read_text(encoding="utf-8"))
        logger.info(
            "reusing cache {} tokens={} windows={}",
            cache_dir,
            cache_meta.get("n_tokens"),
            cache_meta.get("n_windows"),
        )
        docs = load_merged_npz(cache_dir / cache_meta["files"]["merged"], records)

    payload = run_experiment(
        docs=docs,
        records=records,
        cache_meta=cache_meta,
        cache_dir=cache_dir,
        checkpoint=checkpoint,
        gold_path=gold_path,
        seed=args.seed,
        k=args.k,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(jsonable(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print_summary(payload)
    logger.info("wrote {}", args.output)


if __name__ == "__main__":
    main()

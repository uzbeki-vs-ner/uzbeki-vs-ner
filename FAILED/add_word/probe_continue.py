#!/usr/bin/env python3
"""Fossil probe. Not part of the package. Replay is optional:

    PYTHONPATH=FAILED/add_word uv run python FAILED/add_word/probe_continue.py

One-word continuation from leftover I-mass on Mix B official-dev cache.

Word snap cannot cross spaces. This probe grows a snapped span into the next
or previous word when that word is currently unoccupied and still has mass on
B/I of the same label. It does not glue two already-predicted entities.

Sweep τ on the analysis fold (seed 42, 5-fold, fold 0). Held-out F1 is the
score. Never writes checkpoints. CPU-only if the cache already exists.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from continue_span import (  # type: ignore[import-not-found]
    Direction,
    continue_one_word,
    entity_mass,
    next_word,
    prev_word,
)
from uzbek_ner.decode.kfold import make_folds, score_split
from uzbek_ner.decode.snap import is_word_char, snap_entities
from uzbek_ner.io.jsonl import read_jsonl_records
from uzbek_ner.labels import ID_TO_TAG, TAGS
from uzbek_ner.metrics.calibration import temperature_softmax
from uzbek_ner.metrics.error_analysis import analyze_span_errors
from uzbek_ner.settings import REPO_ROOT, get_settings
from uzbek_ner.spans import decode_bio_tokens

N_LABELS = len(TAGS)
TAUS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
DIRECTIONS = ("right", "left", "both")
GATES = (0.0, 0.7)
LABEL_SETS: tuple[tuple[str, ...] | None, ...] = (None, ("NAME",))


@dataclass
class MergedDoc:
    record_hash: str
    text: str
    offsets: NDArray[np.int32]
    logits: NDArray[np.float32]
    gold_keys: set[tuple[str, int, int]]


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, default=settings.official_dev)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "cache" / "mixb_official_dev",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs" / "eval" / "mixb_continue.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=5)
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
    if isinstance(value, set):
        return [jsonable(item) for item in value]
    return value


def micro_view(metrics: dict[str, Any]) -> dict[str, float | int]:
    micro = metrics["micro"]
    return {
        "precision": float(micro["precision"]),
        "recall": float(micro["recall"]),
        "f1": float(micro["f1"]),
        "tp": int(micro["tp"]),
        "fp": int(micro["fp"]),
        "fn": int(micro["fn"]),
    }


def gold_lookup(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        record["hash"]: {
            "text": record["text"],
            "entities": {
                (str(row["label"]), int(row["start"]), int(row["end"]))
                for row in record["entities"]
            },
        }
        for record in records
    }


def keyed_predictions(
    rows: dict[str, list[dict[str, Any]]],
) -> dict[str, set[tuple[str, int, int]]]:
    return {
        record_hash: {(str(row["label"]), int(row["start"]), int(row["end"])) for row in entities}
        for record_hash, entities in rows.items()
    }


def load_merged(cache_dir: Path, records: list[dict[str, Any]]) -> list[MergedDoc]:
    meta = json.loads((cache_dir / "index.json").read_text(encoding="utf-8"))
    if meta.get("status") != "complete":
        raise RuntimeError(f"cache is not complete: {cache_dir}")
    packed = np.load(cache_dir / meta["files"]["merged"], allow_pickle=False)
    by_hash = {record["hash"]: record for record in records}
    hashes = list(meta["hashes"])
    ptr = np.asarray(packed["ptr"])
    if len(hashes) != int(ptr.shape[0]) - 1:
        raise RuntimeError("merged doc count does not match cache hashes")
    docs: list[MergedDoc] = []
    for index, record_hash in enumerate(hashes):
        record = by_hash[record_hash]
        start = int(ptr[index])
        end = int(ptr[index + 1])
        docs.append(
            MergedDoc(
                record_hash=record_hash,
                text=record["text"],
                offsets=np.asarray(packed["offsets"][start:end], dtype=np.int32),
                logits=np.asarray(packed["logits"][start:end], dtype=np.float32),
                gold_keys={
                    (str(row["label"]), int(row["start"]), int(row["end"]))
                    for row in record["entities"]
                },
            )
        )
    return docs


def entity_conf(
    offsets: NDArray[np.int32],
    tags: NDArray[np.integer],
    probs: NDArray[np.floating],
    entity: dict[str, Any],
) -> float:
    start, end = int(entity["start"]), int(entity["end"])
    indices = [
        index
        for index, (tok_s, tok_e) in enumerate(offsets)
        if int(tok_s) >= start and int(tok_e) <= end and int(tok_s) < int(tok_e)
    ]
    if not indices:
        return 0.0
    values = np.asarray(
        [float(probs[index, int(tags[index])]) for index in indices], dtype=np.float64
    )
    return float(values.mean())


def decode_snapped(
    doc: MergedDoc,
    *,
    min_conf: float,
) -> tuple[list[dict[str, Any]], NDArray[np.float64]]:
    if doc.logits.shape[0] == 0:
        return [], np.zeros((0, N_LABELS), dtype=np.float64)
    probs = temperature_softmax(doc.logits, 1.0)
    tags = probs.argmax(axis=1)
    tagged = [
        (int(start), int(end), ID_TO_TAG[int(tag)])
        for (start, end), tag in zip(doc.offsets, tags, strict=True)
    ]
    entities = decode_bio_tokens(tagged)
    if min_conf > 0:
        entities = [
            entity
            for entity in entities
            if entity_conf(doc.offsets, tags, probs, entity) >= min_conf
        ]
    return snap_entities(doc.text, entities), probs


def apply_continue(
    text: str,
    snapped: list[dict[str, Any]],
    offsets: NDArray[np.int32],
    probs: NDArray[np.floating],
    *,
    tau: float,
    direction: Direction,
    labels: tuple[str, ...] | None,
) -> list[dict[str, Any]]:
    if not snapped:
        return snapped
    return continue_one_word(
        text,
        snapped,
        offsets,
        probs,
        tau=tau,
        direction=direction,
        labels=labels,
        max_words=1,
    )


def iter_words(text: str, start: int, end: int) -> list[tuple[int, int]]:
    words: list[tuple[int, int]] = []
    index = start
    while index < end:
        if not is_word_char(text[index]):
            index += 1
            continue
        stop = index
        while stop < end and is_word_char(text[stop]):
            stop += 1
        words.append((index, stop))
        index = stop
    return words


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = q * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return ordered[low] * (1.0 - frac) + ordered[high] * frac


def _iou(start_a: int, end_a: int, start_b: int, end_b: int) -> float:
    inter = min(end_a, end_b) - max(start_a, start_b)
    if inter <= 0:
        return 0.0
    union = (end_a - start_a) + (end_b - start_b) - inter
    return inter / union if union else 0.0


def leftover_mass_probe(docs: list[MergedDoc], hashes: set[str]) -> dict[str, Any]:
    """Gold-aware: I-mass on the extra gold word that snap still drops."""

    adjacent_masses: list[float] = []
    adjacent_right = 0
    adjacent_left = 0
    multiword_narrower = 0
    no_token = 0
    for doc in docs:
        if doc.record_hash not in hashes or doc.logits.shape[0] == 0:
            continue
        snapped, probs = decode_snapped(doc, min_conf=0.0)
        for gold_label, gold_start, gold_end in doc.gold_keys:
            gold_words = iter_words(doc.text, gold_start, gold_end)
            if len(gold_words) < 2:
                continue
            best: tuple[int, int] | None = None
            best_iou = 0.0
            for row in snapped:
                pred_label = str(row["label"])
                pred_start, pred_end = int(row["start"]), int(row["end"])
                if pred_label != gold_label:
                    continue
                if pred_start < gold_start or pred_end > gold_end:
                    continue
                if pred_start == gold_start and pred_end == gold_end:
                    continue
                score = _iou(gold_start, gold_end, pred_start, pred_end)
                if score > best_iou:
                    best = (pred_start, pred_end)
                    best_iou = score
            if best is None:
                continue
            multiword_narrower += 1
            pred_start, pred_end = best
            nxt = next_word(doc.text, pred_end)
            prv = prev_word(doc.text, pred_start)
            dropped = [word for word in gold_words if word[1] <= pred_start or word[0] >= pred_end]
            for word in dropped:
                if word not in {nxt, prv}:
                    continue
                if word == nxt:
                    adjacent_right += 1
                else:
                    adjacent_left += 1
                mass = entity_mass(gold_label, word[0], word[1], doc.offsets, probs)
                has_token = any(
                    int(tok_e) > word[0] and int(tok_s) < word[1] for tok_s, tok_e in doc.offsets
                )
                if not has_token:
                    no_token += 1
                adjacent_masses.append(mass)
    n = len(adjacent_masses)
    return {
        "multiword_gold_narrower_pairs": multiword_narrower,
        "adjacent_dropped_words": n,
        "adjacent_right": adjacent_right,
        "adjacent_left": adjacent_left,
        "adjacent_no_overlapping_token": no_token,
        "mass_mean": (sum(adjacent_masses) / n) if n else 0.0,
        "mass_p25": percentile(adjacent_masses, 0.25),
        "mass_p50": percentile(adjacent_masses, 0.50),
        "mass_p75": percentile(adjacent_masses, 0.75),
        "share_mass_ge": {
            str(tau): (sum(1 for mass in adjacent_masses if mass >= tau) / n if n else 0.0)
            for tau in TAUS
        },
    }


def predict_continue(
    docs: list[MergedDoc],
    cached: dict[str, tuple[list[dict[str, Any]], NDArray[np.floating]]],
    *,
    tau: float,
    direction: Direction,
    labels: tuple[str, ...] | None,
) -> dict[str, list[dict[str, Any]]]:
    return {
        doc.record_hash: apply_continue(
            doc.text,
            cached[doc.record_hash][0],
            doc.offsets,
            cached[doc.record_hash][1],
            tau=tau,
            direction=direction,
            labels=labels,
        )
        for doc in docs
    }


def label_set_name(labels: tuple[str, ...] | None) -> str:
    return "all" if labels is None else "+".join(labels)


def run() -> dict[str, Any]:
    args = parse_args()
    records = read_jsonl_records(args.gold, require_entities=True)
    docs = load_merged(args.cache_dir, records)
    gold = gold_lookup(records)
    hashes = [doc.record_hash for doc in docs]
    folds = make_folds(hashes, k=args.k, seed=args.seed)
    analysis_hashes = list(folds[0])
    held_out_hashes = [record_hash for fold in folds[1:] for record_hash in fold]
    analysis_set = set(analysis_hashes)

    baselines: dict[str, Any] = {}
    decoded: dict[str, dict[str, tuple[list[dict[str, Any]], NDArray[np.floating]]]] = {}
    for gate in GATES:
        cached: dict[str, tuple[list[dict[str, Any]], NDArray[np.floating]]] = {}
        for doc in docs:
            cached[doc.record_hash] = decode_snapped(doc, min_conf=gate)
        decoded[str(gate)] = cached
        preds = {record_hash: snapped for record_hash, (snapped, _probs) in cached.items()}
        keyed = keyed_predictions(preds)
        baselines[str(gate)] = {
            "analysis": micro_view(score_split(gold, keyed, analysis_hashes)),
            "held_out": micro_view(score_split(gold, keyed, held_out_hashes)),
            "full_dev": micro_view(score_split(gold, keyed, hashes)),
            "held_out_errors": analyze_span_errors(
                {record_hash: gold[record_hash] for record_hash in held_out_hashes},
                {record_hash: keyed[record_hash] for record_hash in held_out_hashes},
            ),
        }
        logger.info(
            "baseline gate={} analysis F1={:.4f} held-out F1={:.4f}",
            gate,
            baselines[str(gate)]["analysis"]["f1"],
            baselines[str(gate)]["held_out"]["f1"],
        )

    leftover = leftover_mass_probe(docs, analysis_set)
    logger.info(
        "leftover adjacent words={} mass p50={:.3f} p75={:.3f}",
        leftover["adjacent_dropped_words"],
        leftover["mass_p50"],
        leftover["mass_p75"],
    )

    grid: list[dict[str, Any]] = []
    for gate in GATES:
        for labels in LABEL_SETS:
            for direction in DIRECTIONS:
                for tau in TAUS:
                    cached = decoded[str(gate)]
                    preds = predict_continue(
                        docs,
                        cached,
                        tau=tau,
                        direction=direction,
                        labels=labels,
                    )
                    keyed = keyed_predictions(preds)
                    analysis = micro_view(score_split(gold, keyed, analysis_hashes))
                    held = micro_view(score_split(gold, keyed, held_out_hashes))
                    n_grown = 0
                    for record_hash, entities in preds.items():
                        before = {
                            (str(row["label"]), int(row["start"]), int(row["end"]))
                            for row in cached[record_hash][0]
                        }
                        after = {
                            (str(row["label"]), int(row["start"]), int(row["end"]))
                            for row in entities
                        }
                        n_grown += len(after - before)
                    row = {
                        "gate": gate,
                        "labels": label_set_name(labels),
                        "direction": direction,
                        "tau": tau,
                        "analysis": analysis,
                        "held_out": held,
                        "n_spans_changed_vs_snap": n_grown,
                    }
                    grid.append(row)
                    logger.info(
                        "gate={} labels={} dir={} τ={} analysis={:.4f} held={:.4f} Δspans={}",
                        gate,
                        label_set_name(labels),
                        direction,
                        tau,
                        analysis["f1"],
                        held["f1"],
                        n_grown,
                    )

    by_gate: dict[str, dict[str, Any]] = {}
    for gate in GATES:
        subset = [row for row in grid if row["gate"] == gate]
        best = max(subset, key=lambda item: (item["analysis"]["f1"], -item["tau"]))
        baseline_f1 = float(baselines[str(gate)]["held_out"]["f1"])
        by_gate[str(gate)] = {
            "picked_on_analysis": {
                "labels": best["labels"],
                "direction": best["direction"],
                "tau": best["tau"],
                "analysis_f1": best["analysis"]["f1"],
            },
            "held_out": best["held_out"],
            "held_out_delta_f1": float(best["held_out"]["f1"]) - baseline_f1,
            "baseline_held_out_f1": baseline_f1,
        }
        logger.info(
            "pick gate={} {} {} τ={} → held-out F1={:.4f} (Δ {:+.4f})",
            gate,
            best["labels"],
            best["direction"],
            best["tau"],
            best["held_out"]["f1"],
            by_gate[str(gate)]["held_out_delta_f1"],
        )

    overall = max(grid, key=lambda item: (item["analysis"]["f1"], -item["tau"]))
    verdict = "skip"
    note = "continuation did not beat snap (and optional conf-gate) on held-out F1"
    best_held = max(grid, key=lambda item: item["held_out"]["f1"])
    if (
        float(overall["held_out"]["f1"])
        > float(baselines[str(overall["gate"])]["held_out"]["f1"]) + 0.002
    ):
        verdict = "wire_behind_flag"
        note = (
            "analysis-picked continuation beats the matching baseline on held-out F1; "
            "keep it behind a decode flag, do not change the default yet"
        )
    return {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "cache_dir": str(args.cache_dir),
        "gold": str(args.gold),
        "split": {
            "method": "5-fold round-robin, fold 0 = analysis, folds 1..k-1 = held-out",
            "seed": args.seed,
            "k": args.k,
            "analysis_docs": len(analysis_hashes),
            "held_out_docs": len(held_out_hashes),
        },
        "decode": {
            "baseline": "argmax BIO → optional mean-token conf gate → word+suffix snap",
            "continue": (
                "after snap, grow at most one neighboring space-separated word "
                "if mean P(B-L)+P(I-L) ≥ τ; refuse occupied words and punctuation jumps"
            ),
        },
        "leftover_i_mass_analysis": leftover,
        "baselines": baselines,
        "grid": grid,
        "picked": by_gate,
        "analysis_best": {
            "gate": overall["gate"],
            "labels": overall["labels"],
            "direction": overall["direction"],
            "tau": overall["tau"],
            "analysis": overall["analysis"],
            "held_out": overall["held_out"],
        },
        "oracle_held_out_best": {
            "gate": best_held["gate"],
            "labels": best_held["labels"],
            "direction": best_held["direction"],
            "tau": best_held["tau"],
            "held_out": best_held["held_out"],
            "note": "not a valid pick — shown only to bound the idea",
        },
        "verdict": verdict,
        "note": note,
    }


def main() -> None:
    payload = run()
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(jsonable(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info("wrote {}", args.output)


if __name__ == "__main__":
    main()

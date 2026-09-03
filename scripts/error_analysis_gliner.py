#!/usr/bin/env python3
"""GLiNER error analysis on official dev (exact-span + scored spans).

Reads ``outputs/cache/gliner_multi_v21_official_dev/scored_spans.json``,
applies a confidence floor ``--tau``, and writes a human-readable report
similar to ``outputs/error_analysis.txt`` for uztext.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from uzbek_ner.io.jsonl import read_jsonl_records
from uzbek_ner.metrics.error_analysis import (
    ErrorKind,
    analyze_span_errors,
    classify_predicted_span,
    gold_merge_note,
)
from uzbek_ner.metrics.exact_span import EntityKey
from uzbek_ner.settings import REPO_ROOT, get_settings

JsonObject = dict[str, Any]


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, default=settings.official_dev)
    parser.add_argument(
        "--scored-cache",
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
        default=REPO_ROOT / "outputs" / "error_analysis_gliner.txt",
    )
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument(
        "--fp-conf",
        type=float,
        default=0.7,
        help="Group 1: non-exact predictions with score >= this",
    )
    parser.add_argument("--tp-conf-lo", type=float, default=0.5, help="Group 2 lower bound")
    parser.add_argument(
        "--tp-conf-hi", type=float, default=0.7, help="Group 2 upper bound (exclusive)"
    )
    parser.add_argument("--examples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--context", type=int, default=80, help="Chars of context on each side")
    return parser.parse_args()


def load_scored(cache_path: Path) -> tuple[dict[str, list[JsonObject]], dict[str, Any]]:
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    scored = {row["hash"]: row["spans"] for row in payload["docs"]}
    return scored, payload.get("meta", {})


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


def score_map(
    scored: dict[str, list[JsonObject]],
    *,
    tau: float,
) -> dict[EntityKey, float]:
    out: dict[EntityKey, float] = {}
    for spans in scored.values():
        for row in spans:
            if float(row["score"]) < tau:
                continue
            key = (str(row["label"]), int(row["start"]), int(row["end"]))
            out[key] = float(row["score"])
    return out


def format_entity(label: str, start: int, end: int, text: str) -> str:
    return f"{label} [{start}:{end}] {text[start:end]!r}"


def context_window(text: str, start: int, end: int, *, radius: int) -> str:
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    before = text[left:start]
    span = text[start:end]
    after = text[end:right]
    return f"…{before}⟦{span}⟧{after}…"


def confidence_bins(
    scored: dict[str, list[JsonObject]],
    gold_entities: dict[str, set[EntityKey]],
    *,
    tau: float,
    width: float = 0.1,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n_bins = round(1.0 / width)
    for bin_index in range(n_bins):
        lo = round(bin_index * width, 10)
        hi = round(lo + width, 10)
        if bin_index == n_bins - 1:
            hi = 1.0 + 1e-9
        tp = fp = 0
        for record_hash, spans in scored.items():
            gold_keys = gold_entities[record_hash]
            for row in spans:
                score = float(row["score"])
                if score < max(lo, tau) or score >= hi:
                    continue
                key = (str(row["label"]), int(row["start"]), int(row["end"]))
                if key in gold_keys:
                    tp += 1
                else:
                    fp += 1
        n = tp + fp
        rows.append(
            {
                "lo": lo,
                "hi": min(hi, 1.0),
                "n": n,
                "tp": tp,
                "fp": fp,
                "precision": (tp / n) if n else None,
            }
        )
    return rows


def partial_fp_patterns(
    gold: dict[str, JsonObject],
    predictions: dict[str, set[EntityKey]],
) -> Counter[str]:
    """Classify same-type partial-overlap FPs (pred not exact gold)."""

    patterns: Counter[str] = Counter()

    def iou(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
        inter = min(a_end, b_end) - max(a_start, b_start)
        if inter <= 0:
            return 0.0
        union = (a_end - a_start) + (b_end - b_start) - inter
        return inter / union if union else 0.0

    for record_hash, gold_record in gold.items():
        gold_entities = gold_record["entities"]
        pred_entities = predictions[record_hash]
        matched = gold_entities & pred_entities
        gold_rest = list(gold_entities - matched)
        pred_rest = list(pred_entities - matched)

        used_gold: set[int] = set()
        used_pred: set[int] = set()
        for g_i, (g_label, g_start, g_end) in enumerate(gold_rest):
            if g_i in used_gold:
                continue
            best: tuple[float, int] | None = None
            for p_i, (p_label, p_start, p_end) in enumerate(pred_rest):
                if p_i in used_pred or p_label != g_label:
                    continue
                score = iou(g_start, g_end, p_start, p_end)
                if score <= 0:
                    continue
                if best is None or score > best[0]:
                    best = (score, p_i)
            if best is None:
                continue
            _, p_i = best
            used_gold.add(g_i)
            used_pred.add(p_i)
            _p_label, p_start, p_end = pred_rest[p_i]
            if p_start > g_start and p_end < g_end:
                patterns["pred_narrower"] += 1
            elif p_start < g_start and p_end > g_end:
                patterns["pred_wider"] += 1
            elif p_start > g_start:
                patterns["pred_miss_left"] += 1
            elif p_end < g_end:
                patterns["pred_miss_right"] += 1
            else:
                patterns["pred_shifted"] += 1
    return patterns


def label_bucket_counts(
    gold: dict[str, JsonObject],
    predictions: dict[str, set[EntityKey]],
) -> dict[str, dict[str, int]]:
    by_label: dict[str, dict[str, int]] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0, "gold": 0, "pred": 0}
    )
    for record_hash, gold_record in gold.items():
        gold_entities = gold_record["entities"]
        pred_entities = predictions[record_hash]
        for label in ("ORG", "NAME", "GEO"):
            gold_label = {entity for entity in gold_entities if entity[0] == label}
            pred_label = {entity for entity in pred_entities if entity[0] == label}
            by_label[label]["gold"] += len(gold_label)
            by_label[label]["pred"] += len(pred_label)
            by_label[label]["tp"] += len(gold_label & pred_label)
            by_label[label]["fp"] += len(pred_label - gold_label)
            by_label[label]["fn"] += len(gold_label - pred_label)
    return dict(by_label)


def pick_examples(
    pool: list[JsonObject],
    *,
    n: int,
    seed: int,
) -> list[JsonObject]:
    if not pool:
        return []
    rng = random.Random(seed)
    by_label: dict[str, list[JsonObject]] = defaultdict(list)
    for row in pool:
        by_label[str(row["label"])].append(row)
    picked: list[JsonObject] = []
    for label in ("ORG", "NAME", "GEO"):
        if by_label[label]:
            picked.append(rng.choice(by_label[label]))
    rest = [row for row in pool if row not in picked]
    rng.shuffle(rest)
    picked.extend(rest[: max(0, n - len(picked))])
    return picked[:n]


def format_gold_relation(row: JsonObject) -> list[str]:
    """Human-readable gold lines for a non-exact predicted span."""

    text = str(row["text"])
    error_kind: ErrorKind = row["error_kind"]
    gold_hits: list[EntityKey] = row["gold_hits"]

    if error_kind == "type_mismatch":
        gold_key = gold_hits[0]
        return [
            "gold:     "
            + format_entity(gold_key[0], gold_key[1], gold_key[2], text)
            + "  (type mismatch — same span)"
        ]

    if error_kind == "partial_same_type":
        lines = [
            "gold:     " + " | ".join(format_entity(g[0], g[1], g[2], text) for g in gold_hits)
        ]
        if row.get("merge_note"):
            lines.append(f"note:     {row['merge_note']}")
        return lines

    if error_kind == "partial_diff_type":
        return [
            "gold:     "
            + " | ".join(format_entity(g[0], g[1], g[2], text) for g in gold_hits)
            + "  (partial overlap, different type)"
        ]

    return ["gold:     (none overlapping — spurious)"]


def build_pools(
    records: list[JsonObject],
    gold: dict[str, JsonObject],
    scores: dict[EntityKey, float],
    predictions: dict[str, set[EntityKey]],
    *,
    fp_conf: float,
    tp_conf_lo: float,
    tp_conf_hi: float,
) -> tuple[list[JsonObject], list[JsonObject], list[JsonObject], Counter[str]]:
    text_by_hash = {row["hash"]: row["text"] for row in records}

    group1: list[JsonObject] = []
    group1_kinds: Counter[str] = Counter()
    group2: list[JsonObject] = []
    for record_hash, pred_set in predictions.items():
        text = text_by_hash[record_hash]
        gold_entities = gold[record_hash]["entities"]
        for key in pred_set:
            if key not in scores:
                continue
            conf = scores[key]
            label, start, end = key
            error_kind, gold_hits = classify_predicted_span(key, gold_entities)
            row: JsonObject = {
                "hash": record_hash,
                "label": label,
                "start": start,
                "end": end,
                "conf": conf,
                "text": text,
                "span_text": text[start:end],
                "error_kind": error_kind,
                "gold_hits": gold_hits,
            }
            if error_kind == "exact":
                if tp_conf_lo <= conf < tp_conf_hi:
                    group2.append(row)
                continue
            if conf < fp_conf:
                continue
            if error_kind == "partial_same_type":
                row["merge_note"] = gold_merge_note(key, gold_hits, text)
            group1.append(row)
            group1_kinds[error_kind] += 1

    group3: list[JsonObject] = []
    for record_hash, gold_record in gold.items():
        text = gold_record["text"]
        gold_set = gold_record["entities"]
        pred_set = predictions[record_hash]

        for key in gold_set:
            label, start, end = key
            if any(
                end > pred_start and start < pred_end for (_pl, pred_start, pred_end) in pred_set
            ):
                continue
            others = [
                f"{pred[0]} [{pred[1]}:{pred[2]}] {text[pred[1] : pred[2]]!r} "
                f"conf={scores.get(pred, float('nan')):.3f}"
                for pred in sorted(pred_set, key=lambda item: (item[1], item[2]))
            ]
            group3.append(
                {
                    "hash": record_hash,
                    "label": label,
                    "start": start,
                    "end": end,
                    "text": text,
                    "span_text": text[start:end],
                    "other_pred": others[:6],
                }
            )
    return group1, group2, group3, group1_kinds


def render_examples(
    title: str,
    rows: list[JsonObject],
    *,
    context: int,
    seed: int,
    n: int,
    group: int,
) -> list[str]:
    lines = ["=" * 78, title, "=" * 78]
    picked = pick_examples(rows, n=min(n, len(rows)), seed=seed)
    for index, row in enumerate(picked, 1):
        lines.append(f"--- example {index} ---")
        lines.append(f"hash:     {row['hash']}")
        if "conf" in row:
            lines.append(f"score:    {row['conf']:.4f}")
        if group == 1:
            lines.append(f"kind:     {row['error_kind']}")
        lines.append(f"pred:     {row['label']} [{row['start']}:{row['end']}] {row['span_text']!r}")
        if group == 1:
            lines.extend(format_gold_relation(row))
        elif "other_pred" in row:
            lines.append("pred:     (none overlapping this gold span)")
            lines.append(
                f"gold:     {row['label']} [{row['start']}:{row['end']}] {row['span_text']!r}"
            )
            if row["other_pred"]:
                lines.append("other pred in doc: " + " | ".join(row["other_pred"]))
        elif group == 2:
            lines.append(
                f"gold:     {row['label']} [{row['start']}:{row['end']}] {row['span_text']!r}  (exact)"
            )
        lines.append(
            f"context:  {context_window(row['text'], row['start'], row['end'], radius=context)}"
        )
        lines.append("")
    return lines


def main() -> None:
    args = parse_args()
    records = read_jsonl_records(args.gold)
    scored, meta = load_scored(args.scored_cache)
    gold_lookup_map = {
        record["hash"]: {
            "text": record["text"],
            "entities": {
                (str(row["label"]), int(row["start"]), int(row["end"]))
                for row in record["entities"]
            },
        }
        for record in records
    }
    predictions = filter_scored(scored, tau=args.tau)
    scores = score_map(scored, tau=args.tau)
    diagnostics = analyze_span_errors(gold_lookup_map, predictions)
    buckets = diagnostics["buckets"]
    exact_tp = buckets["exact_match"]
    pred_spans = sum(len(value) for value in predictions.values())
    gold_no_overlap = 0
    for record_hash, gold_record in gold_lookup_map.items():
        pred_set = predictions[record_hash]
        for gold_key in gold_record["entities"]:
            gs, ge = gold_key[1], gold_key[2]
            if not any(ge > ps and gs < pe for (_pl, ps, pe) in pred_set):
                gold_no_overlap += 1

    bins = confidence_bins(
        scored,
        {record_hash: record["entities"] for record_hash, record in gold_lookup_map.items()},
        tau=args.tau,
    )
    partial = partial_fp_patterns(gold_lookup_map, predictions)
    by_label = label_bucket_counts(gold_lookup_map, predictions)

    group1, group2, group3, group1_kinds = build_pools(
        records,
        gold_lookup_map,
        scores,
        predictions,
        fp_conf=args.fp_conf,
        tp_conf_lo=args.tp_conf_lo,
        tp_conf_hi=args.tp_conf_hi,
    )

    lines: list[str] = [
        "Error analysis · gliner_multi_v21 · official dev",
        f"scored cache: {args.scored_cache}",
        f"tau (score floor): {args.tau}",
        "score: GLiNER span score from predict_entities; no BIO / snap post-process",
        "TP = exact (label, start, end). Non-exact predictions map to diagnostic buckets.",
        (
            f"predicted spans: {pred_spans} (exact TP={exact_tp}, "
            f"non-exact={pred_spans - exact_tp}); "
            f"gold with no overlapping pred: {gold_no_overlap}"
        ),
        "",
        f"reading: {diagnostics['reading_id']} — {diagnostics['reading']}",
        "",
        "Diagnostic buckets (organizer exact-span is only exact_match):",
        (
            f"  exact_match={buckets['exact_match']}  type_mismatch={buckets['type_mismatch']}  "
            f"partial_same_type={buckets['partial_same_type']}  "
            f"partial_diff_type={buckets['partial_diff_type']}  "
            f"missed={buckets['missed']}  spurious={buckets['spurious']}"
        ),
        (
            f"  boundary_exact F1={diagnostics['boundary_exact']['f1']:.4f}  "
            f"type@boundary acc={diagnostics['type_given_boundary']['accuracy']:.4f}"
        ),
        "",
        "Per-label (exact-span):",
    ]
    for label in ("ORG", "NAME", "GEO"):
        row = by_label[label]
        precision = row["tp"] / (row["tp"] + row["fp"]) if row["tp"] + row["fp"] else 0.0
        recall = row["tp"] / (row["tp"] + row["fn"]) if row["tp"] + row["fn"] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        lines.append(
            f"  {label}: gold={row['gold']} pred={row['pred']} "
            f"TP={row['tp']} FP={row['fp']} FN={row['fn']} "
            f"P={precision:.3f} R={recall:.3f} F1={f1:.3f}"
        )
    lines.extend(
        [
            "",
            "Same-type partial-overlap FP patterns (pred paired to overlapping gold):",
        ]
    )
    for key, count in partial.most_common():
        lines.append(f"  {key}: {count}")
    lines.extend(
        [
            "",
            "1. Confidence bins (width 0.1, spans with score >= tau)",
            f"{'bin':<14} {'n':>6} {'TP':>6} {'FP':>6}  {'Precision':>10}",
        ]
    )
    for row in bins:
        if row["n"] == 0:
            precision = "—"
        else:
            precision = f"{row['precision']:.4f}" if row["precision"] is not None else "—"
        hi_display = 1.0 if row["hi"] >= 1.0 else row["hi"]
        bin_label = f"[{row['lo']:.1f}, {hi_display:.1f}]"
        lines.append(f"{bin_label:<14}{row['n']:>6}{row['tp']:>6}{row['fp']:>6}  {precision:>10}")
    group1_breakdown = "  ".join(f"{kind}={count}" for kind, count in group1_kinds.most_common())
    lines.extend(
        [
            "",
            (f"Group 1 pool: non-exact pred and score >= {args.fp_conf} → {len(group1)} spans"),
            f"  breakdown: {group1_breakdown or '(empty)'}",
            (
                f"Group 2 pool: exact TP and {args.tp_conf_lo} <= score < "
                f"{args.tp_conf_hi} → {len(group2)} spans"
            ),
            f"Group 3 pool: gold with zero overlapping predicted span → {len(group3)} entities",
            f"Examples: seed={args.seed}, up to one per label then random fill, n={args.examples}.",
            "",
        ]
    )
    lines.extend(
        render_examples(
            (f"Group 1 — high-confidence non-exact predictions (score >= {args.fp_conf})"),
            group1,
            context=args.context,
            seed=args.seed,
            n=args.examples,
            group=1,
        )
    )
    lines.extend(
        render_examples(
            f"Group 2 — exact TP with {args.tp_conf_lo} <= score < {args.tp_conf_hi}",
            group2,
            context=args.context,
            seed=args.seed + 100,
            n=args.examples,
            group=2,
        )
    )
    lines.extend(
        render_examples(
            "Group 3 — gold entities with no overlapping prediction",
            group3,
            context=args.context,
            seed=args.seed + 200,
            n=args.examples,
            group=3,
        )
    )
    if meta:
        lines.extend(["", f"cache meta: {json.dumps(meta, ensure_ascii=False)}"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"wrote {args.output} ({len(lines)} lines)")


if __name__ == "__main__":
    main()

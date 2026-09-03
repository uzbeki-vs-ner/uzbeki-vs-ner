#!/usr/bin/env python3
"""Interactive GLiNER error review: look up a dev doc by hash and see gold vs preds.

Examples:
  uv run python scripts/review_gliner_case.py --hash 07107ca7c68fc5eca6bd7f77c69b098220260301
  uv run python scripts/review_gliner_case.py --kind partial_same_type --limit 8
  uv run python scripts/review_gliner_case.py --kind missed --limit 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from uzbek_ner.io.jsonl import read_jsonl_records
from uzbek_ner.metrics.error_analysis import classify_predicted_span
from uzbek_ner.metrics.exact_span import EntityKey
from uzbek_ner.settings import REPO_ROOT, get_settings

JsonObject = dict[str, Any]

DEFAULT_CACHE = (
    REPO_ROOT / "outputs" / "cache" / "gliner_multi_v21_official_dev" / "scored_spans.json"
)


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, default=settings.official_dev)
    parser.add_argument("--scored-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--hash", type=str, default=None, help="Show one document in full")
    parser.add_argument(
        "--kind",
        choices=(
            "partial_same_type",
            "partial_diff_type",
            "type_mismatch",
            "spurious",
            "missed",
            "low_conf_tp",
        ),
        default=None,
        help="List example spans of this error kind",
    )
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--min-score", type=float, default=0.7, help="For pred error kinds")
    parser.add_argument("--context", type=int, default=120)
    return parser.parse_args()


def load_scored(cache_path: Path) -> dict[str, list[JsonObject]]:
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    return {row["hash"]: row["spans"] for row in payload["docs"]}


def gold_map(records: list[JsonObject]) -> dict[str, JsonObject]:
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


def preds_at_tau(scored: dict[str, list[JsonObject]], tau: float) -> dict[str, set[EntityKey]]:
    return {
        record_hash: {
            (str(row["label"]), int(row["start"]), int(row["end"]))
            for row in spans
            if float(row["score"]) >= tau
        }
        for record_hash, spans in scored.items()
    }


def score_lookup(
    scored: dict[str, list[JsonObject]], tau: float
) -> dict[tuple[str, EntityKey], float]:
    out: dict[tuple[str, EntityKey], float] = {}
    for record_hash, spans in scored.items():
        for row in spans:
            if float(row["score"]) < tau:
                continue
            key = (str(row["label"]), int(row["start"]), int(row["end"]))
            out[(record_hash, key)] = float(row["score"])
    return out


def annotate_text(
    text: str,
    marks: list[tuple[int, int, str, str]],
    *,
    radius: int | None = None,
) -> str:
    """marks: (start, end, tag, label) sorted by start."""

    clip_lo = 0
    clip_hi = len(text)
    if radius is not None and marks:
        focus_start = min(start for start, _, _, _ in marks)
        focus_end = max(end for _, end, _, _ in marks)
        clip_lo = max(0, focus_start - radius)
        clip_hi = min(len(text), focus_end + radius)

    snippet = text[clip_lo:clip_hi]
    offset = clip_lo
    local_marks = [(s - offset, e - offset, tag, label) for s, e, tag, label in marks]

    parts: list[str] = []
    if clip_lo > 0:
        parts.append("…")
    cursor = 0
    for start, end, tag, label in sorted(local_marks, key=lambda item: item[0]):
        if start < cursor:
            continue
        parts.append(snippet[cursor:start])
        parts.append(f"⟦{tag}:{label}:{snippet[start:end]}⟧")
        cursor = end
    parts.append(snippet[cursor:])
    if clip_hi < len(text):
        parts.append("…")
    return "".join(parts)


def show_document(
    record_hash: str,
    gold: dict[str, JsonObject],
    scored: dict[str, list[JsonObject]],
    *,
    tau: float,
    context: int,
) -> None:
    record = gold[record_hash]
    text = record["text"]
    gold_entities = record["entities"]
    pred_entities = preds_at_tau(scored, tau)[record_hash]
    scores = score_lookup(scored, tau)
    print(f"hash: {record_hash}")
    print(f"text length: {len(text)} chars")
    print()
    print("--- gold ---")
    for label, start, end in sorted(gold_entities, key=lambda item: (item[1], item[2])):
        print(f"  {label} [{start}:{end}] {text[start:end]!r}")

    print()
    print(f"--- predictions (tau >= {tau:.1f}) ---")
    for key in sorted(pred_entities, key=lambda item: (item[1], item[2])):
        label, start, end = key
        kind, hits = classify_predicted_span(key, gold_entities)
        conf = scores.get((record_hash, key), float("nan"))
        flag = "TP" if kind == "exact" else kind
        hit_str = ""
        if hits:
            hit_str = " | gold " + " / ".join(
                f"{g[0]} [{g[1]}:{g[2]}] {text[g[1] : g[2]]!r}" for g in hits
            )
        print(f"  {label} [{start}:{end}] {text[start:end]!r}  score={conf:.4f}  {flag}{hit_str}")

    # problem spans only
    problems: list[tuple[int, int, str, str]] = []
    for key in pred_entities:
        kind, _ = classify_predicted_span(key, gold_entities)
        if kind != "exact":
            label, start, end = key
            problems.append((start, end, "PRED", f"{kind}/{label}"))
    for key in gold_entities:
        gs, ge = key[1], key[2]
        if not any(ge > ps and gs < pe for (_pl, ps, pe) in pred_entities):
            label = key[0]
            problems.append((gs, ge, "MISS", label))

    if problems:
        print()
        print("--- annotated (errors only) ---")
        print(annotate_text(text, problems, radius=context))


def iter_error_pool(
    gold: dict[str, JsonObject],
    scored: dict[str, list[JsonObject]],
    *,
    tau: float,
    kind: str,
    min_score: float,
) -> list[JsonObject]:
    scores = score_lookup(scored, tau)
    rows: list[JsonObject] = []

    if kind == "missed":
        for record_hash, record in gold.items():
            text = record["text"]
            gold_entities = record["entities"]
            pred_entities = preds_at_tau(scored, tau)[record_hash]
            for key in gold_entities:
                gs, ge = key[1], key[2]
                if any(ge > ps and gs < pe for (_pl, ps, pe) in pred_entities):
                    continue
                rows.append(
                    {
                        "hash": record_hash,
                        "label": key[0],
                        "start": gs,
                        "end": ge,
                        "span_text": text[gs:ge],
                        "text": text,
                        "conf": None,
                        "error_kind": "missed",
                    }
                )
        rows.sort(key=lambda row: (row["label"], row["hash"]))
        return rows

    if kind == "low_conf_tp":
        for record_hash, pred_entities in preds_at_tau(scored, tau).items():
            gold_entities = gold[record_hash]["entities"]
            text = gold[record_hash]["text"]
            for key in pred_entities:
                if key not in gold_entities:
                    continue
                conf = scores.get((record_hash, key), 0.0)
                if conf >= 0.7:
                    continue
                label, start, end = key
                rows.append(
                    {
                        "hash": record_hash,
                        "label": label,
                        "start": start,
                        "end": end,
                        "span_text": text[start:end],
                        "text": text,
                        "conf": conf,
                        "error_kind": "low_conf_tp",
                    }
                )
        rows.sort(key=lambda row: row["conf"] or 0.0)
        return rows

    for record_hash, pred_entities in preds_at_tau(scored, tau).items():
        gold_entities = gold[record_hash]["entities"]
        text = gold[record_hash]["text"]
        for key in pred_entities:
            error_kind, hits = classify_predicted_span(key, gold_entities)
            if error_kind != kind:
                continue
            conf = scores.get((record_hash, key), 0.0)
            if conf < min_score:
                continue
            label, start, end = key
            rows.append(
                {
                    "hash": record_hash,
                    "label": label,
                    "start": start,
                    "end": end,
                    "span_text": text[start:end],
                    "text": text,
                    "conf": conf,
                    "error_kind": error_kind,
                    "gold_hits": hits,
                }
            )
    rows.sort(key=lambda row: -(row["conf"] or 0.0))
    return rows


def context_line(text: str, start: int, end: int, *, radius: int) -> str:
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    before = text[left:start]
    span = text[start:end]
    after = text[end:right]
    return f"…{before}⟦{span}⟧{after}…"


def list_cases(rows: list[JsonObject], *, context: int) -> None:
    for index, row in enumerate(rows, 1):
        print(f"--- {index} ---")
        print(f"hash:  {row['hash']}")
        if row.get("conf") is not None:
            print(f"score: {row['conf']:.4f}")
        print(f"kind:  {row['error_kind']}")
        print(f"span:  {row['label']} [{row['start']}:{row['end']}] {row['span_text']!r}")
        if row.get("gold_hits"):
            text = row["text"]
            for hit in row["gold_hits"]:
                print(f"gold:  {hit[0]} [{hit[1]}:{hit[2]}] {text[hit[1] : hit[2]]!r}")
        print(f"ctx:   {context_line(row['text'], row['start'], row['end'], radius=context)}")
        print(f"view:  uv run python scripts/review_gliner_case.py --hash {row['hash']}")
        print()


def main() -> None:
    args = parse_args()
    records = read_jsonl_records(args.gold)
    gold = gold_map(records)
    scored = load_scored(args.scored_cache)

    if args.hash:
        if args.hash not in gold:
            raise SystemExit(f"hash not in dev: {args.hash}")
        show_document(
            args.hash,
            gold,
            scored,
            tau=args.tau,
            context=args.context,
        )
        return

    if args.kind is None:
        raise SystemExit("pass --hash or --kind")

    rows = iter_error_pool(
        gold,
        scored,
        tau=args.tau,
        kind=args.kind,
        min_score=args.min_score,
    )
    print(f"pool: {len(rows)} spans (showing {min(args.limit, len(rows))})")
    print()
    list_cases(rows[: args.limit], context=args.context)


if __name__ == "__main__":
    main()

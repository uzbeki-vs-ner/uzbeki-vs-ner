"""Sliding-word windows for GLiNER inference on long official JSONL docs."""

from __future__ import annotations

from typing import Any

from uzbek_ner.gliner_data import split_words
from uzbek_ner.labels import ENTITY_LABELS

JsonObject = dict[str, Any]
EntityKey = tuple[str, int, int]

# Eval-tuned defaults (+0.048 micro-F1 vs single-pass on official dev).
DEFAULT_MAX_WORDS = 384
DEFAULT_STRIDE = 128


def word_window_spans(text: str, *, max_words: int, stride: int) -> list[tuple[int, int]]:
    """Return char spans ``[start, end)`` aligned to GLiNER word boundaries."""

    if max_words < 1:
        raise ValueError("max_words must be positive")
    if stride < 1:
        raise ValueError("stride must be positive")
    tokens = split_words(text)
    if not tokens:
        return []
    if len(tokens) <= max_words:
        return [(0, len(text))]

    spans: list[tuple[int, int]] = []
    for start_idx in range(0, len(tokens), stride):
        end_idx = min(start_idx + max_words, len(tokens))
        spans.append((tokens[start_idx][1], tokens[end_idx - 1][2]))
        if end_idx >= len(tokens):
            break

    tail_start = max(0, len(tokens) - max_words)
    tail = (tokens[tail_start][1], tokens[-1][2])
    if spans[-1] != tail:
        spans.append(tail)
    return spans


def merge_window_entities(candidates: list[JsonObject]) -> list[JsonObject]:
    """Dedupe exact spans (max score), then greedy non-overlap by score."""

    best_score: dict[EntityKey, float] = {}
    for row in candidates:
        key = (str(row["label"]), int(row["start"]), int(row["end"]))
        score = float(row.get("score", 1.0))
        if key not in best_score or score > best_score[key]:
            best_score[key] = score

    ranked = sorted(
        (
            {"label": key[0], "start": key[1], "end": key[2], "score": score}
            for key, score in best_score.items()
        ),
        key=lambda row: (-float(row["score"]), int(row["start"]), int(row["end"]), row["label"]),
    )
    kept: list[JsonObject] = []
    occupied: list[tuple[int, int]] = []
    for row in ranked:
        start, end = int(row["start"]), int(row["end"])
        if any(end > other_start and start < other_end for other_start, other_end in occupied):
            continue
        kept.append(row)
        occupied.append((start, end))
    kept.sort(key=lambda row: (int(row["start"]), int(row["end"]), row["label"]))
    return kept


def entities_for_submission(rows: list[JsonObject]) -> list[JsonObject]:
    return [
        {"label": row["label"], "start": int(row["start"]), "end": int(row["end"])} for row in rows
    ]


def predict_records_windowed(
    model: Any,
    records: list[JsonObject],
    *,
    labels: list[str] | None = None,
    max_words: int = DEFAULT_MAX_WORDS,
    stride: int = DEFAULT_STRIDE,
    threshold: float = 0.5,
    batch_size: int = 8,
) -> dict[str, list[JsonObject]]:
    """Run GLiNER on overlapping word windows and merge to document entities."""

    label_list = list(labels or ENTITY_LABELS)
    work: list[tuple[str, int, str]] = []
    windows_per_doc: dict[str, int] = {}
    for record in records:
        text = str(record["text"])
        record_hash = str(record["hash"])
        spans = word_window_spans(text, max_words=max_words, stride=stride)
        windows_per_doc[record_hash] = len(spans)
        for start, end in spans:
            work.append((record_hash, start, text[start:end]))

    candidates: dict[str, list[JsonObject]] = {str(record["hash"]): [] for record in records}
    for batch_start in range(0, len(work), batch_size):
        batch = work[batch_start : batch_start + batch_size]
        texts = [chunk for _record_hash, _base, chunk in batch]
        batch_preds = model.batch_predict_entities(
            texts,
            label_list,
            flat_ner=True,
            threshold=threshold,
            batch_size=len(texts),
        )
        for (record_hash, base, _chunk), ents in zip(batch, batch_preds, strict=True):
            for entity in ents:
                candidates[record_hash].append(
                    {
                        "label": str(entity["label"]),
                        "start": base + int(entity["start"]),
                        "end": base + int(entity["end"]),
                        "score": float(entity["score"]),
                    }
                )

    merged = {record_hash: merge_window_entities(rows) for record_hash, rows in candidates.items()}
    meta = {"windows_per_doc": windows_per_doc}
    return merged, meta

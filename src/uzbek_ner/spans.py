"""Token BIO alignment and span decoding (ported from organizer baseline)."""

from __future__ import annotations

from typing import Any

from uzbek_ner.labels import ENTITY_LABELS, TAG_TO_ID

JsonObject = dict[str, Any]
Offsets = list[tuple[int, int]]


def align_labels(offsets: Offsets, entities: list[JsonObject]) -> list[int]:
    """Map char-level entity spans to per-token BIO label ids for one window."""

    labels: list[int] = []
    entity_index = 0
    for start, end in offsets:
        if start == end:
            labels.append(-100)
            continue
        while entity_index < len(entities) and entities[entity_index]["end"] <= start:
            entity_index += 1
        if entity_index >= len(entities):
            labels.append(TAG_TO_ID["O"])
            continue

        entity = entities[entity_index]
        if end <= entity["start"] or start >= entity["end"]:
            labels.append(TAG_TO_ID["O"])
            continue
        prefix = "B" if start <= entity["start"] < end else "I"
        labels.append(TAG_TO_ID[f"{prefix}-{entity['label']}"])
    return labels


def decode_bio_tokens(tokens: list[tuple[int, int, str]]) -> list[JsonObject]:
    """Rebuild char spans from ordered (start, end, BIO-tag) token predictions."""

    entities: list[JsonObject] = []
    current: JsonObject | None = None

    def flush() -> None:
        nonlocal current
        if current is not None:
            entities.append(current)
            current = None

    for start, end, tag in tokens:
        if tag == "O":
            flush()
            continue
        prefix, separator, label = tag.partition("-")
        if separator != "-" or prefix not in {"B", "I"} or label not in ENTITY_LABELS:
            msg = f"unsupported tag {tag!r}"
            raise ValueError(msg)

        if prefix == "B" or current is None or current["label"] != label:
            flush()
            current = {"label": label, "start": start, "end": end}
        else:
            current["end"] = max(current["end"], end)
    flush()
    return entities

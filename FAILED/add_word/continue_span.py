"""Fossil: one-word continuation. Not imported by ``uzbek_ner``.

See ``FAILED/add_word/README.md``. Held-out ΔF1 was +0.003 at best; do not
put this back on the default decode path.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from uzbek_ner.decode.snap import is_word_char, nms_longer_wins, snap_entities
from uzbek_ner.labels import ENTITY_LABELS, TAG_TO_ID

JsonObject = dict[str, Any]
Direction = Literal["right", "left", "both"]

__all__ = [
    "Direction",
    "continue_one_word",
    "entity_mass",
    "next_word",
    "prev_word",
]

_B_IDS = {label: TAG_TO_ID[f"B-{label}"] for label in ENTITY_LABELS}
_I_IDS = {label: TAG_TO_ID[f"I-{label}"] for label in ENTITY_LABELS}
_INLINE_SPACE = frozenset(" \t")


def next_word(text: str, end: int) -> tuple[int, int] | None:
    """First word after ``end``, skipping only spaces/tabs. ``None`` if punct/EOL."""

    index = end
    n = len(text)
    while index < n and text[index] in _INLINE_SPACE:
        index += 1
    if index >= n or not is_word_char(text[index]):
        return None
    stop = index
    while stop < n and is_word_char(text[stop]):
        stop += 1
    return index, stop


def prev_word(text: str, start: int) -> tuple[int, int] | None:
    """Word immediately before ``start``, skipping only spaces/tabs."""

    index = start
    while index > 0 and text[index - 1] in _INLINE_SPACE:
        index -= 1
    if index <= 0 or not is_word_char(text[index - 1]):
        return None
    begin = index
    while begin > 0 and is_word_char(text[begin - 1]):
        begin -= 1
    return begin, index


def _overlaps_other(start: int, end: int, occupied: list[tuple[int, int]]) -> bool:
    return any(start < other_end and end > other_start for other_start, other_end in occupied)


def entity_mass(
    label: str,
    word_start: int,
    word_end: int,
    offsets: np.ndarray,
    probs: np.ndarray,
) -> float:
    """Mean P(B-label)+P(I-label) on content tokens overlapping the word."""

    starts = offsets[:, 0]
    ends = offsets[:, 1]
    overlap = (starts != ends) & (ends > word_start) & (starts < word_end)
    if not np.any(overlap):
        return 0.0
    return float((probs[overlap, _B_IDS[label]] + probs[overlap, _I_IDS[label]]).mean())


def continue_one_word(
    text: str,
    entities: list[JsonObject],
    offsets: np.ndarray,
    probs: np.ndarray,
    *,
    tau: float,
    direction: Direction = "both",
    labels: tuple[str, ...] | None = None,
    max_words: int = 1,
) -> list[JsonObject]:
    """Extend each span by neighboring words if entity-mass ≥ ``tau``.

    Does not glue two already-predicted entities. Default ``max_words=1``
    takes the higher-mass side when both neighbors pass the gate.
    """

    allowed = set(labels) if labels is not None else set(ENTITY_LABELS)
    occupied = [(int(row["start"]), int(row["end"])) for row in entities]
    grown: list[tuple[str, int, int]] = []
    for row in entities:
        label = str(row["label"])
        start, end = int(row["start"]), int(row["end"])
        if label not in allowed or max_words <= 0:
            grown.append((label, start, end))
            continue
        others = [(left, right) for left, right in occupied if (left, right) != (start, end)]
        candidates: list[tuple[float, int, int, int]] = []
        if direction in {"right", "both"}:
            nxt = next_word(text, end)
            if nxt is not None and not _overlaps_other(*nxt, others):
                mass = entity_mass(label, nxt[0], nxt[1], offsets, probs)
                if mass >= tau:
                    candidates.append((mass, 1, nxt[0], nxt[1]))
        if direction in {"left", "both"}:
            prv = prev_word(text, start)
            if prv is not None and not _overlaps_other(*prv, others):
                mass = entity_mass(label, prv[0], prv[1], offsets, probs)
                if mass >= tau:
                    candidates.append((mass, 0, prv[0], prv[1]))
        # Higher mass first; ties prefer the right neighbor (surnames, last tokens).
        candidates.sort(key=lambda item: (-item[0], -item[1]))
        for _mass, side, word_start, word_end in candidates[:max_words]:
            if side == 1:
                end = word_end
            else:
                start = word_start
        grown.append((label, start, end))
    snapped = snap_entities(
        text,
        [{"label": label, "start": start, "end": end} for label, start, end in grown],
    )
    keyed = nms_longer_wins(
        [(str(row["label"]), int(row["start"]), int(row["end"])) for row in snapped]
    )
    return [{"label": label, "start": start, "end": end} for label, start, end in keyed]

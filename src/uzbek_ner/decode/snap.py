# ruff: noqa: RUF001
"""Gold-free span postprocess for exact-offset NER.

Official scoring has no partial credit, so decode may only move ``(start, end)``
using the text — never gold. Word-edge expansion repairs mid-subword cuts.
A global start/end offset learned from errors also shifts already-exact spans.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from uzbek_ner.labels import ENTITY_LABELS

JsonObject = dict[str, Any]
Entity = tuple[str, int, int]

# Same inventory as scripts/extract_silver.py (official suffix-inside policy).
_LATIN_SUFFIXES = (
    "lardagi",
    "laridan",
    "larining",
    "lariga",
    "larida",
    "lardan",
    "larni",
    "larga",
    "larda",
    "ning",
    "dagi",
    "dan",
    "ga",
    "ni",
    "da",
    "lar",
)
_CYR_SUFFIXES = (
    "лардаги",
    "ларидан",
    "ларининг",
    "ларига",
    "ларида",
    "лардан",
    "ларни",
    "ларга",
    "ларда",
    "нинг",
    "даги",
    "дан",
    "га",
    "ни",
    "да",
    "лар",
)
SUFFIXES: tuple[str, ...] = tuple(
    sorted(set(_LATIN_SUFFIXES + _CYR_SUFFIXES), key=len, reverse=True)
)
APOSTROPHES = ("'", "ʼ", "ʻ", "‘", "’", "`", "´")
_APOSTROPHE_PREFIXES: tuple[str, ...] = ("", *APOSTROPHES)
_QUOTE_OPENERS = frozenset('«»"“”()[]{}')
_LABEL_PRIORITY = {label: index for index, label in enumerate(ENTITY_LABELS)}


def is_word_char(char: str) -> bool:
    return char.isalnum() or char in APOSTROPHES or char in "-‑"


def is_mid_token_start(text: str, start: int) -> bool:
    return 0 < start < len(text) and is_word_char(text[start - 1]) and is_word_char(text[start])


def is_mid_token_end(text: str, end: int) -> bool:
    return 0 < end < len(text) and is_word_char(text[end - 1]) and is_word_char(text[end])


def expand_to_word_edges(text: str, start: int, end: int) -> tuple[int, int]:
    """Grow ``[start, end)`` through attached letters/digits/apostrophes."""

    start = max(0, min(start, len(text)))
    end = max(start, min(end, len(text)))
    while start > 0 and is_word_char(text[start - 1]):
        start -= 1
    while end < len(text) and is_word_char(text[end]):
        end += 1
    return start, end


def match_attached_suffix(rest: str) -> str | None:
    """Longest official case ending that prefixes ``rest`` (original casing)."""

    if not rest:
        return None
    rest_cf = rest.casefold()
    best: str | None = None
    for suffix in SUFFIXES:
        for prefix in _APOSTROPHE_PREFIXES:
            candidate = prefix + suffix
            length = len(candidate)
            if length > len(rest) or not rest_cf.startswith(candidate.casefold()):
                continue
            after = rest[length:]
            if after and after[0].isalpha():
                continue
            if best is None or length > len(best):
                best = rest[:length]
    return best


def attach_suffix(text: str, start: int, end: int) -> tuple[int, int]:
    """Extend ``end`` when a case ending is glued on (official span policy)."""

    if not 0 <= start < end <= len(text):
        return start, end
    rest = text[end:]
    if not rest or rest[0].isspace() or rest[0] in _QUOTE_OPENERS:
        return start, end
    last = text[end - 1]
    if not (last.isalnum() or last in APOSTROPHES or last in "-‑"):
        return start, end
    matched = match_attached_suffix(rest)
    if not matched:
        return start, end
    return start, end + len(matched)


def shift_span(
    start: int,
    end: int,
    *,
    d_start: int,
    d_end: int,
    text_length: int,
) -> tuple[int, int]:
    """Undo ``pred = gold + (d_start, d_end)``. Refuse empty/inverted spans."""

    new_start = start - d_start
    new_end = end - d_end
    new_start = max(0, min(new_start, text_length))
    new_end = max(0, min(new_end, text_length))
    if new_end <= new_start:
        return start, end
    return new_start, new_end


def nms_longer_wins(entities: list[Entity]) -> list[Entity]:
    """Official gold is flat: drop overlapping preds, keep the longer span."""

    ordered = sorted(
        entities,
        key=lambda item: (
            -(item[2] - item[1]),
            item[1],
            _LABEL_PRIORITY.get(item[0], 9),
            item[0],
        ),
    )
    accepted: list[Entity] = []
    for label, start, end in ordered:
        if not start < end or label not in _LABEL_PRIORITY:
            continue
        if any(start < other_end and other_start < end for _, other_start, other_end in accepted):
            continue
        accepted.append((label, start, end))
    accepted.sort(key=lambda item: (item[1], item[2], item[0]))
    return accepted


def snap_entity(
    text: str,
    label: str,
    start: int,
    end: int,
    *,
    word: bool = True,
    suffix: bool = True,
) -> Entity:
    if word:
        start, end = expand_to_word_edges(text, start, end)
    if suffix:
        start, end = attach_suffix(text, start, end)
    return (label, start, end)


def snap_entities(
    text: str,
    entities: list[JsonObject],
    *,
    word: bool = True,
    suffix: bool = True,
    nms: bool = True,
) -> list[JsonObject]:
    snapped = [
        snap_entity(
            text, str(row["label"]), int(row["start"]), int(row["end"]), word=word, suffix=suffix
        )
        for row in entities
    ]
    if nms:
        snapped = nms_longer_wins(snapped)
    return [{"label": label, "start": start, "end": end} for label, start, end in snapped]


def mode_int(values: list[int]) -> int:
    """Most frequent integer; ties break toward 0, then the smaller signed value."""

    if not values:
        return 0
    counts = Counter(values)
    best = max(counts.values())
    tied = [value for value, count in counts.items() if count == best]
    return sorted(tied, key=lambda value: (abs(value), value))[0]

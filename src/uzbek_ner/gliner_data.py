"""Convert official JSONL records to GLiNER word-span examples."""

from __future__ import annotations

import re
from typing import Any

from uzbek_ner.labels import ENTITY_LABELS

WORD_RE = re.compile(r"\w+(?:[-_]\w+)*|\S")
JsonObject = dict[str, Any]


def split_words(text: str) -> list[tuple[str, int, int]]:
    """Match GLiNER ``WhitespaceTokenSplitter``: words plus leftover non-space."""

    return [(match.group(), match.start(), match.end()) for match in WORD_RE.finditer(text)]


def convert_record(record: JsonObject, *, max_words: int) -> JsonObject | None:
    """Map char gold spans onto whitespace tokens. Truncate to ``max_words``."""

    text = str(record["text"])
    tokens = split_words(text)[:max_words]
    if not tokens:
        return None
    spans: list[list[object]] = []
    for entity in record["entities"]:
        gold_start, gold_end = int(entity["start"]), int(entity["end"])
        covering = [
            index
            for index, (_token, start, end) in enumerate(tokens)
            if end > gold_start and start < gold_end
        ]
        if not covering:
            continue
        label = str(entity["label"])
        if label not in ENTITY_LABELS:
            continue
        spans.append([covering[0], covering[-1], label])
    return {
        "tokenized_text": [token for token, _start, _end in tokens],
        "ner": spans,
        "ner_labels": list(ENTITY_LABELS),
    }

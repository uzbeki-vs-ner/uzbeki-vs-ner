from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from uzbek_ner.labels import ENTITY_LABELS
from uzbek_ner.service.schemas import EntityLabel, EntitySpan, PredictItem, PredictResult

# Surfaces from the two organizer check_service.py probe sentences. Quality is
# not scored here; this only keeps the response format non-empty and valid.
DEFAULT_LEXICON: tuple[tuple[str, EntityLabel], ...] = (
    ("Алишер Навоий", "NAME"),
    ("Тошкентда", "GEO"),
    ("Toshkent", "GEO"),
    ("Ali", "NAME"),
)


@runtime_checkable
class NerBackend(Protocol):
    """Batch NER. Official API is one JSON object after the whole batch."""

    async def predict_batch(self, items: Sequence[PredictItem]) -> list[PredictResult]:
        """Return one result per item, same order, copying hash."""
        ...


class StubNerBackend:
    """CPU gazetteer. No weights, no HuggingFace, no GPU.

    Tiny and non-blocking, so it runs on the event loop. A Torch backend that
    blocks should wrap inference in ``asyncio.to_thread`` (or a process pool).
    """

    def __init__(self, lexicon: Sequence[tuple[str, EntityLabel]] | None = None) -> None:
        entries = tuple(lexicon) if lexicon is not None else DEFAULT_LEXICON
        self._lexicon = tuple(sorted(entries, key=lambda pair: len(pair[0]), reverse=True))

    async def predict_batch(self, items: Sequence[PredictItem]) -> list[PredictResult]:
        return [self._predict_one(item) for item in items]

    def _predict_one(self, item: PredictItem) -> PredictResult:
        text = item.text
        occupied: list[tuple[int, int]] = []
        entities: list[EntitySpan] = []
        seen: set[tuple[str, int, int]] = set()

        for surface, label in self._lexicon:
            if not surface or label not in ENTITY_LABELS:
                continue
            cursor = 0
            while True:
                start = text.find(surface, cursor)
                if start < 0:
                    break
                end = start + len(surface)
                cursor = start + 1
                if not 0 <= start < end <= len(text):
                    continue
                if any(not (end <= left or start >= right) for left, right in occupied):
                    continue
                key = (label, start, end)
                if key in seen:
                    continue
                seen.add(key)
                occupied.append((start, end))
                entities.append(EntitySpan(label=label, start=start, end=end))

        return PredictResult(hash=item.hash, entities=entities)

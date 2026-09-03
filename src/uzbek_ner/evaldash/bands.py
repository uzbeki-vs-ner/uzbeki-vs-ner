"""How to read official exact-span micro-F1 on this hackathon.

Organizers require the same document hash and exact ``(label, start, end)``.
There is no partial credit for overlapping spans or a correct type with a
wrong suffix. English CoNLL ~0.90 is the wrong reference frame.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class F1Band:
    """One half-open interval ``[min_inclusive, max_exclusive)``."""

    id: str
    min_inclusive: float
    max_exclusive: float
    title: str
    meaning: str
    color: str


# Measured on this repo, not a leaderboard claim.
ANCHORS: dict[str, str] = {
    "frozen_linear_probe": "0.05-0.13 (EMA, 64+64 docs, encoder frozen)",
    "uztext_2ep_smoke": "0.578 (official dev, 2 epochs, 2026-09-02)",
    "organizer_floor": "no hard F1 cutoff — API/Docker must pass",
}

F1_BANDS: tuple[F1Band, ...] = (
    F1Band(
        id="broken",
        min_inclusive=0.0,
        max_exclusive=0.20,
        title="Сломано / ещё не NER",
        meaning=(
            "Случайная голова, gazetteer-stub, frozen probe. Энкодер не выделяет "
            "сущности. Пайплайн, скорее всего, не учится или декодирует мусор."
        ),
        color="#7F1D1D",
    ),
    F1Band(
        id="weak",
        min_inclusive=0.20,
        max_exclusive=0.45,
        title="Слабый смоук",
        meaning=(
            "Градиенты живые, но система ещё не читает тексты. Много FP/FN, "
            "суффиксы и границы почти случайны. Не сравнивать модели на этом уровне."
        ),
        color="#C2410C",
    ),
    F1Band(
        id="baseline",
        min_inclusive=0.45,
        max_exclusive=0.60,
        title="Рабочий бейзлайн",
        meaning=(
            "Fine-tune в принципе тянем: 2 эпохи uztext сели сюда (~0.58). "
            "F1≈0.5 — не «плохо как ноль», но и не продукт: обычно дырявый precision "
            "и промахи по агглютинации. Дальше учить/менять модель, не праздновать."
        ),
        color="#CA8A04",
    ),
    F1Band(
        id="competitive",
        min_inclusive=0.60,
        max_exclusive=0.75,
        title="Конкурентно",
        meaning=(
            "F1≈0.7 на этой метрике — уже хорошо. Exact-span без частичных совпадений "
            "съедает пункты на каждом суффиксе. Имеет смысл сравнивать архитектуры "
            "и смотреть ORG vs NAME vs GEO, а не гнаться за 0.95."
        ),
        color="#65A30D",
    ),
    F1Band(
        id="strong",
        min_inclusive=0.75,
        max_exclusive=0.85,
        title="Сильный хакатон",
        meaning=(
            "Похоже на систему, которую не стыдно сдавать. Остаточный штраф — "
            "ORG-клубы, смешанный скрипт, границы NAME. Precision и recall должны "
            "быть оба живыми, не один ценой другого."
        ),
        color="#15803D",
    ),
    F1Band(
        id="excellent",
        min_inclusive=0.85,
        max_exclusive=1.01,
        title="Отлично / проверь скорер",
        meaning=(
            "Редко на strict exact-span + узбекские падежи. Если внезапно 0.90+ "
            "после двух эпох — сначала убедись, что gold/pred не протекли и "
            "метрика не токен-accuracy."
        ),
        color="#0F766E",
    ),
)

# Grafana absolute thresholds: first step is the colour below the first cut.
GRAFANA_F1_THRESHOLD_STEPS: tuple[tuple[str, float | None], ...] = (
    ("#7F1D1D", None),
    ("#C2410C", 0.20),
    ("#CA8A04", 0.45),
    ("#65A30D", 0.60),
    ("#15803D", 0.75),
    ("#0F766E", 0.85),
)


def band_for_f1(f1: float) -> F1Band:
    """Map a micro-F1 (or per-label F1) onto the hackathon reading-guide band."""

    if f1 < 0:
        return F1_BANDS[0]
    for band in F1_BANDS:
        if band.min_inclusive <= f1 < band.max_exclusive:
            return band
    return F1_BANDS[-1]


def scale_payload() -> dict[str, object]:
    """JSON for ``GET /api/v1/scale`` and HTML/Grafana copy."""

    return {
        "metric": "official_dev exact-span micro-F1",
        "matching": "same hash and exact label/start/end; no partial credit",
        "note": (
            "Шкала для этого хакатона, не для CoNLL-English. "
            "Жёсткого порога у организаторов нет; API/Docker обязательны. "
            "0.5 ≈ рабочий бейзлайн, 0.7 ≈ уже хорошо, 0.85+ редко и подозрительно."
        ),
        "anchors": ANCHORS,
        "bands": [
            {
                "id": band.id,
                "min_inclusive": band.min_inclusive,
                "max_exclusive": band.max_exclusive,
                "title": band.title,
                "meaning": band.meaning,
                "color": band.color,
            }
            for band in F1_BANDS
        ],
    }

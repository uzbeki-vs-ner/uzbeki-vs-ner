"""Diagnostic span breakdown — not the organizer leaderboard metric.

Official score stays exact ``(label, start, end)``. These buckets answer *why*
micro-F1 is low: wrong type on a perfect span vs jittered boundaries vs misses.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from uzbek_ner.labels import ENTITY_LABELS
from uzbek_ner.metrics.exact_span import (
    EntityKey,
    JsonObject,
    load_gold_and_predictions,
)

Span = tuple[int, int]


def _prf(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "gold": tp + fn,
        "predicted": tp + fp,
    }


def _iou(start_a: int, end_a: int, start_b: int, end_b: int) -> float:
    inter = min(end_a, end_b) - max(start_a, start_b)
    if inter <= 0:
        return 0.0
    union = (end_a - start_a) + (end_b - start_b) - inter
    return inter / union if union else 0.0


def _empty_confusion() -> dict[str, dict[str, int]]:
    return {gold: dict.fromkeys(ENTITY_LABELS, 0) for gold in ENTITY_LABELS}


def _reading(
    *,
    exact_match: int,
    type_mismatch: int,
    partial_same_type: int,
    partial_diff_type: int,
    missed: int,
    spurious: int,
    type_accuracy: float,
    boundary_f1: float,
) -> tuple[str, str]:
    """Return ``(reading_id, reading_ru)`` for the HTML/Grafana copy."""

    detected = exact_match + type_mismatch + partial_same_type + partial_diff_type
    gold_total = exact_match + type_mismatch + partial_same_type + partial_diff_type + missed
    if gold_total == 0 and spurious == 0:
        return "empty", "Нет сущностей в gold и pred — смотреть нечего."

    type_share = type_mismatch / detected if detected else 0.0
    partial_share = (partial_same_type + partial_diff_type) / gold_total if gold_total else 0.0
    miss_share = missed / gold_total if gold_total else 0.0

    if type_mismatch >= 1 and type_share >= 0.25 and type_mismatch >= partial_same_type:
        return (
            "type_confusion",
            "Границы часто верные, тип путается: strict F1 ниже, чем F1 по (start,end). "
            "Смотри матрицу ORG/NAME/GEO.",
        )
    if partial_same_type >= 1 and partial_share >= 0.15 and partial_same_type >= type_mismatch:
        return (
            "boundary_jitter",
            "Тип часто угадан, границы съезжают (суффиксы, обрезка токена). "
            "Type-agnostic F1 тоже низкий — это не путаница классов.",
        )
    if miss_share >= 0.35 or (spurious > exact_match and exact_match < gold_total * 0.5):
        return (
            "detection_gap",
            "Много промахов или лишних спанов без пересечения с gold. "
            "Модель слабо детектит упоминания, не только типы/границы.",
        )
    if type_accuracy >= 0.9 and boundary_f1 >= 0.7:
        return (
            "healthy",
            "И границы, и типы в основном на месте. Дальше добивать хвост ошибок по классам.",
        )
    return (
        "mixed",
        "Смешанный профиль: и типы, и границы, и пропуски. Смотри корзины по отдельности.",
    )


def analyze_span_errors(
    gold: dict[str, JsonObject],
    predictions: dict[str, set[EntityKey]],
) -> dict[str, Any]:
    """Break gold/pred entities into type vs boundary vs miss/spurious buckets."""

    exact_match = 0
    type_mismatch = 0
    partial_same_type = 0
    partial_diff_type = 0
    missed = 0
    spurious = 0
    boundary_tp = 0
    boundary_fp = 0
    boundary_fn = 0
    aligned_spans = 0
    type_correct = 0
    confusion = _empty_confusion()

    for record_hash, gold_record in gold.items():
        gold_entities: set[EntityKey] = gold_record["entities"]
        pred_entities = predictions[record_hash]

        gold_spans = {(start, end) for _, start, end in gold_entities}
        pred_spans = {(start, end) for _, start, end in pred_entities}
        aligned = gold_spans & pred_spans
        aligned_spans += len(aligned)
        boundary_tp += len(aligned)
        boundary_fp += len(pred_spans - gold_spans)
        boundary_fn += len(gold_spans - pred_spans)

        for span in aligned:
            gold_labels = {label for label, start, end in gold_entities if (start, end) == span}
            pred_labels = {label for label, start, end in pred_entities if (start, end) == span}
            gold_label = next(iter(gold_labels)) if len(gold_labels) == 1 else None
            pred_label = next(iter(pred_labels)) if len(pred_labels) == 1 else None
            if gold_label is not None and pred_label is not None:
                confusion[gold_label][pred_label] += 1
                if gold_label == pred_label:
                    type_correct += 1

        matched = gold_entities & pred_entities
        exact_match += len(matched)
        gold_rest = list(gold_entities - matched)
        pred_rest = list(pred_entities - matched)

        gold_rest_by_span: dict[Span, list[int]] = {}
        for index, (_label, start, end) in enumerate(gold_rest):
            gold_rest_by_span.setdefault((start, end), []).append(index)
        pred_rest_by_span: dict[Span, list[int]] = {}
        for index, (_label, start, end) in enumerate(pred_rest):
            pred_rest_by_span.setdefault((start, end), []).append(index)

        used_gold: set[int] = set()
        used_pred: set[int] = set()
        for span in set(gold_rest_by_span) & set(pred_rest_by_span):
            g_idx = gold_rest_by_span[span][0]
            p_idx = pred_rest_by_span[span][0]
            used_gold.add(g_idx)
            used_pred.add(p_idx)
            type_mismatch += 1

        leftover_gold = [item for index, item in enumerate(gold_rest) if index not in used_gold]
        leftover_pred = [item for index, item in enumerate(pred_rest) if index not in used_pred]

        pairs: list[tuple[float, int, int]] = []
        for g_i, (_g_label, g_start, g_end) in enumerate(leftover_gold):
            for p_i, (_p_label, p_start, p_end) in enumerate(leftover_pred):
                score = _iou(g_start, g_end, p_start, p_end)
                if score > 0:
                    pairs.append((score, g_i, p_i))
        pairs.sort(reverse=True)
        matched_g: set[int] = set()
        matched_p: set[int] = set()
        for _score, g_i, p_i in pairs:
            if g_i in matched_g or p_i in matched_p:
                continue
            matched_g.add(g_i)
            matched_p.add(p_i)
            if leftover_gold[g_i][0] == leftover_pred[p_i][0]:
                partial_same_type += 1
            else:
                partial_diff_type += 1

        missed += len(leftover_gold) - len(matched_g)
        spurious += len(leftover_pred) - len(matched_p)

    boundary = _prf(boundary_tp, boundary_fp, boundary_fn)
    type_accuracy = type_correct / aligned_spans if aligned_spans else 0.0
    reading_id, reading = _reading(
        exact_match=exact_match,
        type_mismatch=type_mismatch,
        partial_same_type=partial_same_type,
        partial_diff_type=partial_diff_type,
        missed=missed,
        spurious=spurious,
        type_accuracy=type_accuracy,
        boundary_f1=float(boundary["f1"]),
    )
    return {
        "schema_version": 1,
        "matching": "diagnostic only; organizer score remains exact label/start/end",
        "boundary_exact": boundary,
        "type_given_boundary": {
            "aligned_spans": aligned_spans,
            "type_correct": type_correct,
            "accuracy": type_accuracy,
            "confusion": confusion,
        },
        "buckets": {
            "exact_match": exact_match,
            "type_mismatch": type_mismatch,
            "partial_same_type": partial_same_type,
            "partial_diff_type": partial_diff_type,
            "missed": missed,
            "spurious": spurious,
        },
        "reading_id": reading_id,
        "reading": reading,
    }


def analyze_prediction_files(gold_path: Path, predictions_path: Path) -> dict[str, Any]:
    gold, predictions = load_gold_and_predictions(gold_path, predictions_path)
    return analyze_span_errors(gold, predictions)

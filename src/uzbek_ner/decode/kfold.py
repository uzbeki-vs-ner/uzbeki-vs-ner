"""K-fold decode tuning on official dev (analysis fold never scored)."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from typing import Any

from uzbek_ner.decode.snap import (
    is_mid_token_end,
    is_mid_token_start,
    mode_int,
    nms_longer_wins,
    shift_span,
    snap_entities,
)
from uzbek_ner.metrics.exact_span import EntityKey, JsonObject, calculate_exact_span_metrics


def _iou(start_a: int, end_a: int, start_b: int, end_b: int) -> float:
    inter = min(end_a, end_b) - max(start_a, start_b)
    if inter <= 0:
        return 0.0
    union = (end_a - start_a) + (end_b - start_b) - inter
    return inter / union if union else 0.0


PolicyFn = Callable[[str, list[JsonObject]], list[JsonObject]]


def make_folds(hashes: Sequence[str], k: int, seed: int) -> list[list[str]]:
    if k < 2:
        raise ValueError("k must be >= 2")
    items = list(hashes)
    rng = random.Random(seed)
    rng.shuffle(items)
    folds: list[list[str]] = [[] for _ in range(k)]
    for index, record_hash in enumerate(items):
        folds[index % k].append(record_hash)
    return folds


def _overlap_pairs(
    gold_entities: set[EntityKey],
    pred_entities: set[EntityKey],
) -> list[tuple[EntityKey, EntityKey]]:
    """Greedy IoU>0 matches after removing exact (label, start, end) hits."""

    matched = gold_entities & pred_entities
    leftover_gold = list(gold_entities - matched)
    leftover_pred = list(pred_entities - matched)
    scored: list[tuple[float, int, int]] = []
    for gold_index, gold in enumerate(leftover_gold):
        for pred_index, pred in enumerate(leftover_pred):
            score = _iou(gold[1], gold[2], pred[1], pred[2])
            if score > 0:
                scored.append((score, gold_index, pred_index))
    scored.sort(reverse=True)
    used_gold: set[int] = set()
    used_pred: set[int] = set()
    pairs: list[tuple[EntityKey, EntityKey]] = []
    for _score, gold_index, pred_index in scored:
        if gold_index in used_gold or pred_index in used_pred:
            continue
        used_gold.add(gold_index)
        used_pred.add(pred_index)
        pairs.append((leftover_gold[gold_index], leftover_pred[pred_index]))
    return pairs


def fit_offset_mode(
    gold: dict[str, JsonObject],
    predictions: dict[str, set[EntityKey]],
    hashes: Sequence[str],
    *,
    same_type_only: bool = True,
) -> dict[str, Any]:
    """Mode of (pred − gold) on overlapping non-exact pairs in the analysis fold."""

    d_start: list[int] = []
    d_end: list[int] = []
    by_label_start: dict[str, list[int]] = defaultdict(list)
    by_label_end: dict[str, list[int]] = defaultdict(list)
    n_pairs = 0
    for record_hash in hashes:
        gold_entities = gold[record_hash]["entities"]
        pred_entities = predictions[record_hash]
        for gold_ent, pred_ent in _overlap_pairs(gold_entities, pred_entities):
            if same_type_only and gold_ent[0] != pred_ent[0]:
                continue
            n_pairs += 1
            start_delta = pred_ent[1] - gold_ent[1]
            end_delta = pred_ent[2] - gold_ent[2]
            d_start.append(start_delta)
            d_end.append(end_delta)
            by_label_start[gold_ent[0]].append(start_delta)
            by_label_end[gold_ent[0]].append(end_delta)
    joint = Counter(zip(d_start, d_end, strict=True)).most_common(1)
    joint_d_start, joint_d_end = joint[0][0] if joint else (0, 0)
    per_label = {
        label: {
            "d_start": mode_int(by_label_start[label]),
            "d_end": mode_int(by_label_end[label]),
            "n": len(by_label_start[label]),
        }
        for label in by_label_start
    }
    return {
        "d_start": mode_int(d_start),
        "d_end": mode_int(d_end),
        "joint_d_start": int(joint_d_start),
        "joint_d_end": int(joint_d_end),
        "n_pairs": n_pairs,
        "per_label": per_label,
    }


def identity_policy(_text: str, entities: list[JsonObject]) -> list[JsonObject]:
    return list(entities)


def make_global_mode_policy(d_start: int, d_end: int) -> PolicyFn:
    def policy(text: str, entities: list[JsonObject]) -> list[JsonObject]:
        shifted: list[tuple[str, int, int]] = []
        n = len(text)
        for row in entities:
            start, end = shift_span(
                int(row["start"]),
                int(row["end"]),
                d_start=d_start,
                d_end=d_end,
                text_length=n,
            )
            shifted.append((str(row["label"]), start, end))
        return [
            {"label": label, "start": start, "end": end}
            for label, start, end in nms_longer_wins(shifted)
        ]

    return policy


def make_midtoken_mode_policy(d_start: int, d_end: int) -> PolicyFn:
    """Apply the learned offset only when that edge sits inside a word."""

    def policy(text: str, entities: list[JsonObject]) -> list[JsonObject]:
        shifted: list[tuple[str, int, int]] = []
        n = len(text)
        for row in entities:
            start, end = int(row["start"]), int(row["end"])
            use_start = d_start if is_mid_token_start(text, start) else 0
            use_end = d_end if is_mid_token_end(text, end) else 0
            start, end = shift_span(start, end, d_start=use_start, d_end=use_end, text_length=n)
            shifted.append((str(row["label"]), start, end))
        return [
            {"label": label, "start": start, "end": end}
            for label, start, end in nms_longer_wins(shifted)
        ]

    return policy


def make_per_label_mode_policy(fitted: dict[str, Any]) -> PolicyFn:
    per_label: dict[str, dict[str, int]] = fitted["per_label"]
    fallback_start = int(fitted["d_start"])
    fallback_end = int(fitted["d_end"])

    def policy(text: str, entities: list[JsonObject]) -> list[JsonObject]:
        shifted: list[tuple[str, int, int]] = []
        n = len(text)
        for row in entities:
            label = str(row["label"])
            stats = per_label.get(label, {})
            start, end = shift_span(
                int(row["start"]),
                int(row["end"]),
                d_start=int(stats.get("d_start", fallback_start)),
                d_end=int(stats.get("d_end", fallback_end)),
                text_length=n,
            )
            shifted.append((label, start, end))
        return [
            {"label": name, "start": start, "end": end}
            for name, start, end in nms_longer_wins(shifted)
        ]

    return policy


def word_snap_policy(text: str, entities: list[JsonObject]) -> list[JsonObject]:
    return snap_entities(text, entities, word=True, suffix=False, nms=True)


def word_suffix_snap_policy(text: str, entities: list[JsonObject]) -> list[JsonObject]:
    return snap_entities(text, entities, word=True, suffix=True, nms=True)


def entities_to_keys(entities: list[JsonObject]) -> set[EntityKey]:
    return {(str(row["label"]), int(row["start"]), int(row["end"])) for row in entities}


def apply_policy_to_split(
    gold: dict[str, JsonObject],
    raw_predictions: dict[str, list[JsonObject]],
    hashes: Sequence[str],
    policy: PolicyFn,
) -> dict[str, set[EntityKey]]:
    return {
        record_hash: entities_to_keys(
            policy(gold[record_hash]["text"], raw_predictions[record_hash])
        )
        for record_hash in hashes
    }


def score_split(
    gold: dict[str, JsonObject],
    predictions: dict[str, set[EntityKey]],
    hashes: Sequence[str],
) -> dict[str, Any]:
    gold_slice = {record_hash: gold[record_hash] for record_hash in hashes}
    pred_slice = {record_hash: predictions[record_hash] for record_hash in hashes}
    return calculate_exact_span_metrics(gold_slice, pred_slice)


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    var = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, var**0.5

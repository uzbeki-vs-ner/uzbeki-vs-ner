"""Exact-span metrics (organizer-compatible)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from uzbek_ner.io.jsonl import read_jsonl_records
from uzbek_ner.labels import ENTITY_LABELS

JsonObject = dict[str, Any]
EntityKey = tuple[str, int, int]


def _validate_prediction_entities(raw: Any, text_length: int, source: str) -> set[EntityKey]:
    if not isinstance(raw, list):
        raise ValueError(f"{source}: entities must be an array")
    entities: set[EntityKey] = set()
    for index, entity in enumerate(raw):
        if not isinstance(entity, dict):
            raise ValueError(f"{source}/entities[{index}]: entity must be an object")
        label = entity.get("label")
        start = entity.get("start")
        end = entity.get("end")
        if label not in ENTITY_LABELS:
            raise ValueError(f"{source}/entities[{index}]: invalid label {label!r}")
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(end, int)
            or isinstance(end, bool)
            or not 0 <= start < end <= text_length
        ):
            raise ValueError(f"{source}/entities[{index}]: invalid offsets")
        key = (label, start, end)
        if key in entities:
            raise ValueError(f"{source}/entities[{index}]: duplicate entity")
        entities.add(key)
    return entities


def _gold_by_hash(records: list[JsonObject], path: Path) -> dict[str, JsonObject]:
    result: dict[str, JsonObject] = {}
    for index, record in enumerate(records, start=1):
        text = record.get("text")
        if not isinstance(text, str):
            raise ValueError(f"{path}:{index}: gold text must be a string")
        entities = _validate_prediction_entities(
            record.get("entities"),
            len(text),
            f"{path}:{index}",
        )
        result[record["hash"]] = {"text": text, "entities": entities}
    return result


def _predictions_by_hash(
    records: list[JsonObject],
    path: Path,
    gold: dict[str, JsonObject],
) -> dict[str, set[EntityKey]]:
    predicted_hashes = {record["hash"] for record in records}
    gold_hashes = set(gold)
    missing = sorted(gold_hashes - predicted_hashes)
    extra = sorted(predicted_hashes - gold_hashes)
    if missing or extra:
        raise ValueError(
            "gold/prediction hashes differ: "
            f"missing={missing[:5]} ({len(missing)} total), "
            f"extra={extra[:5]} ({len(extra)} total)"
        )

    result: dict[str, set[EntityKey]] = {}
    for index, record in enumerate(records, start=1):
        record_hash = record["hash"]
        gold_record = gold[record_hash]
        if "text" in record and record["text"] != gold_record["text"]:
            raise ValueError(f"{path}:{index}: prediction text differs from gold for {record_hash}")
        result[record_hash] = _validate_prediction_entities(
            record.get("entities"),
            len(gold_record["text"]),
            f"{path}:{index}",
        )
    return result


def _metric_values(tp: int, fp: int, fn: int) -> JsonObject:
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


def calculate_exact_span_metrics(
    gold: dict[str, JsonObject],
    predictions: dict[str, set[EntityKey]],
) -> JsonObject:
    counts = {label: {"tp": 0, "fp": 0, "fn": 0} for label in ENTITY_LABELS}
    for record_hash, gold_record in gold.items():
        gold_entities = gold_record["entities"]
        predicted_entities = predictions[record_hash]
        for label in ENTITY_LABELS:
            gold_label = {entity for entity in gold_entities if entity[0] == label}
            predicted_label = {entity for entity in predicted_entities if entity[0] == label}
            counts[label]["tp"] += len(gold_label & predicted_label)
            counts[label]["fp"] += len(predicted_label - gold_label)
            counts[label]["fn"] += len(gold_label - predicted_label)

    by_label = {
        label: _metric_values(values["tp"], values["fp"], values["fn"])
        for label, values in counts.items()
    }
    micro = _metric_values(
        sum(values["tp"] for values in counts.values()),
        sum(values["fp"] for values in counts.values()),
        sum(values["fn"] for values in counts.values()),
    )
    macro = {
        metric: sum(by_label[label][metric] for label in ENTITY_LABELS) / len(ENTITY_LABELS)
        for metric in ("precision", "recall", "f1")
    }
    return {
        "schema_version": 1,
        "matching": "same hash and exact label/start/end",
        "records": len(gold),
        "by_label": by_label,
        "micro": micro,
        "macro": macro,
    }


def evaluate_prediction_files(gold_path: Path, predictions_path: Path) -> JsonObject:
    gold_records = read_jsonl_records(gold_path, require_entities=True)
    gold = _gold_by_hash(gold_records, gold_path)
    predictions = _predictions_by_hash(
        read_jsonl_records(predictions_path, require_entities=False),
        predictions_path,
        gold,
    )
    return calculate_exact_span_metrics(gold, predictions)


def write_metrics(path: Path, metrics: JsonObject) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

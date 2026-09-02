"""Exact-span IO: JSONL records and entity validation."""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path
from typing import Any

from uzbek_ner.labels import ENTITY_LABELS

JsonObject = dict[str, Any]


def read_jsonl_records(
    path: Path,
    *,
    require_entities: bool = True,
    limit: int | None = None,
) -> list[JsonObject]:
    """Read JSONL and validate hackathon record schema."""

    records: list[JsonObject] = []
    seen_hashes: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: empty line")
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            record = _validate_record(raw, path, line_number, require_entities)
            record_hash = record["hash"]
            if record_hash in seen_hashes:
                raise ValueError(f"{path}:{line_number}: duplicate hash {record_hash}")
            seen_hashes.add(record_hash)
            records.append(record)
            if limit is not None and len(records) >= limit:
                break

    if not records:
        raise ValueError(f"{path}: no records")
    return records


def _validate_record(
    raw: Any,
    path: Path,
    line_number: int,
    require_entities: bool,
) -> JsonObject:
    if not isinstance(raw, dict):
        raise ValueError(f"{path}:{line_number}: record must be an object")
    record_hash = raw.get("hash")
    if not isinstance(record_hash, str) or not record_hash:
        raise ValueError(f"{path}:{line_number}: hash must be a non-empty string")

    result: JsonObject = {"hash": record_hash}
    if "text" in raw:
        if not isinstance(raw.get("text"), str):
            raise ValueError(f"{path}:{line_number}: text must be a string")
        result["text"] = raw["text"]
    elif require_entities:
        raise ValueError(f"{path}:{line_number}: text must be a string")
    if require_entities:
        result["entities"] = validate_entities(
            raw.get("entities"),
            result["text"],
            f"{path}:{line_number}",
        )
    elif "entities" in raw:
        # predictions: entities validated later against gold text length
        if not isinstance(raw.get("entities"), list):
            raise ValueError(f"{path}:{line_number}: entities must be an array")
        result["entities"] = raw["entities"]
    return result


def validate_entities(raw: Any, text: str, source: str) -> list[JsonObject]:
    """Validate entity list with exact char spans against text."""

    if not isinstance(raw, list):
        raise ValueError(f"{source}: entities must be an array")

    entities: list[JsonObject] = []
    seen: set[tuple[str, int, int]] = set()
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
            or not 0 <= start < end <= len(text)
        ):
            raise ValueError(f"{source}/entities[{index}]: invalid offsets")
        if text[start:end] == "":
            raise ValueError(f"{source}/entities[{index}]: empty span")
        key = (label, start, end)
        if key in seen:
            raise ValueError(f"{source}/entities[{index}]: duplicate entity")
        seen.add(key)
        entities.append({"label": label, "start": start, "end": end})

    entities.sort(key=lambda item: (item["start"], item["end"], item["label"]))
    for left, right in pairwise(entities):
        if left["end"] > right["start"]:
            raise ValueError(f"{source}: overlapping entities are not allowed")
    return entities

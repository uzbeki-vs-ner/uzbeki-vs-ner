"""Sliding-window tokenization for official JSONL."""

from __future__ import annotations

from typing import Any

from uzbek_ner.spans import align_labels

JsonObject = dict[str, Any]
Offsets = list[tuple[int, int]]
ModelFeature = dict[str, list[int]]

# HF RoBERTa position_ids = pad_id + 1 + token_index; a 512-long window hits index 513.
_ROBERTA_POSITION_TYPES = frozenset({"roberta", "xlm-roberta", "camembert", "longformer"})


def clamp_max_length(
    requested: int,
    *,
    max_position_embeddings: int,
    pad_token_id: int,
    model_type: str,
) -> int:
    """Cap window size so position embeddings stay in-bounds."""

    if requested < 1:
        raise ValueError("max_length must be positive")
    budget = max_position_embeddings
    if model_type in _ROBERTA_POSITION_TYPES:
        budget = max_position_embeddings - pad_token_id - 1
    if budget < 8:
        raise ValueError(f"position table too small for {model_type}: {max_position_embeddings}")
    return min(requested, budget)


def tokenize_windows(
    tokenizer: Any,
    text: str,
    *,
    max_length: int,
    stride: int,
) -> list[tuple[ModelFeature, Offsets]]:
    """Split text into overlapping tokenizer windows with char offsets."""
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        stride=stride,
        return_offsets_mapping=True,
        return_overflowing_tokens=True,
    )
    input_chunks = encoded["input_ids"]
    offset_chunks = encoded["offset_mapping"]
    if input_chunks and isinstance(input_chunks[0], int):
        input_chunks = [input_chunks]
        offset_chunks = [offset_chunks]

    windows: list[tuple[ModelFeature, Offsets]] = []
    for chunk_index, offsets in enumerate(offset_chunks):
        feature: ModelFeature = {}
        for key in ("input_ids", "attention_mask"):
            if key not in encoded:
                continue
            values = encoded[key]
            feature[key] = values[chunk_index] if values and isinstance(values[0], list) else values
        windows.append((feature, [(int(start), int(end)) for start, end in offsets]))
    return windows


def labeled_windows(
    tokenizer: Any,
    record: JsonObject,
    *,
    max_length: int,
    stride: int,
) -> list[JsonObject]:
    """Tokenize one gold record into windows with BIO labels."""
    features: list[JsonObject] = []
    entities = sorted(record["entities"], key=lambda item: (item["start"], item["end"]))
    for feature, offsets in tokenize_windows(
        tokenizer,
        record["text"],
        max_length=max_length,
        stride=stride,
    ):
        labels = align_labels(offsets, entities)
        if all(label == -100 for label in labels):
            continue
        features.append({**feature, "labels": labels, "offset_mapping": offsets})
    return features

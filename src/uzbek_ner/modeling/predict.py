"""Window-merged token classification inference."""

from __future__ import annotations

from typing import Any

import torch
from transformers import PreTrainedTokenizerBase

from uzbek_ner.decode.snap import snap_entities
from uzbek_ner.labels import ID_TO_TAG
from uzbek_ner.modeling.windows import tokenize_windows
from uzbek_ner.spans import decode_bio_tokens

JsonObject = dict[str, Any]


@torch.inference_mode()
def predict_records(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    records: list[JsonObject],
    *,
    max_length: int,
    stride: int,
    batch_size: int,
    device: torch.device,
    snap: bool = True,
) -> list[JsonObject]:
    """Average overlapping-window token scores and decode exact-span entities.

    ``snap`` expands predicted spans to word edges using the input text only
    (no gold). That is a decode constraint, not a leak of the eval split.
    """
    windows: list[tuple[int, dict[str, list[int]], list[tuple[int, int]]]] = []
    for record_index, record in enumerate(records):
        for feature, offsets in tokenize_windows(
            tokenizer,
            record["text"],
            max_length=max_length,
            stride=stride,
        ):
            windows.append((record_index, feature, offsets))

    aggregated: list[dict[tuple[int, int], tuple[torch.Tensor, int]]] = [{} for _ in records]
    model.eval()
    for batch_start in range(0, len(windows), batch_size):
        batch_windows = windows[batch_start : batch_start + batch_size]
        batch = tokenizer.pad(
            [feature for _, feature, _ in batch_windows],
            padding=True,
            return_tensors="pt",
        )
        batch = {key: value.to(device) for key, value in batch.items()}
        probabilities = torch.softmax(model(**batch).logits.float(), dim=-1).cpu()
        for row_index, (record_index, _, offsets) in enumerate(batch_windows):
            record_scores = aggregated[record_index]
            for token_index, (start, end) in enumerate(offsets):
                if start == end:
                    continue
                key = (start, end)
                score = probabilities[row_index, token_index]
                if key in record_scores:
                    previous, count = record_scores[key]
                    record_scores[key] = (previous + score, count + 1)
                else:
                    record_scores[key] = (score.clone(), 1)

    predictions: list[JsonObject] = []
    for record, record_scores in zip(records, aggregated, strict=True):
        tagged = [
            (start, end, ID_TO_TAG[int((score_sum / count).argmax().item())])
            for (start, end), (score_sum, count) in sorted(record_scores.items())
        ]
        entities = decode_bio_tokens(tagged)
        if snap:
            entities = snap_entities(record["text"], entities)
        predictions.append({"hash": record["hash"], "entities": entities})
    return predictions

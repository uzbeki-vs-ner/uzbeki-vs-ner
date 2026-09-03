"""Entity-confidence gate: drop low-scoring spans after BIO decode.

Does not change argmax tags. ``τ`` is fit on an analysis fold; held-out
exact-span F1 is the number that matters. Gold-free at apply time.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from uzbek_ner.decode.kfold import make_folds, score_split
from uzbek_ner.decode.snap import snap_entities
from uzbek_ner.labels import ID_TO_TAG
from uzbek_ner.metrics.calibration import temperature_softmax
from uzbek_ner.modeling.eval_cache import MergedDoc
from uzbek_ner.spans import decode_bio_tokens

JsonObject = dict[str, Any]
EntityKey = tuple[str, int, int]

DEFAULT_TAUS: tuple[float, ...] = (0.0, 0.3, 0.5, 0.7)


def gold_lookup(records: list[JsonObject]) -> dict[str, JsonObject]:
    return {
        record["hash"]: {
            "text": record["text"],
            "entities": {
                (str(row["label"]), int(row["start"]), int(row["end"]))
                for row in record["entities"]
            },
        }
        for record in records
    }


def keyed_predictions(rows: dict[str, list[JsonObject]]) -> dict[str, set[EntityKey]]:
    return {
        record_hash: {(str(row["label"]), int(row["start"]), int(row["end"])) for row in entities}
        for record_hash, entities in rows.items()
    }


def micro_view(metrics: JsonObject) -> dict[str, float | int]:
    micro = metrics["micro"]
    return {
        "precision": float(micro["precision"]),
        "recall": float(micro["recall"]),
        "f1": float(micro["f1"]),
        "tp": int(micro["tp"]),
        "fp": int(micro["fp"]),
        "fn": int(micro["fn"]),
    }


def entity_conf(
    offsets: NDArray[np.int32],
    tags: NDArray[np.integer],
    probs: NDArray[np.floating],
    entity: JsonObject,
    *,
    reduce: str = "mean",
) -> float:
    start, end = int(entity["start"]), int(entity["end"])
    indices = [
        index
        for index, (tok_s, tok_e) in enumerate(offsets)
        if int(tok_s) >= start and int(tok_e) <= end and int(tok_s) < int(tok_e)
    ]
    if not indices:
        return 0.0
    values = np.asarray(
        [float(probs[index, int(tags[index])]) for index in indices], dtype=np.float64
    )
    if reduce == "max":
        return float(values.max())
    return float(values.mean())


def decode_doc(
    doc: MergedDoc,
    *,
    temperature: float = 1.0,
    min_conf: float = 0.0,
    reduce: str = "mean",
    snap: bool = True,
) -> list[JsonObject]:
    if doc.logits.shape[0] == 0:
        return []
    probs = temperature_softmax(doc.logits, temperature)
    tags = probs.argmax(axis=1)
    tagged = [
        (int(start), int(end), ID_TO_TAG[int(tag)])
        for (start, end), tag in zip(doc.offsets, tags, strict=True)
    ]
    entities = decode_bio_tokens(tagged)
    if min_conf > 0:
        entities = [
            entity
            for entity in entities
            if entity_conf(doc.offsets, tags, probs, entity, reduce=reduce) >= min_conf
        ]
    if snap:
        entities = snap_entities(doc.text, entities)
    return entities


def predict_split(
    docs: Sequence[MergedDoc],
    hashes: Sequence[str] | None,
    *,
    temperature: float = 1.0,
    min_conf: float = 0.0,
    reduce: str = "mean",
) -> dict[str, list[JsonObject]]:
    allowed = None if hashes is None else set(hashes)
    return {
        doc.record_hash: decode_doc(
            doc, temperature=temperature, min_conf=min_conf, reduce=reduce, snap=True
        )
        for doc in docs
        if allowed is None or doc.record_hash in allowed
    }


def sweep_confidence_gate(
    docs: Sequence[MergedDoc],
    gold: dict[str, JsonObject],
    *,
    analysis_hashes: Sequence[str],
    held_out_hashes: Sequence[str],
    taus: Sequence[float] = DEFAULT_TAUS,
    temperature: float = 1.0,
    reduce: str = "mean",
) -> dict[str, Any]:
    """Fit ``τ`` on analysis F1; report held-out exact-span metrics."""

    all_hashes = [doc.record_hash for doc in docs]
    tau_list = [float(tau) for tau in taus]
    if 0.0 not in tau_list:
        tau_list.insert(0, 0.0)
    rows: list[dict[str, Any]] = []
    for tau in tau_list:
        preds = predict_split(
            docs, None, temperature=temperature, min_conf=float(tau), reduce=reduce
        )
        keyed = keyed_predictions(preds)
        rows.append(
            {
                "tau": float(tau),
                "analysis": micro_view(score_split(gold, keyed, analysis_hashes)),
                "held_out": micro_view(score_split(gold, keyed, held_out_hashes)),
                "full_dev": micro_view(score_split(gold, keyed, all_hashes)),
                "held_out_predicted": sum(len(preds[h]) for h in held_out_hashes),
            }
        )
    baseline = next(row for row in rows if row["tau"] == 0.0)
    picked = max(rows, key=lambda row: (float(row["analysis"]["f1"]), -float(row["tau"])))
    held_delta = float(picked["held_out"]["f1"]) - float(baseline["held_out"]["f1"])
    return {
        "taus": [float(tau) for tau in taus],
        "reduce": reduce,
        "temperature": float(temperature),
        "grid": rows,
        "baseline_tau": float(baseline["tau"]),
        "selected_tau": float(picked["tau"]),
        "analysis": picked["analysis"],
        "held_out": picked["held_out"],
        "full_dev": picked["full_dev"],
        "baseline_held_out": baseline["held_out"],
        "held_out_delta_f1": held_delta,
    }


def analysis_held_split(hashes: Sequence[str], *, k: int, seed: int) -> tuple[list[str], list[str]]:
    folds = make_folds(list(hashes), k=k, seed=seed)
    analysis = list(folds[0])
    held_out = [record_hash for fold in folds[1:] for record_hash in fold]
    return analysis, held_out

"""Token-level calibration diagnostics (not the organizer exact-span score).

Definitions used by the Mix B temperature experiment:

* **NLL** — mean ``-log p[y]`` over tokens (7-way BIO).
* **Brier** — multiclass mean ``||p - one_hot(y)||^2`` (range ``[0, 2]``).
* **ECE** — max-confidence expected calibration error: bin ``max_k p_k`` into
  equal-width bins, then ``sum_b (n_b / N) |acc_b - conf_b|``. Empty bins are
  skipped. Empty gold docs still contribute their ``O`` tokens.

Argmax tags are invariant to softmax temperature on a *single* logit vector.
Merged overlapping windows use **mean logits**, so argmax stays T-invariant.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArr = NDArray[np.floating]
IntArr = NDArray[np.integer]


def temperature_softmax(logits: FloatArr, temperature: float) -> NDArray[np.float64]:
    """Row-wise ``softmax(z / T)`` with a max-shift for stability."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    z = np.asarray(logits, dtype=np.float64) / float(temperature)
    z = z - z.max(axis=-1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=-1, keepdims=True)


def multiclass_nll(
    probabilities: FloatArr,
    labels: IntArr,
    *,
    eps: float = 1e-12,
) -> float:
    """Mean negative log-likelihood of the gold class."""

    probs = np.asarray(probabilities, dtype=np.float64)
    gold = np.asarray(labels, dtype=np.int64)
    if probs.ndim != 2 or gold.ndim != 1 or probs.shape[0] != gold.shape[0]:
        raise ValueError("probabilities must be [N, C] aligned with labels [N]")
    clipped = np.clip(probs[np.arange(gold.shape[0]), gold], eps, 1.0)
    return float(-np.log(clipped).mean()) if gold.size else 0.0


def multiclass_brier(probabilities: FloatArr, labels: IntArr) -> float:
    """Multiclass Brier score: mean squared error vs one-hot gold."""

    probs = np.asarray(probabilities, dtype=np.float64)
    gold = np.asarray(labels, dtype=np.int64)
    if probs.ndim != 2 or gold.ndim != 1 or probs.shape[0] != gold.shape[0]:
        raise ValueError("probabilities must be [N, C] aligned with labels [N]")
    if gold.size == 0:
        return 0.0
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(gold.shape[0]), gold] = 1.0
    return float(((probs - one_hot) ** 2).sum(axis=1).mean())


def reliability_bins(
    probabilities: FloatArr,
    labels: IntArr,
    *,
    n_bins: int = 15,
) -> dict[str, Any]:
    """Equal-width max-confidence reliability diagram plus scalar ECE."""

    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    probs = np.asarray(probabilities, dtype=np.float64)
    gold = np.asarray(labels, dtype=np.int64)
    if probs.ndim != 2 or gold.ndim != 1 or probs.shape[0] != gold.shape[0]:
        raise ValueError("probabilities must be [N, C] aligned with labels [N]")

    n_tokens = int(gold.shape[0])
    if n_tokens == 0:
        return {
            "n_bins": n_bins,
            "n": 0,
            "ece": 0.0,
            "accuracy": 0.0,
            "mean_confidence": 0.0,
            "confidence_minus_accuracy": 0.0,
            "bins": [],
        }

    confidence = probs.max(axis=1)
    predicted = probs.argmax(axis=1)
    correct = predicted == gold
    accuracy = float(correct.mean())
    mean_confidence = float(confidence.mean())

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[dict[str, float | int]] = []
    ece = 0.0
    # Assign 1.0 to the last bin: edges are [0, 1/n, ..., 1].
    clipped = np.clip(confidence, 0.0, 1.0 - 1e-12)
    indices = np.minimum((clipped * n_bins).astype(np.int64), n_bins - 1)
    for bin_index in range(n_bins):
        mask = indices == bin_index
        count = int(mask.sum())
        lo = float(edges[bin_index])
        hi = float(edges[bin_index + 1])
        if count == 0:
            bins.append(
                {
                    "lo": lo,
                    "hi": hi,
                    "n": 0,
                    "confidence": 0.0,
                    "accuracy": 0.0,
                    "gap": 0.0,
                }
            )
            continue
        bin_conf = float(confidence[mask].mean())
        bin_acc = float(correct[mask].mean())
        gap = abs(bin_acc - bin_conf)
        ece += (count / n_tokens) * gap
        bins.append(
            {
                "lo": lo,
                "hi": hi,
                "n": count,
                "confidence": bin_conf,
                "accuracy": bin_acc,
                "gap": gap,
            }
        )

    return {
        "n_bins": n_bins,
        "n": n_tokens,
        "ece": float(ece),
        "accuracy": accuracy,
        "mean_confidence": mean_confidence,
        "confidence_minus_accuracy": mean_confidence - accuracy,
        "bins": bins,
    }


def token_calibration(
    logits: FloatArr,
    labels: IntArr,
    *,
    temperature: float,
    n_bins: int = 15,
) -> dict[str, Any]:
    """ECE / Brier / NLL for 7-way token logits after temperature scaling."""

    probs = temperature_softmax(logits, temperature)
    diagram = reliability_bins(probs, labels, n_bins=n_bins)
    return {
        "temperature": float(temperature),
        "n": int(np.asarray(labels).shape[0]),
        "ece": diagram["ece"],
        "brier": multiclass_brier(probs, labels),
        "nll": multiclass_nll(probs, labels),
        "accuracy": diagram["accuracy"],
        "mean_confidence": diagram["mean_confidence"],
        "confidence_minus_accuracy": diagram["confidence_minus_accuracy"],
        "reliability": diagram,
    }


def binary_calibration(
    confidence: FloatArr,
    correct: NDArray[np.bool_],
    *,
    n_bins: int = 15,
) -> dict[str, Any]:
    """ECE / Brier / NLL for entity-level correctness vs a scalar confidence."""

    conf = np.asarray(confidence, dtype=np.float64).reshape(-1)
    ok = np.asarray(correct, dtype=np.bool_).reshape(-1)
    if conf.shape != ok.shape:
        raise ValueError("confidence and correct must be aligned")
    n = int(conf.shape[0])
    empty = {
        "n_bins": n_bins,
        "n": 0,
        "ece": 0.0,
        "accuracy": 0.0,
        "mean_confidence": 0.0,
        "confidence_minus_accuracy": 0.0,
        "bins": [],
    }
    if n == 0:
        return {
            "n": 0,
            "ece": 0.0,
            "brier": 0.0,
            "nll": 0.0,
            "accuracy": 0.0,
            "mean_confidence": 0.0,
            "confidence_minus_accuracy": 0.0,
            "reliability": empty,
        }

    # Bin the entity confidence itself (not max(p, 1-p) of a fake 2-way softmax).
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    clipped = np.clip(conf, 0.0, 1.0 - 1e-12)
    indices = np.minimum((clipped * n_bins).astype(np.int64), n_bins - 1)
    ece = 0.0
    bins: list[dict[str, float | int]] = []
    for bin_index in range(n_bins):
        mask = indices == bin_index
        count = int(mask.sum())
        lo = float(edges[bin_index])
        hi = float(edges[bin_index + 1])
        if count == 0:
            bins.append(
                {"lo": lo, "hi": hi, "n": 0, "confidence": 0.0, "accuracy": 0.0, "gap": 0.0}
            )
            continue
        bin_conf = float(conf[mask].mean())
        bin_acc = float(ok[mask].mean())
        gap = abs(bin_acc - bin_conf)
        ece += (count / n) * gap
        bins.append(
            {
                "lo": lo,
                "hi": hi,
                "n": count,
                "confidence": bin_conf,
                "accuracy": bin_acc,
                "gap": gap,
            }
        )
    brier = float(((conf - ok.astype(np.float64)) ** 2).mean())
    clipped_p = np.clip(np.where(ok, conf, 1.0 - conf), 1e-12, 1.0)
    nll = float(-np.log(clipped_p).mean())
    accuracy = float(ok.mean())
    mean_confidence = float(conf.mean())
    return {
        "n": n,
        "ece": float(ece),
        "brier": brier,
        "nll": nll,
        "accuracy": accuracy,
        "mean_confidence": mean_confidence,
        "confidence_minus_accuracy": mean_confidence - accuracy,
        "reliability": {
            "n_bins": n_bins,
            "n": n,
            "ece": float(ece),
            "accuracy": accuracy,
            "mean_confidence": mean_confidence,
            "confidence_minus_accuracy": mean_confidence - accuracy,
            "bins": bins,
        },
    }


def confidence_reading(confidence_minus_accuracy: float, *, gap: float = 0.02) -> str:
    """Over / under / calibrated from mean(max p) − accuracy."""

    if confidence_minus_accuracy > gap:
        return "overconfident"
    if confidence_minus_accuracy < -gap:
        return "underconfident"
    return "calibrated"

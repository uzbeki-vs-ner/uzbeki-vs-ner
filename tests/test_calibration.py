"""CPU tests for token ECE / Brier / NLL (no model download)."""

import numpy as np

from uzbek_ner.metrics.calibration import (
    binary_calibration,
    confidence_reading,
    multiclass_brier,
    multiclass_nll,
    reliability_bins,
    temperature_softmax,
    token_calibration,
)


def test_temperature_softmax_preserves_argmax() -> None:
    logits = np.array([[2.0, 0.1, -1.0], [0.0, 0.0, 3.0]], dtype=np.float64)
    for temperature in (0.1, 1.0, 2.0):
        probs = temperature_softmax(logits, temperature)
        assert np.allclose(probs.sum(axis=1), 1.0)
        np.testing.assert_array_equal(probs.argmax(axis=1), logits.argmax(axis=1))


def test_perfect_onehot_brier_and_nll_are_zero() -> None:
    probs = np.eye(3, dtype=np.float64)
    labels = np.array([0, 1, 2], dtype=np.int64)
    assert multiclass_brier(probs, labels) == 0.0
    assert multiclass_nll(probs, labels) < 1e-10


def test_uniform_three_way_nll_is_log3() -> None:
    probs = np.full((10, 3), 1.0 / 3.0)
    labels = np.zeros(10, dtype=np.int64)
    np.testing.assert_allclose(multiclass_nll(probs, labels), np.log(3.0), rtol=1e-6)


def test_perfectly_calibrated_single_bin_ece_is_zero() -> None:
    # 90% confident and 90% correct, all in the same bin.
    n_ok, n_bad = 9, 1
    probs = np.vstack(
        [np.array([[0.9, 0.05, 0.05]])] * n_ok + [np.array([[0.9, 0.05, 0.05]])] * n_bad
    )
    labels = np.array([0] * n_ok + [1] * n_bad, dtype=np.int64)
    diagram = reliability_bins(probs, labels, n_bins=10)
    assert diagram["n"] == 10
    np.testing.assert_allclose(diagram["ece"], 0.0, atol=1e-12)
    np.testing.assert_allclose(diagram["mean_confidence"], 0.9)
    np.testing.assert_allclose(diagram["accuracy"], 0.9)


def test_overconfident_ece_and_reading() -> None:
    probs = np.tile(np.array([[0.99, 0.005, 0.005]]), (10, 1))
    labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int64)
    metrics = token_calibration(np.log(probs), labels, temperature=1.0, n_bins=10)
    assert metrics["ece"] > 0.4
    assert metrics["confidence_minus_accuracy"] > 0.4
    assert confidence_reading(float(metrics["confidence_minus_accuracy"])) == "overconfident"


def test_empty_tokens_are_zero_not_nan() -> None:
    logits = np.zeros((0, 7), dtype=np.float64)
    labels = np.zeros((0,), dtype=np.int64)
    metrics = token_calibration(logits, labels, temperature=1.0)
    assert metrics["n"] == 0
    assert metrics["ece"] == 0.0
    assert metrics["brier"] == 0.0
    assert metrics["nll"] == 0.0


def test_binary_entity_ece_bins_raw_confidence() -> None:
    # One bin at 0.8 with 80% exact matches → ECE 0, Brier 0.16.
    conf = np.full(5, 0.8, dtype=np.float64)
    correct = np.array([True, True, True, True, False])
    metrics = binary_calibration(conf, correct, n_bins=5)
    assert metrics["n"] == 5
    np.testing.assert_allclose(metrics["ece"], 0.0, atol=1e-12)
    np.testing.assert_allclose(metrics["accuracy"], 0.8)
    np.testing.assert_allclose(metrics["brier"], 0.16, atol=1e-12)

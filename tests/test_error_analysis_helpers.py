"""Unit tests for per-span error classification helpers."""

from uzbek_ner.metrics.error_analysis import (
    classify_predicted_span,
    gold_merge_note,
)


def test_classify_exact_match() -> None:
    gold = {("NAME", 8, 12)}
    assert classify_predicted_span(("NAME", 8, 12), gold) == ("exact", [])


def test_classify_type_mismatch_same_span() -> None:
    gold = {("GEO", 42, 55)}
    kind, hits = classify_predicted_span(("ORG", 42, 55), gold)
    assert kind == "type_mismatch"
    assert hits == [("GEO", 42, 55)]


def test_classify_partial_same_type() -> None:
    gold = {("NAME", 8, 12), ("NAME", 17, 24)}
    kind, hits = classify_predicted_span(("NAME", 8, 24), gold)
    assert kind == "partial_same_type"
    assert hits == [("NAME", 8, 12), ("NAME", 17, 24)]


def test_classify_spurious() -> None:
    gold = {("ORG", 0, 3)}
    kind, hits = classify_predicted_span(("ORG", 100, 110), gold)
    assert kind == "spurious"
    assert hits == []


def test_gold_merge_note_for_split_name() -> None:
    text = "UFC 327 Jiri Van Blaydes Torres Walker"
    pred = ("NAME", 8, 24)
    gold_parts = [("NAME", 8, 12), ("NAME", 17, 24)]
    note = gold_merge_note(pred, gold_parts, text)
    assert note is not None
    assert "merges 2 gold NAME spans" in note
    assert "' Van '" in note

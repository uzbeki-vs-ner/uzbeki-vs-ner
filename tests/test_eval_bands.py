"""Hackathon exact-span F1 reading-guide bands."""

from itertools import pairwise

from uzbek_ner.evaldash.bands import (
    F1_BANDS,
    GRAFANA_F1_THRESHOLD_STEPS,
    band_for_f1,
    scale_payload,
)


def test_half_open_intervals_cover_unit_range() -> None:
    payload = scale_payload()
    bands = payload["bands"]
    assert isinstance(bands, list)
    assert bands[0]["min_inclusive"] == 0.0
    for previous, current in pairwise(bands):
        assert previous["max_exclusive"] == current["min_inclusive"]
    assert bands[-1]["max_exclusive"] > 1.0


def test_half_is_baseline_not_broken() -> None:
    band = band_for_f1(0.50)
    assert band.id == "baseline"


def test_point_seven_is_competitive() -> None:
    band = band_for_f1(0.70)
    assert band.id == "competitive"


def test_uztext_smoke_lands_in_baseline() -> None:
    assert band_for_f1(0.578).id == "baseline"


def test_probe_and_ceiling() -> None:
    assert band_for_f1(0.08).id == "broken"
    assert band_for_f1(0.30).id == "weak"
    assert band_for_f1(0.60).id == "competitive"
    assert band_for_f1(0.80).id == "strong"
    assert band_for_f1(0.90).id == "excellent"
    assert band_for_f1(1.0).id == "excellent"
    assert band_for_f1(-0.1).id == "broken"


def test_grafana_threshold_cuts_match_bands() -> None:
    cuts = [value for _, value in GRAFANA_F1_THRESHOLD_STEPS if value is not None]
    expected = [band.min_inclusive for band in F1_BANDS[1:]]
    assert cuts == expected

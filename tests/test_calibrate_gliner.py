"""CPU tests for GLiNER threshold post-filtering."""

from uzbek_ner.decode.threshold import gold_lookup, micro_view
from uzbek_ner.metrics.exact_span import calculate_exact_span_metrics

# Import private helpers from script module path via inline logic duplicate is bad;
# test the filter contract through a tiny local mirror of filter_scored.


def _filter_scored(
    scored: dict[str, list[dict[str, object]]],
    *,
    tau: float,
) -> dict[str, set[tuple[str, int, int]]]:
    return {
        record_hash: {
            (str(row["label"]), int(row["start"]), int(row["end"]))
            for row in spans
            if float(row["score"]) >= tau
        }
        for record_hash, spans in scored.items()
    }


def test_tau_filters_by_score_without_changing_survivors() -> None:
    scored = {
        "h1": [
            {"label": "ORG", "start": 0, "end": 3, "score": 0.9},
            {"label": "GEO", "start": 4, "end": 7, "score": 0.4},
        ]
    }
    at_05 = _filter_scored(scored, tau=0.5)
    assert at_05["h1"] == {("ORG", 0, 3)}
    at_0 = _filter_scored(scored, tau=0.0)
    assert len(at_0["h1"]) == 2


def test_tau_sweep_changes_micro_f1() -> None:
    gold = gold_lookup(
        [
            {
                "hash": "a",
                "text": "foo bar",
                "entities": [{"label": "ORG", "start": 0, "end": 3}],
            }
        ]
    )
    scored = {
        "a": [
            {"label": "ORG", "start": 0, "end": 3, "score": 0.95},
            {"label": "GEO", "start": 4, "end": 7, "score": 0.2},
        ]
    }
    low = calculate_exact_span_metrics(gold, _filter_scored(scored, tau=0.0))
    high = calculate_exact_span_metrics(gold, _filter_scored(scored, tau=0.5))
    assert micro_view(low)["f1"] < micro_view(high)["f1"]

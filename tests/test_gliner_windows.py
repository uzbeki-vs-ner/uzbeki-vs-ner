"""CPU tests for GLiNER sliding-word windows."""

from uzbek_ner.gliner_windows import (
    DEFAULT_MAX_WORDS,
    DEFAULT_STRIDE,
    merge_window_entities,
    predict_records_windowed,
    word_window_spans,
)


def test_default_window_constants() -> None:
    assert DEFAULT_MAX_WORDS == 384
    assert DEFAULT_STRIDE == 128


def test_single_window_when_short() -> None:
    text = "alpha beta gamma"
    assert word_window_spans(text, max_words=8, stride=4) == [(0, len(text))]


def test_overlapping_word_windows_cover_tail() -> None:
    words = " ".join(f"w{i}" for i in range(10))
    spans = word_window_spans(words, max_words=4, stride=2)
    assert spans[0] == (0, words.index("w3") + len("w3"))
    assert spans[-1][1] == len(words)


def test_merge_prefers_higher_score_on_duplicate() -> None:
    merged = merge_window_entities(
        [
            {"label": "ORG", "start": 0, "end": 3, "score": 0.6},
            {"label": "ORG", "start": 0, "end": 3, "score": 0.9},
        ]
    )
    assert len(merged) == 1
    assert merged[0]["score"] == 0.9


def test_merge_greedy_drops_lower_overlap() -> None:
    merged = merge_window_entities(
        [
            {"label": "NAME", "start": 0, "end": 10, "score": 0.95},
            {"label": "NAME", "start": 5, "end": 12, "score": 0.8},
        ]
    )
    assert len(merged) == 1
    assert merged[0]["start"] == 0


class _FakeModel:
    def batch_predict_entities(
        self,
        texts: list[str],
        labels: list[str],
        *,
        flat_ner: bool,
        threshold: float,
        batch_size: int,
    ) -> list[list[dict[str, object]]]:
        del labels, flat_ner, threshold, batch_size
        out: list[list[dict[str, object]]] = []
        for text in texts:
            if "tail" in text:
                start = text.index("tail")
                out.append(
                    [
                        {
                            "label": "GEO",
                            "start": start,
                            "end": start + len("tail"),
                            "score": 0.99,
                        }
                    ]
                )
            else:
                out.append([])
        return out


def test_predict_records_windowed_shifts_offsets() -> None:
    words = ["head"] + [f"f{i}" for i in range(400)] + ["tail"]
    text = " ".join(words)
    tail_start = text.index("tail")
    preds, _meta = predict_records_windowed(
        _FakeModel(),
        [{"hash": "h1", "text": text}],
        max_words=200,
        stride=100,
        batch_size=4,
    )
    entities = preds["h1"]
    assert any(
        row["start"] == tail_start and row["end"] == tail_start + len("tail") for row in entities
    )

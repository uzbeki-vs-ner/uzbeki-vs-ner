"""CPU tests for sliding-window BIO features (no model download)."""

from uzbek_ner.labels import TAG_TO_ID
from uzbek_ner.modeling.windows import clamp_max_length, labeled_windows, tokenize_windows


class _FakeTokenizer:
    def __call__(self, text: str, **kwargs: object) -> dict[str, list[object]]:
        _ = kwargs
        if text == "Ali Toshkent":
            return {
                "input_ids": [0, 11, 22, 1],
                "attention_mask": [1, 1, 1, 1],
                "offset_mapping": [(0, 0), (0, 3), (4, 12), (0, 0)],
            }
        return {
            "input_ids": [0, 1],
            "attention_mask": [1, 1],
            "offset_mapping": [(0, 0), (0, 0)],
        }


def test_tokenize_windows_single_chunk() -> None:
    windows = tokenize_windows(_FakeTokenizer(), "Ali Toshkent", max_length=16, stride=8)
    assert len(windows) == 1
    feature, offsets = windows[0]
    assert feature["input_ids"] == [0, 11, 22, 1]
    assert offsets[1] == (0, 3)


def test_labeled_windows_aligns_entities() -> None:
    record = {
        "hash": "x",
        "text": "Ali Toshkent",
        "entities": [
            {"label": "NAME", "start": 0, "end": 3},
            {"label": "GEO", "start": 4, "end": 12},
        ],
    }
    features = labeled_windows(_FakeTokenizer(), record, max_length=16, stride=8)
    assert len(features) == 1
    assert features[0]["labels"] == [-100, TAG_TO_ID["B-NAME"], TAG_TO_ID["B-GEO"], -100]


def test_clamp_max_length_roberta_pad_offset() -> None:
    assert (
        clamp_max_length(
            512,
            max_position_embeddings=512,
            pad_token_id=1,
            model_type="roberta",
        )
        == 510
    )


def test_clamp_max_length_xlmr_has_headroom() -> None:
    assert (
        clamp_max_length(
            512,
            max_position_embeddings=514,
            pad_token_id=1,
            model_type="xlm-roberta",
        )
        == 512
    )


def test_labeled_windows_skips_special_only() -> None:
    record = {"hash": "empty", "text": "", "entities": []}
    assert labeled_windows(_FakeTokenizer(), record, max_length=16, stride=8) == []

"""Tests for BIO alignment and span decoding."""

from uzbek_ner.labels import ID_TO_TAG, TAG_TO_ID
from uzbek_ner.spans import align_labels, decode_bio_tokens


def test_align_labels_single_entity() -> None:
    entities = [{"label": "NAME", "start": 0, "end": 3}]
    offsets = [(0, 0), (0, 3), (4, 12), (0, 0)]
    labels = align_labels(offsets, entities)
    assert labels == [-100, TAG_TO_ID["B-NAME"], TAG_TO_ID["O"], -100]


def test_decode_bio_tokens_roundtrip_style() -> None:
    tokens = [
        (0, 3, "B-NAME"),
        (4, 12, "B-GEO"),
    ]
    entities = decode_bio_tokens(tokens)
    assert entities == [
        {"label": "NAME", "start": 0, "end": 3},
        {"label": "GEO", "start": 4, "end": 12},
    ]


def test_decode_bio_tokens_merges_i_tags() -> None:
    tokens = [
        (0, 2, "B-ORG"),
        (2, 5, "I-ORG"),
        (2, 5, "I-ORG"),
    ]
    entities = decode_bio_tokens(tokens)
    assert entities == [{"label": "ORG", "start": 0, "end": 5}]


def test_id_to_tag_covers_all_labels() -> None:
    assert len(ID_TO_TAG) == len(TAG_TO_ID)

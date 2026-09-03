"""Fossil CPU tests. Not collected by pytest (testpaths = tests/)."""

import numpy as np

from continue_span import continue_one_word, entity_mass, next_word, prev_word  # type: ignore[import-not-found]
from uzbek_ner.labels import TAG_TO_ID


def test_next_and_prev_word_skip_spaces_only() -> None:
    text = "Devid Ebi, Toshkent"
    assert next_word(text, 5) == (6, 9)  # after "Devid"
    assert prev_word(text, 6) == (0, 5)
    assert next_word(text, 9) is None  # comma, not a word


def test_entity_mass_averages_overlapping_tokens() -> None:
    offsets = np.array([[0, 5], [6, 9], [10, 12]], dtype=np.int32)
    probs = np.zeros((3, 7), dtype=np.float32)
    probs[1, TAG_TO_ID["I-NAME"]] = 0.8
    probs[1, TAG_TO_ID["B-NAME"]] = 0.1
    assert abs(entity_mass("NAME", 6, 9, offsets, probs) - 0.9) < 1e-6
    assert entity_mass("NAME", 10, 12, offsets, probs) == 0.0


def test_continue_one_word_extends_when_i_mass_high() -> None:
    text = "Devid Ebi keldi"
    offsets = np.array([[0, 5], [6, 9], [10, 15]], dtype=np.int32)
    probs = np.zeros((3, 7), dtype=np.float32)
    probs[1, TAG_TO_ID["I-NAME"]] = 0.6
    rows = continue_one_word(
        text,
        [{"label": "NAME", "start": 0, "end": 5}],
        offsets,
        probs,
        tau=0.5,
        direction="right",
    )
    assert rows == [{"label": "NAME", "start": 0, "end": 9}]


def test_continue_one_word_skips_when_mass_below_tau() -> None:
    text = "Devid Ebi keldi"
    offsets = np.array([[0, 5], [6, 9]], dtype=np.int32)
    probs = np.zeros((2, 7), dtype=np.float32)
    probs[1, TAG_TO_ID["I-NAME"]] = 0.2
    rows = continue_one_word(
        text,
        [{"label": "NAME", "start": 0, "end": 5}],
        offsets,
        probs,
        tau=0.5,
        direction="right",
    )
    assert rows == [{"label": "NAME", "start": 0, "end": 5}]


def test_continue_does_not_glue_two_predicted_entities() -> None:
    text = "Devid Ebi"
    offsets = np.array([[0, 5], [6, 9]], dtype=np.int32)
    probs = np.ones((2, 7), dtype=np.float32)
    rows = continue_one_word(
        text,
        [
            {"label": "NAME", "start": 0, "end": 5},
            {"label": "NAME", "start": 6, "end": 9},
        ],
        offsets,
        probs,
        tau=0.1,
        direction="right",
    )
    assert rows == [
        {"label": "NAME", "start": 0, "end": 5},
        {"label": "NAME", "start": 6, "end": 9},
    ]


def test_continue_both_picks_higher_mass_word_only() -> None:
    text = "Janob Devid Ebi"
    offsets = np.array([[0, 5], [6, 11], [12, 15]], dtype=np.int32)
    probs = np.zeros((3, 7), dtype=np.float32)
    probs[0, TAG_TO_ID["I-NAME"]] = 0.4
    probs[2, TAG_TO_ID["I-NAME"]] = 0.9
    rows = continue_one_word(
        text,
        [{"label": "NAME", "start": 6, "end": 11}],
        offsets,
        probs,
        tau=0.3,
        direction="both",
        max_words=1,
    )
    assert rows == [{"label": "NAME", "start": 6, "end": 15}]

"""CPU tests for gold-free span snap and offset-mode ties."""

from uzbek_ner.decode.kfold import fit_offset_mode, make_folds
from uzbek_ner.decode.snap import expand_to_word_edges, mode_int, snap_entities


def test_expand_to_word_edges_repairs_subword_cut() -> None:
    text = "Mark Karni keldi"
    start, end = expand_to_word_edges(text, 0, 8)  # "Mark Kar"
    assert text[start:end] == "Mark Karni"


def test_snap_is_noop_when_span_already_covers_the_word() -> None:
    text = "Toshkentda yomg'ir"
    rows = snap_entities(text, [{"label": "GEO", "start": 0, "end": 10}])
    assert rows == [{"label": "GEO", "start": 0, "end": 10}]


def test_snap_extends_lemma_into_attached_locative() -> None:
    text = "Kanadada yomg'ir"
    rows = snap_entities(text, [{"label": "GEO", "start": 0, "end": 6}])  # "Kanada"
    assert rows == [{"label": "GEO", "start": 0, "end": 8}]


def test_mode_int_breaks_ties_toward_zero() -> None:
    assert mode_int([2, 2, 0, 0]) == 0
    assert mode_int([3, 3, 3]) == 3
    assert mode_int([]) == 0


def test_make_folds_partition() -> None:
    hashes = [f"h{i}" for i in range(10)]
    folds = make_folds(hashes, k=5, seed=42)
    assert len(folds) == 5
    flat = [item for fold in folds for item in fold]
    assert sorted(flat) == sorted(hashes)
    assert len(set(flat)) == 10


def test_fit_offset_mode_on_narrower_end() -> None:
    gold = {
        "a": {
            "text": "Toshkentda",
            "entities": {("GEO", 0, 10)},
        }
    }
    predictions = {"a": {("GEO", 0, 8)}}
    fitted = fit_offset_mode(gold, predictions, ["a"])
    assert fitted["n_pairs"] == 1
    assert fitted["d_start"] == 0
    assert fitted["d_end"] == -2
    assert fitted["joint_d_start"] == 0
    assert fitted["joint_d_end"] == -2

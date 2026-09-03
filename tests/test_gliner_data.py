from uzbek_ner.gliner_data import convert_record, split_words


def test_split_words_keeps_hyphen_and_offsets() -> None:
    text = "Milliy-banki."
    tokens = split_words(text)
    assert tokens[0][0] == "Milliy-banki"
    assert text[tokens[0][1] : tokens[0][2]] == "Milliy-banki"
    assert tokens[-1][0] == "."


def test_convert_maps_char_span_to_token_indices() -> None:
    text = "Toshkent shahrida yashaydi"
    start = text.index("Toshkent")
    end = start + len("Toshkent")
    row = convert_record(
        {
            "text": text,
            "entities": [{"label": "GEO", "start": start, "end": end}],
        },
        max_words=32,
    )
    assert row is not None
    assert row["tokenized_text"][0] == "Toshkent"
    assert row["ner"] == [[0, 0, "GEO"]]


def test_convert_drops_spans_past_max_words() -> None:
    text = "alpha beta gamma"
    start = text.index("gamma")
    end = start + len("gamma")
    row = convert_record(
        {
            "text": text,
            "entities": [{"label": "ORG", "start": start, "end": end}],
        },
        max_words=2,
    )
    assert row is not None
    assert row["tokenized_text"] == ["alpha", "beta"]
    assert row["ner"] == []

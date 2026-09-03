"""CPU tests for Mix B gold+silver concatenation."""

import json
from pathlib import Path

from uzbek_ner.io.mix import load_train_mix


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
        encoding="utf-8",
    )


def test_load_train_mix_skips_gold_hash_and_caps(tmp_path: Path) -> None:
    gold_path = tmp_path / "gold.jsonl"
    extra_path = tmp_path / "silver.jsonl"
    _write_jsonl(
        gold_path,
        [
            {
                "hash": "gold-1",
                "text": "Ali",
                "entities": [{"label": "NAME", "start": 0, "end": 3}],
            }
        ],
    )
    _write_jsonl(
        extra_path,
        [
            {
                "hash": "gold-1",
                "text": "Ali",
                "entities": [{"label": "NAME", "start": 0, "end": 3}],
            },
            {
                "hash": "sv-1",
                "text": "Toshkent",
                "entities": [{"label": "GEO", "start": 0, "end": 8}],
            },
            {
                "hash": "sv-2",
                "text": "Uzbekiston",
                "entities": [{"label": "GEO", "start": 0, "end": 10}],
            },
        ],
    )
    mixed, meta = load_train_mix(gold_path, [extra_path], extra_cap=1, seed=42)
    hashes = {row["hash"] for row in mixed}
    assert "gold-1" in hashes
    assert meta["gold_docs"] == 1
    assert meta["extra_docs"] == 1
    assert meta["extra_skipped_hash"] == 1
    assert meta["train_docs"] == 2
    assert hashes <= {"gold-1", "sv-1", "sv-2"}
    assert len(hashes) == 2

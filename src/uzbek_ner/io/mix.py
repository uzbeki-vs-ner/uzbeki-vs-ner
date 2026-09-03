"""Concatenate official gold with extra JSONL (silver Mix B)."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from uzbek_ner.io.jsonl import read_jsonl_records


def load_train_mix(
    gold_path: Path,
    extra_paths: list[Path],
    *,
    extra_cap: int | None,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Official gold plus optional silver. Extra hashes must not collide with gold."""

    gold = read_jsonl_records(gold_path)
    extra: list[dict[str, Any]] = []
    seen = {record["hash"] for record in gold}
    skipped_hash = 0
    for path in extra_paths:
        for record in read_jsonl_records(path):
            if record["hash"] in seen:
                skipped_hash += 1
                continue
            seen.add(record["hash"])
            extra.append(record)
    rng = random.Random(seed)
    rng.shuffle(extra)
    if extra_cap is not None:
        extra = extra[:extra_cap]
    mixed = gold + extra
    rng.shuffle(mixed)
    mix_meta = {
        "gold_docs": len(gold),
        "extra_docs": len(extra),
        "extra_paths": [str(path) for path in extra_paths],
        "extra_cap": extra_cap,
        "extra_skipped_hash": skipped_hash,
        "train_docs": len(mixed),
    }
    return mixed, mix_meta

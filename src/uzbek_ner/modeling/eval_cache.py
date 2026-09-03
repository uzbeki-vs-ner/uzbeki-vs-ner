"""Packed eval-logit cache (GPU fill, CPU reload).

Window logits live in memmaps; decode uses mean logits over overlapping
windows aligned to unique ``(start, end)`` char spans. The confidence-gate
stage reads ``merged_mean_logits.npz`` and does not need the hidden states.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from loguru import logger
from numpy.typing import NDArray
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from uzbek_ner.labels import TAGS
from uzbek_ner.modeling.heads import load_token_classifier
from uzbek_ner.modeling.windows import clamp_max_length, tokenize_windows
from uzbek_ner.spans import align_labels

N_LABELS = len(TAGS)
HIDDEN_WIDTH = 768

JsonObject = dict[str, Any]


@dataclass
class MergedDoc:
    record_hash: str
    text: str
    offsets: NDArray[np.int32]
    logits: NDArray[np.float32]
    counts: NDArray[np.int32]
    gold_labels: NDArray[np.int64]
    gold_keys: set[tuple[str, int, int]]


def cache_complete(cache_dir: Path) -> bool:
    index_path = cache_dir / "index.json"
    if not index_path.is_file():
        return False
    try:
        meta = json.loads(index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if meta.get("status") != "complete":
        return False
    files = meta.get("files", {})
    merged_name = files.get("merged")
    if not merged_name or not (cache_dir / merged_name).is_file():
        return False
    hidden_name = files.get("hidden")
    if not hidden_name:
        return True
    hidden_path = cache_dir / hidden_name
    if not hidden_path.is_file():
        return False
    hidden_shape = tuple(meta["hidden_shape"])
    expected = int(hidden_shape[0]) * int(hidden_shape[1]) * 2
    return hidden_path.stat().st_size == expected


def merged_ready(cache_dir: Path) -> bool:
    """CPU sweep only needs the packed mean-logit table."""

    index_path = cache_dir / "index.json"
    merged_path = cache_dir / "merged_mean_logits.npz"
    if merged_path.is_file():
        return True
    if not index_path.is_file():
        return False
    try:
        meta = json.loads(index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    name = meta.get("files", {}).get("merged")
    return bool(name) and (cache_dir / str(name)).is_file()


def open_memmaps(
    cache_dir: Path, meta: dict[str, Any], *, mode: Literal["r", "w+", "r+"]
) -> dict[str, np.memmap]:
    files = meta["files"]
    n_tokens = int(meta["n_tokens"])
    hidden_width = int(meta["hidden_width"])
    return {
        "hidden": np.memmap(
            cache_dir / files["hidden"],
            dtype=np.float16,
            mode=mode,
            shape=(n_tokens, hidden_width),
        ),
        "logits": np.memmap(
            cache_dir / files["logits"],
            dtype=np.float32,
            mode=mode,
            shape=(n_tokens, N_LABELS),
        ),
        "input_ids": np.memmap(
            cache_dir / files["input_ids"],
            dtype=np.int32,
            mode=mode,
            shape=(n_tokens,),
        ),
        "attention_mask": np.memmap(
            cache_dir / files["attention_mask"],
            dtype=np.uint8,
            mode=mode,
            shape=(n_tokens,),
        ),
        "offsets": np.memmap(
            cache_dir / files["offsets"],
            dtype=np.int32,
            mode=mode,
            shape=(n_tokens, 2),
        ),
        "labels": np.memmap(
            cache_dir / files["labels"],
            dtype=np.int32,
            mode=mode,
            shape=(n_tokens,),
        ),
    }


def merge_from_cache(
    records: list[JsonObject],
    cache_dir: Path,
    meta: dict[str, Any],
) -> list[MergedDoc]:
    maps = open_memmaps(cache_dir, meta, mode="r")
    grouped: dict[int, list[tuple[int, int]]] = {}
    for window in meta["windows"]:
        grouped.setdefault(int(window["doc_index"]), []).append(
            (int(window["token_start"]), int(window["token_count"]))
        )
    merged: list[MergedDoc] = []
    for doc_index, record in enumerate(records):
        sums: dict[tuple[int, int], np.ndarray] = {}
        counts: dict[tuple[int, int], int] = {}
        for token_start, token_count in grouped.get(doc_index, []):
            offsets = np.asarray(maps["offsets"][token_start : token_start + token_count])
            logits = np.asarray(maps["logits"][token_start : token_start + token_count])
            for token_index, (start, end) in enumerate(offsets):
                start_i, end_i = int(start), int(end)
                if start_i == end_i:
                    continue
                key = (start_i, end_i)
                if key in sums:
                    sums[key] += logits[token_index]
                    counts[key] += 1
                else:
                    sums[key] = logits[token_index].astype(np.float64, copy=True)
                    counts[key] = 1
        keys = sorted(sums)
        if keys:
            offset_arr = np.asarray(keys, dtype=np.int32)
            count_arr = np.asarray([counts[key] for key in keys], dtype=np.int32)
            logit_arr = np.stack([sums[key] / counts[key] for key in keys], axis=0).astype(
                np.float32
            )
        else:
            offset_arr = np.zeros((0, 2), dtype=np.int32)
            count_arr = np.zeros((0,), dtype=np.int32)
            logit_arr = np.zeros((0, N_LABELS), dtype=np.float32)
        gold_entities = list(record["entities"])
        gold_labels = np.asarray(
            align_labels([(int(s), int(e)) for s, e in offset_arr], gold_entities),
            dtype=np.int64,
        )
        merged.append(
            MergedDoc(
                record_hash=record["hash"],
                text=record["text"],
                offsets=offset_arr,
                logits=logit_arr,
                counts=count_arr,
                gold_labels=gold_labels,
                gold_keys={
                    (str(row["label"]), int(row["start"]), int(row["end"])) for row in gold_entities
                },
            )
        )
    for array in maps.values():
        array.flush()
    return merged


def pack_merged(docs: list[MergedDoc]) -> dict[str, Any]:
    ptr = [0]
    for doc in docs:
        ptr.append(ptr[-1] + int(doc.offsets.shape[0]))
    n = ptr[-1]
    offsets = np.zeros((n, 2), dtype=np.int32)
    logits = np.zeros((n, N_LABELS), dtype=np.float32)
    counts = np.zeros((n,), dtype=np.int32)
    gold_labels = np.zeros((n,), dtype=np.int64)
    for doc, start, end in zip(docs, ptr[:-1], ptr[1:], strict=True):
        offsets[start:end] = doc.offsets
        logits[start:end] = doc.logits
        counts[start:end] = doc.counts
        gold_labels[start:end] = doc.gold_labels
    return {
        "ptr": np.asarray(ptr, dtype=np.int64),
        "offsets": offsets,
        "logits": logits,
        "counts": counts,
        "gold_labels": gold_labels,
    }


def load_merged_npz(path: Path, records: list[JsonObject]) -> list[MergedDoc]:
    packed = np.load(path, allow_pickle=False)
    by_hash = {record["hash"]: record for record in records}
    hashes = [record["hash"] for record in records]
    ptr = np.asarray(packed["ptr"])
    offsets = packed["offsets"]
    logits = packed["logits"]
    counts = packed["counts"]
    gold_labels = packed["gold_labels"]
    if int(ptr[-1]) != int(offsets.shape[0]):
        raise RuntimeError("merged ptr does not match offsets")
    if len(hashes) != int(ptr.shape[0]) - 1:
        raise RuntimeError("merged doc count does not match gold records")
    docs: list[MergedDoc] = []
    for index, record_hash in enumerate(hashes):
        start = int(ptr[index])
        end = int(ptr[index + 1])
        record = by_hash[record_hash]
        docs.append(
            MergedDoc(
                record_hash=record_hash,
                text=record["text"],
                offsets=np.asarray(offsets[start:end], dtype=np.int32),
                logits=np.asarray(logits[start:end], dtype=np.float32),
                counts=np.asarray(counts[start:end], dtype=np.int32),
                gold_labels=np.asarray(gold_labels[start:end], dtype=np.int64),
                gold_keys={
                    (str(row["label"]), int(row["start"]), int(row["end"]))
                    for row in record["entities"]
                },
            )
        )
    return docs


def fill_cache(
    *,
    checkpoint: Path,
    records: list[JsonObject],
    cache_dir: Path,
    batch_size: int,
    max_length: int,
    stride: int,
) -> dict[str, Any]:
    """Encode official eval windows. Call under ``flock outputs/.gpu.lock``."""

    if not torch.cuda.is_available():
        raise RuntimeError("GPU cache fill needs CUDA; wait on flock and retry")
    device = torch.device("cuda")
    cache_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True, use_fast=True)
    model = load_token_classifier(checkpoint)
    max_length = clamp_max_length(
        max_length,
        max_position_embeddings=int(model.config.max_position_embeddings),
        pad_token_id=int(model.config.pad_token_id or 0),
        model_type=str(model.config.model_type),
    )
    hidden_width = int(model.config.hidden_size)
    if hidden_width != HIDDEN_WIDTH:
        logger.warning("hidden_size {} (expected {})", hidden_width, HIDDEN_WIDTH)
    model.to(device)
    model.eval()
    backbone = getattr(model, "roberta", None) or model.base_model
    dropout = getattr(model, "dropout", torch.nn.Identity())

    windows: list[tuple[int, int, dict[str, list[int]], list[tuple[int, int]]]] = []
    for doc_index, record in enumerate(tqdm(records, desc="tokenize official dev")):
        for window_index, (feature, offsets) in enumerate(
            tokenize_windows(tokenizer, record["text"], max_length=max_length, stride=stride)
        ):
            windows.append((doc_index, window_index, feature, offsets))
    token_starts: list[int] = []
    cursor = 0
    for _doc_index, _window_index, feature, offsets in windows:
        length = len(offsets)
        if length != len(feature["input_ids"]):
            raise RuntimeError("offset/input_ids length mismatch")
        token_starts.append(cursor)
        cursor += length
    n_tokens = cursor
    logger.info("windows={} tokens={} docs={}", len(windows), n_tokens, len(records))

    files = {
        "hidden": "hidden_f16.dat",
        "logits": "logits_f32.dat",
        "input_ids": "input_ids_i32.dat",
        "attention_mask": "attention_mask_u8.dat",
        "offsets": "offsets_i32.dat",
        "labels": "labels_i32.dat",
        "merged": "merged_mean_logits.npz",
    }
    meta: dict[str, Any] = {
        "status": "writing",
        "format": "packed_memmap_v1",
        "checkpoint": str(checkpoint),
        "gold": "data/official/dev.jsonl",
        "created_at": datetime.now(UTC).isoformat(),
        "max_length": max_length,
        "stride": stride,
        "hidden_width": hidden_width,
        "n_labels": N_LABELS,
        "n_docs": len(records),
        "n_windows": len(windows),
        "n_tokens": n_tokens,
        "hidden_dtype": "float16",
        "logits_dtype": "float32",
        "hidden_shape": [n_tokens, hidden_width],
        "logits_shape": [n_tokens, N_LABELS],
        "files": files,
        "window_merge": (
            "Per-window logits are stored packed. Decode uses mean logits over "
            "overlapping windows aligned to unique (start, end) char spans. "
            "softmax(mean_logits / T) is argmax-invariant in T."
        ),
        "windows": [
            {
                "hash": records[doc_index]["hash"],
                "doc_index": doc_index,
                "window_index": window_index,
                "token_start": token_start,
                "token_count": len(offsets),
            }
            for token_start, (doc_index, window_index, _feature, offsets) in zip(
                token_starts, windows, strict=True
            )
        ],
    }
    maps = open_memmaps(cache_dir, meta, mode="w+")

    for batch_start in tqdm(range(0, len(windows), batch_size), desc="encode windows"):
        batch_rows = windows[batch_start : batch_start + batch_size]
        batch_starts = token_starts[batch_start : batch_start + batch_size]
        padded = tokenizer.pad(
            [feature for _d, _w, feature, _offsets in batch_rows],
            padding=True,
            return_tensors="pt",
        )
        padded = {key: value.to(device) for key, value in padded.items()}
        with torch.inference_mode():
            hidden = backbone(
                input_ids=padded["input_ids"],
                attention_mask=padded["attention_mask"],
            ).last_hidden_state
            logits = model.classifier(dropout(hidden))
        hidden_cpu = hidden.float().cpu().numpy()
        logits_cpu = logits.float().cpu().numpy()
        input_ids_cpu = padded["input_ids"].cpu().numpy()
        mask_cpu = padded["attention_mask"].cpu().numpy()
        for row, ((doc_index, _window_index, _feature, offsets), token_start) in enumerate(
            zip(batch_rows, batch_starts, strict=True)
        ):
            length = len(offsets)
            end = token_start + length
            maps["hidden"][token_start:end] = hidden_cpu[row, :length].astype(
                np.float16, copy=False
            )
            maps["logits"][token_start:end] = logits_cpu[row, :length].astype(
                np.float32, copy=False
            )
            maps["input_ids"][token_start:end] = input_ids_cpu[row, :length].astype(
                np.int32, copy=False
            )
            maps["attention_mask"][token_start:end] = mask_cpu[row, :length].astype(
                np.uint8, copy=False
            )
            maps["offsets"][token_start:end] = np.asarray(offsets, dtype=np.int32)
            maps["labels"][token_start:end] = np.asarray(
                align_labels(offsets, records[doc_index]["entities"]),
                dtype=np.int32,
            )

    for array in maps.values():
        array.flush()
    del maps
    del model
    del backbone
    torch.cuda.empty_cache()

    merged = merge_from_cache(records, cache_dir, meta)
    packed = pack_merged(merged)
    np.savez(cache_dir / files["merged"], **packed)
    meta["hashes"] = [doc.record_hash for doc in merged]
    meta["status"] = "complete"
    meta["merged_shape"] = {
        "n_docs": len(merged),
        "n_content_tokens": int(packed["ptr"][-1]),
        "offsets": list(packed["offsets"].shape),
        "logits": list(packed["logits"].shape),
        "counts": list(packed["counts"].shape),
        "gold_labels": list(packed["gold_labels"].shape),
        "ptr": list(packed["ptr"].shape),
    }
    meta["nbytes"] = {
        name: (cache_dir / filename).stat().st_size for name, filename in files.items()
    }
    meta["reload"] = (
        "import json, numpy as np\n"
        "from pathlib import Path\n"
        f"root = Path({str(cache_dir)!r})\n"
        "meta = json.loads((root / 'index.json').read_text())\n"
        "merged = np.load(root / meta['files']['merged'], allow_pickle=False)\n"
    )
    (cache_dir / "index.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    logger.info("cache complete → {}", cache_dir)
    return meta

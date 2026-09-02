#!/usr/bin/env python3
# ruff: noqa: RUF001
"""Explatory Model Analysis of local BERT MLM checkpoints vs public Uzbek NER heads.

Answers: are the downloaded MaskedLM checkpoints applicable as NER encoders?
Can public NER models teach ORG/NAME/GEO via distillation? Should we drop the
MLM head and fine-tune a 7-tag TokenClassification head?

Does not full-train. GPU work (short random-head forward + frozen linear probe)
runs under outputs/.gpu.lock via flock and is skipped after 45 minutes.

Usage:
    HF_HUB_OFFLINE=1 uv run python scripts/ema_models.py
    uv run python scripts/ema_models.py --skip-gpu
    uv run python scripts/ema_models.py --skip-remote
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from uzbek_ner.labels import ID_TO_TAG, TAG_TO_ID, TAGS  # noqa: E402
from uzbek_ner.settings import get_settings  # noqa: E402
from uzbek_ner.spans import align_labels, decode_bio_tokens  # noqa: E402

LATIN_RE = re.compile(r"[A-Za-zÀ-ÿ]")
CYR_RE = re.compile(r"[\u0400-\u04FF]")
WORD_RE = re.compile(r"\S+")

LATIN_SUFFIXES = (
    "lardagi",
    "laridan",
    "larining",
    "lariga",
    "larida",
    "lardan",
    "larni",
    "larga",
    "larda",
    "ning",
    "dagi",
    "dan",
    "ga",
    "ni",
    "da",
    "lar",
)
CYR_SUFFIXES = (
    "лардаги",
    "ларидан",
    "ларининг",
    "ларига",
    "ларида",
    "лардан",
    "ларни",
    "ларга",
    "ларда",
    "нинг",
    "даги",
    "дан",
    "га",
    "ни",
    "да",
    "лар",
)

LOCAL_MODELS = {
    "exp0_xlm_roberta_base": {
        "display": "XLM-RoBERTa-base",
        "hf_id": "FacebookAI/xlm-roberta-base",
        "role": "multilingual MLM encoder (organizer baseline stack)",
    },
    "exp1_uztext_roberta": {
        "display": "uztext-3Gb-BPE-Roberta",
        "hf_id": "rifkat/uztext-3Gb-BPE-Roberta",
        "role": "monolingual Uzbek MLM (news, Latin+Cyrillic)",
    },
    "exp2_mdeberta_v3_base": {
        "display": "mDeBERTa-v3-base",
        "hf_id": "microsoft/mdeberta-v3-base",
        "role": "multilingual DeBERTa-v3 MLM encoder",
    },
}

# Official 7-tag BIO. Public teachers map into this or are dropped.
HACKATHON_BIO = list(TAGS)
TEACHER_TYPE_MAP = {
    "PERSON": "NAME",
    "PER": "NAME",
    "ORG": "ORG",
    "LOC": "GEO",
    "GPE": "GEO",
    "FAC": "GEO",
    "FACILITY": "GEO",
}

# Card example from risqaliyevds/xlm-roberta-large-ner: suffix left outside GEO.
RISQ_CARD_EXAMPLE = {
    "text": "Shavkat Mirziyoyev Rossiyada rasmiy safarda bo'ldi.",
    "card_spans": [
        {"start": 0, "end": 7, "label": "PERSON", "surface": "Shavkat"},
        {"start": 8, "end": 18, "label": "PERSON", "surface": "Mirziyoyev"},
        {"start": 19, "end": 26, "label": "GPE", "surface": "Rossiya"},
    ],
    "official_policy_would_want": [
        {"start": 0, "end": 18, "label": "NAME", "surface": "Shavkat Mirziyoyev"},
        {"start": 19, "end": 28, "label": "GEO", "surface": "Rossiyada"},
    ],
}

SUFFIX_PROBES = (
    "Toshkent",
    "Toshkentda",
    "Toshkentdagi",
    "Toshkentning",
    "Oʻzbekiston",
    "Oʻzbekistonda",
    "Rossiya",
    "Rossiyada",
    "Jungkook",
    "Jungkookning",
    "Jungkookni",
    "Telegram",
    "Telegramda",
    "Qodirov",
    "Qodirovning",
    "Эрон",
    "Oqtepa",
    "Murad Buildings",
)

KPOP_NAMES = (
    "Jungkook",
    "Jungkookning",
    "Jk",
    "Tae",
    "Taehyung",
    "Jimin",
    "Jin",
    "Yoongi",
)

DROP_KEY_PREFIXES = (
    "lm_head",
    "cls.",
    "lm_predictions",
    "mask_predictions",
    "generator",
)


def percentile(values: list[int] | list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    k = (len(ordered) - 1) * p
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return float(ordered[lo])
    return float(ordered[lo] * (hi - k) + ordered[hi] * (k - lo))


def attached_suffix(surface: str) -> str | None:
    token = surface.strip()
    if not token or " " in token:
        return None
    lower = token.lower()
    for suffix in LATIN_SUFFIXES + CYR_SUFFIXES:
        if len(lower) > len(suffix) + 2 and lower.endswith(suffix):
            stem = lower[: -len(suffix)]
            if stem.endswith(("'", "ʼ", "ʻ", "‘", "’")):
                return suffix
            if stem[-1:].isalpha():
                return suffix
    return None


def script_of(text: str) -> str:
    has_lat = bool(LATIN_RE.search(text))
    has_cyr = bool(CYR_RE.search(text))
    if has_lat and has_cyr:
        return "mixed"
    if has_cyr:
        return "cyrillic"
    if has_lat:
        return "latin"
    return "other"


def load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if limit is not None and len(records) >= limit:
                break
    return records


def print_section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def n_params_from_shape(shape: tuple[int, ...]) -> int:
    n = 1
    for dim in shape:
        n *= int(dim)
    return n


def classify_tensor_key(key: str) -> str:
    lower = key.replace("/", ".")
    parts = lower.split(".")
    if "pooler" in parts:
        return "drop_pooler"
    if "_weight" in parts or lower.endswith("._weight"):
        return "drop_duplicate_buffer"
    for prefix in DROP_KEY_PREFIXES:
        if lower.startswith(prefix) or f".{prefix}" in f".{lower}":
            if prefix == "generator" or lower.startswith("generator"):
                return "drop_generator"
            return "drop_lm_head"
    if "classifier" in parts:
        return "reinit_classifier"
    return "transfer_encoder"


def iter_weight_shapes(model_dir: Path) -> list[tuple[str, tuple[int, ...]]]:
    """Read tensor names/shapes on CPU without placing the full model on GPU."""
    safetensors_path = model_dir / "model.safetensors"
    if safetensors_path.exists():
        from safetensors import safe_open

        rows: list[tuple[str, tuple[int, ...]]] = []
        with safe_open(str(safetensors_path), framework="pt", device="cpu") as handle:
            for key in handle.keys():  # noqa: SIM118
                rows.append((key, tuple(handle.get_slice(key).get_shape())))
        return rows

    bin_path = model_dir / "pytorch_model.bin"
    if not bin_path.exists():
        msg = f"no model.safetensors or pytorch_model.bin in {model_dir}"
        raise FileNotFoundError(msg)
    import torch

    state = torch.load(str(bin_path), map_location="cpu", weights_only=True)
    rows = [(key, tuple(tensor.shape)) for key, tensor in state.items()]
    del state
    return rows


def bucket_parameters(shapes: list[tuple[str, tuple[int, ...]]]) -> dict[str, Any]:
    buckets: dict[str, int] = defaultdict(int)
    examples: dict[str, list[str]] = defaultdict(list)
    tied_note = None
    keys = {key for key, _ in shapes}
    if "lm_head.decoder.weight" not in keys and any(
        k.endswith("word_embeddings.weight") for k in keys
    ):
        tied_note = (
            "lm_head.decoder.weight absent — embeddings are tied; decoder is not extra params"
        )
    for key, shape in shapes:
        bucket = classify_tensor_key(key)
        n = n_params_from_shape(shape)
        buckets[bucket] += n
        if len(examples[bucket]) < 8:
            examples[bucket].append(f"{key} {list(shape)}")
    total = sum(buckets.values())
    classifier_reinit = 768 * len(TAGS) + len(TAGS)  # overwritten per model hidden size
    return {
        "n_tensors": len(shapes),
        "n_params_on_disk": total,
        "buckets_params": dict(buckets),
        "buckets_pct": {k: round(100.0 * v / total, 3) if total else 0 for k, v in buckets.items()},
        "examples": dict(examples),
        "tied_lm_head": tied_note,
        "default_hidden_for_head_note": classifier_reinit,
    }


def inspect_config(model_dir: Path) -> dict[str, Any]:
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    extras = sorted(
        p.name
        for p in model_dir.iterdir()
        if p.is_file()
        and p.name
        not in {
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
            "sentencepiece.bpe.model",
            "spm.model",
            "README.md",
            ".gitattributes",
        }
    )
    return {
        "architectures": config.get("architectures"),
        "model_type": config.get("model_type"),
        "hidden_size": config.get("hidden_size"),
        "num_hidden_layers": config.get("num_hidden_layers"),
        "num_attention_heads": config.get("num_attention_heads"),
        "intermediate_size": config.get("intermediate_size"),
        "max_position_embeddings": config.get("max_position_embeddings"),
        "vocab_size": config.get("vocab_size"),
        "type_vocab_size": config.get("type_vocab_size"),
        "relative_attention": config.get("relative_attention"),
        "position_biased_input": config.get("position_biased_input"),
        "files": extras,
        "has_generator_bin": (model_dir / "pytorch_model.generator.bin").exists(),
        "disk_bytes": {
            p.name: p.stat().st_size
            for p in model_dir.iterdir()
            if p.is_file() and p.suffix in {".bin", ".safetensors", ".h5", ".onnx", ".msgpack"}
        },
    }


def load_local_tokenizer(model_dir: Path):
    from transformers import AutoTokenizer, DebertaV2Tokenizer

    try:
        return AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, use_fast=True)
    except Exception:
        pass
    if model_dir.name == "exp2_mdeberta_v3_base" or (model_dir / "spm.model").exists():
        return DebertaV2Tokenizer(vocab_file=str(model_dir / "spm.model"), do_lower_case=False)
    return AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, use_fast=False)


def tokenize_pieces(tokenizer, text: str) -> list[str]:
    if hasattr(tokenizer, "tokenize"):
        return list(tokenizer.tokenize(text))
    ids = tokenizer.encode(text, add_special_tokens=False)
    convert = getattr(tokenizer, "convert_ids_to_tokens", None)
    if convert is None:
        return [str(i) for i in ids]
    return list(convert(ids))


def fertility_entity_surfaces(tokenizer, records: list[dict[str, Any]]) -> dict[str, Any]:
    per_label: dict[str, list[int]] = defaultdict(list)
    suffix_vs_lemma: dict[str, list[int]] = {"suffix": [], "lemma": []}
    kpop: list[int] = []
    other_name: list[int] = []
    single_piece = 0
    n_ent = 0
    suffix_last_token_is_suffix = 0
    suffix_n = 0
    for rec in records:
        text = rec.get("text") or ""
        for ent in rec.get("entities") or []:
            surface = text[ent["start"] : ent["end"]]
            pieces = tokenize_pieces(tokenizer, surface)
            n_tok = max(len(pieces), 1)
            n_ent += 1
            per_label[ent["label"]].append(n_tok)
            if n_tok == 1:
                single_piece += 1
            suf = attached_suffix(surface)
            if suf:
                suffix_vs_lemma["suffix"].append(n_tok)
                suffix_n += 1
                last = pieces[-1].lstrip("▁_Ġ") if pieces else ""
                if last.lower() == suf.lower() or last.lower().endswith(suf.lower()):
                    suffix_last_token_is_suffix += 1
            elif " " not in surface:
                suffix_vs_lemma["lemma"].append(n_tok)
            if ent["label"] == "NAME":
                if any(surface.casefold().startswith(n.casefold()) for n in KPOP_NAMES):
                    kpop.append(n_tok)
                else:
                    other_name.append(n_tok)

    def mean_or_none(vals: list[int]) -> float | None:
        return round(float(statistics.fmean(vals)), 3) if vals else None

    return {
        "n_entities": n_ent,
        "single_piece_pct": round(100.0 * single_piece / n_ent, 2) if n_ent else 0,
        "mean_subwords_by_label": {lab: mean_or_none(vals) for lab, vals in per_label.items()},
        "median_subwords_by_label": {
            lab: round(percentile(vals, 0.5), 3) if vals else None
            for lab, vals in per_label.items()
        },
        "mean_subwords_suffix_entities": mean_or_none(suffix_vs_lemma["suffix"]),
        "mean_subwords_lemma_entities": mean_or_none(suffix_vs_lemma["lemma"]),
        "suffix_entities": suffix_n,
        "suffix_split_as_own_last_token_pct": (
            round(100.0 * suffix_last_token_is_suffix / suffix_n, 2) if suffix_n else 0
        ),
        "mean_subwords_kpop_name": mean_or_none(kpop),
        "mean_subwords_other_name": mean_or_none(other_name),
        "n_kpop_name_mentions": len(kpop),
        "n_other_name_mentions": len(other_name),
    }


def probe_suffix_splits(tokenizer) -> dict[str, list[str]]:
    return {surface: tokenize_pieces(tokenizer, surface) for surface in SUFFIX_PROBES}


def convert_mlm_to_token_classifier_cpu(model_dir: Path, hidden_size: int) -> dict[str, Any]:
    """Instantiate TokenClassification from a MaskedLM dir on CPU; record load report."""
    import torch
    from transformers import AutoConfig, AutoModelForTokenClassification

    config = AutoConfig.from_pretrained(str(model_dir), local_files_only=True)
    config.num_labels = len(TAGS)
    config.id2label = dict(ID_TO_TAG)
    config.label2id = dict(TAG_TO_ID)
    config.architectures = None
    model = AutoModelForTokenClassification.from_pretrained(
        str(model_dir),
        config=config,
        local_files_only=True,
        ignore_mismatched_sizes=True,
        torch_dtype=torch.float32,
    )
    model.eval()
    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    head_params = 0
    encoder_params = 0
    head_names: list[str] = []
    for name, param in model.named_parameters():
        if "classifier" in name or name.startswith("score"):
            head_params += param.numel()
            head_names.append(f"{name} {list(param.shape)}")
        else:
            encoder_params += param.numel()
    expected_head = hidden_size * len(TAGS) + len(TAGS)
    del model
    gc.collect()
    return {
        "token_classification_params": n_total,
        "encoder_params_loaded": encoder_params,
        "classifier_params": head_params,
        "classifier_tensors": head_names,
        "expected_classifier_params": expected_head,
        "trainable_params": n_trainable,
        "transfer_pct_of_ner_model": round(100.0 * encoder_params / n_total, 4) if n_total else 0,
        "reinit_pct_of_ner_model": round(100.0 * head_params / n_total, 4) if n_total else 0,
        "note": (
            "AutoModelForTokenClassification.from_pretrained on a MaskedLM dir keeps "
            "embeddings+encoder, drops lm_head, randomly inits classifier (7 BIO tags)."
        ),
    }


def inspect_remote_ner(*, offline: bool) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    # Large teacher: config only (already known; never pull 560M weights).
    reports["risqaliyevds/xlm-roberta-large-ner"] = inspect_risqaliyevds_config(offline=offline)
    reports["jamshidahmadov/roberta-ner-uz"] = inspect_jamshid(offline=offline)
    return reports


def inspect_risqaliyevds_config(*, offline: bool) -> dict[str, Any]:
    id2label = {
        0: "O",
        1: "B-LOC",
        2: "I-LOC",
        3: "B-ORG",
        4: "I-ORG",
        5: "B-PERSON",
        6: "I-PERSON",
        7: "B-DATE",
        8: "I-DATE",
        9: "B-MONEY",
        10: "I-MONEY",
        11: "B-PERCENT",
        12: "I-PERCENT",
        13: "B-QUANTITY",
        14: "I-QUANTITY",
        15: "B-TIME",
        16: "I-TIME",
        17: "B-PRODUCT",
        18: "I-PRODUCT",
        19: "B-EVENT",
        20: "I-EVENT",
        21: "B-WORK_OF_ART",
        22: "I-WORK_OF_ART",
        23: "B-LANGUAGE",
        24: "I-LANGUAGE",
        25: "B-CARDINAL",
        26: "I-CARDINAL",
        27: "B-ORDINAL",
        28: "I-ORDINAL",
        29: "B-NORP",
        30: "I-NORP",
        31: "B-FACILITY",
        32: "I-FACILITY",
        33: "B-LAW",
        34: "I-LAW",
        35: "B-GPE",
        36: "I-GPE",
    }
    source = "hardcoded_from_hub_config"
    if not offline:
        try:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(
                repo_id="risqaliyevds/xlm-roberta-large-ner",
                filename="config.json",
            )
            cfg = json.loads(Path(path).read_text(encoding="utf-8"))
            raw = cfg.get("id2label") or {}
            id2label = {int(k): str(v) for k, v in raw.items()}
            source = str(path)
        except Exception as exc:
            source = f"hardcoded_fallback ({type(exc).__name__}: {exc})"

    types = sorted({lab.split("-", 1)[-1] for lab in id2label.values() if lab != "O"})
    mapped = {typ: TEACHER_TYPE_MAP[typ] for typ in types if typ in TEACHER_TYPE_MAP}
    dropped = [typ for typ in types if typ not in TEACHER_TYPE_MAP]
    return {
        "gated": False,
        "weights_downloaded": False,
        "architecture": "XLMRobertaForTokenClassification",
        "base": "xlm-roberta-large (24L, hidden=1024, ~559M params)",
        "disk_safetensors_bytes_hub": 2_257_715_447,
        "n_labels": len(id2label),
        "id2label": {str(k): v for k, v in sorted(id2label.items())},
        "entity_types": types,
        "map_to_hackathon": mapped,
        "drop_types": dropped,
        "mappable_type_pct": round(100.0 * len(mapped) / len(types), 1) if types else 0,
        "config_source": source,
        "card_example": RISQ_CARD_EXAMPLE,
        "card_note": (
            "Hub card tags Rossiyada as B-GPE on subword ▁Rossiya only (chars 19-26). "
            "Official GEO wants Rossiyada (19-28). Distilling BIO from this teacher "
            "teaches the opposite suffix policy."
        ),
        "vram_note": (
            "fp32 ~2.2 GB weights; fp16 ~1.1 GB plus activations. Teacher+student on "
            "RTX A1000 6GB during distill is not practical. Full FT of large is out."
        ),
        "verdict": "ignore_as_teacher",
    }


def inspect_jamshid(*, offline: bool) -> dict[str, Any]:
    card_types = ["LOC", "PERSON", "ORG", "PRODUCT", "DATE", "TIME", "LANGUAGE", "GPE"]
    mapped = {typ: TEACHER_TYPE_MAP[typ] for typ in card_types if typ in TEACHER_TYPE_MAP}
    dropped = [typ for typ in card_types if typ not in TEACHER_TYPE_MAP]
    gated = True
    error = None
    readme_ok = False
    if not offline:
        try:
            from huggingface_hub import hf_hub_download, model_info

            info = model_info("jamshidahmadov/roberta-ner-uz")
            gated = bool(getattr(info, "gated", False))
            try:
                hf_hub_download("jamshidahmadov/roberta-ner-uz", filename="config.json")
                gated = False
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            try:
                hf_hub_download("jamshidahmadov/roberta-ner-uz", filename="README.md")
                readme_ok = True
            except Exception:
                pass
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    return {
        "gated": gated,
        "weights_downloaded": False,
        "access_error": error,
        "readme_visible": readme_ok,
        "architecture": "XLMRobertaForTokenClassification (card)",
        "base": "FacebookAI/xlm-roberta-base — same encoder we already have as local MLM",
        "dataset": "risqaliyevds/uzbek_ner (~19k news rows)",
        "training_card": "1 epoch, lr=1e-6, batch=4; reported token accuracy 0.979 / P/R/F 0.97",
        "card_types_B_only_listed": card_types,
        "map_to_hackathon": mapped,
        "drop_types": dropped,
        "mappable_type_pct": round(100.0 * len(mapped) / len(card_types), 1),
        "note": (
            "HF repo is gated (auto). Card lists only B-* tags (no I-*), matching the "
            "broken B-only conversion already rejected in EDA for ner_prepared_uzbek. "
            "Even if ungated, this is one-epoch news NER on the same silver we would "
            "map as Mix B — not a teacher over the official ontology."
        ),
        "verdict": "ignore_as_teacher",
    }


def pick_demo_records(train: list[dict[str, Any]], n: int = 8) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []

    def add(predicate, why: str) -> None:
        if len(chosen) >= n:
            return
        for rec in train:
            if any(rec.get("hash") == c.get("hash") for c in chosen):
                continue
            try:
                if predicate(rec):
                    row = {
                        "hash": rec.get("hash"),
                        "text": rec["text"],
                        "entities": rec.get("entities") or [],
                        "why": why,
                    }
                    chosen.append(row)
                    return
            except Exception:
                continue

    def surface(rec: dict[str, Any], ent: dict[str, Any]) -> str:
        return rec["text"][ent["start"] : ent["end"]]

    add(
        lambda r: any("Jungkook" in surface(r, e) for e in (r.get("entities") or [])),
        "K-pop NAME (Jungkook)",
    )
    add(
        lambda r: any(
            e["label"] == "GEO" and attached_suffix(surface(r, e)) in {"da", "да"}
            for e in (r.get("entities") or [])
        ),
        "GEO with locative suffix",
    )
    add(
        lambda r: any(
            surface(r, e) in {"Telegram", "Instagram"} for e in (r.get("entities") or [])
        ),
        "platform ORG",
    )
    add(
        lambda r: any(
            "Oqtepa" in surface(r, e) or "Murad" in surface(r, e) for e in (r.get("entities") or [])
        ),
        "local brand ORG",
    )
    add(
        lambda r: script_of(r.get("text") or "") == "cyrillic" and bool(r.get("entities")),
        "Cyrillic news",
    )
    add(
        lambda r: script_of(r.get("text") or "") == "mixed" and bool(r.get("entities")),
        "mixed script",
    )
    add(lambda r: not (r.get("entities") or []), "empty document (no gold entities)")
    add(
        lambda r: any(
            e["label"] == "NAME" and attached_suffix(surface(r, e))
            for e in (r.get("entities") or [])
        ),
        "NAME with genitive/accusative suffix",
    )
    return chosen[:n]


def demo_tokenization(
    tokenizers: dict[str, Any], records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rec in records:
        text = rec["text"]
        gold = [
            {
                "start": e["start"],
                "end": e["end"],
                "label": e["label"],
                "surface": text[e["start"] : e["end"]],
            }
            for e in rec.get("entities") or []
        ]
        tok_views = {}
        for alias, tok in tokenizers.items():
            pieces = tokenize_pieces(tok, text[:240])
            tok_views[alias] = {
                "n_pieces_head": len(tokenize_pieces(tok, text)),
                "head_pieces": pieces[:40],
            }
        rows.append(
            {
                "why": rec.get("why"),
                "hash": rec.get("hash"),
                "text_preview": text[:240],
                "gold": gold,
                "tokenizers": tok_views,
            }
        )
    return rows


def try_acquire_gpu_lock(lock_path: Path, timeout_s: float) -> Any | None:
    import fcntl

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.touch(exist_ok=True)
    handle = lock_path.open("a+")
    deadline = time.time() + timeout_s
    while True:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            handle.write(f"ema_models pid={os.getpid()} ts={time.time():.0f}\n")
            handle.flush()
            return handle
        except BlockingIOError:
            if time.time() >= deadline:
                handle.close()
                return None
            time.sleep(5.0)


def release_gpu_lock(handle: Any) -> None:
    import fcntl

    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def decode_logits_to_spans(
    text: str,
    offsets: list[tuple[int, int]],
    pred_ids: list[int],
) -> list[dict[str, Any]]:
    tokens: list[tuple[int, int, str]] = []
    for (start, end), pred in zip(offsets, pred_ids, strict=False):
        if start == end:
            continue
        tag = ID_TO_TAG.get(int(pred), "O")
        tokens.append((start, end, tag))
    ents = decode_bio_tokens(tokens)
    for ent in ents:
        ent["surface"] = text[ent["start"] : ent["end"]]
    return ents


def collect_encoder_features(
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    *,
    device: Any,
    max_length: int,
) -> tuple[Any, Any, list[dict[str, Any]]]:
    import numpy as np
    import torch

    xs: list[Any] = []
    ys: list[int] = []
    packed: list[dict[str, Any]] = []
    hidden_dim = int(getattr(model.config, "hidden_size", 768))
    with torch.no_grad():
        for rec in records:
            enc = tokenizer(
                rec["text"],
                return_offsets_mapping=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            offsets = [(int(s), int(e)) for s, e in enc.pop("offset_mapping")[0].tolist()]
            gold_ids = align_labels(offsets, rec.get("entities") or [])
            enc_dev = {k: v.to(device) for k, v in enc.items()}
            hidden = model(**enc_dev, output_hidden_states=True).hidden_states[-1][0]
            hidden_cpu = hidden.float().cpu().numpy()
            hidden_dim = int(hidden_cpu.shape[-1])
            token_rows = []
            for i, ((start, end), lab) in enumerate(zip(offsets, gold_ids, strict=True)):
                if start == end or lab < 0:
                    continue
                xs.append(hidden_cpu[i])
                ys.append(int(lab))
                token_rows.append((start, end, i, int(lab)))
            packed.append({"record": rec, "offsets": offsets, "token_rows": token_rows})
    features = np.stack(xs) if xs else np.zeros((0, hidden_dim))
    return features, np.array(ys, dtype=np.int64), packed


def gpu_random_head_and_features(
    models_dir: Path,
    tokenizers: dict[str, Any],
    demo: list[dict[str, Any]],
    probe_train: list[dict[str, Any]],
    probe_eval: list[dict[str, Any]],
    max_length: int,
) -> dict[str, Any]:
    """CUDA only: random TokenClassification head + frozen encoder hidden states."""
    import torch
    from transformers import AutoConfig, AutoModelForTokenClassification

    device = torch.device("cuda")
    out: dict[str, Any] = {}
    for alias, meta in LOCAL_MODELS.items():
        path = models_dir / alias
        print(f"[gpu] {alias} TokenClassification forward …")
        config = AutoConfig.from_pretrained(str(path), local_files_only=True)
        config.num_labels = len(TAGS)
        config.id2label = dict(ID_TO_TAG)
        config.label2id = dict(TAG_TO_ID)
        model = AutoModelForTokenClassification.from_pretrained(
            str(path),
            config=config,
            local_files_only=True,
            ignore_mismatched_sizes=True,
            torch_dtype=torch.float16,
        )
        model.to(device)
        model.eval()
        tok = tokenizers[alias]
        random_head_preds: list[dict[str, Any]] = []
        with torch.no_grad():
            for rec in demo:
                enc = tok(
                    rec["text"],
                    return_offsets_mapping=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                offsets = [(int(s), int(e)) for s, e in enc.pop("offset_mapping")[0].tolist()]
                enc = {k: v.to(device) for k, v in enc.items()}
                logits = model(**enc).logits[0]
                pred_ids = logits.argmax(dim=-1).tolist()
                spans = decode_logits_to_spans(rec["text"], offsets, pred_ids)
                random_head_preds.append(
                    {
                        "why": rec.get("why"),
                        "gold": [
                            {
                                "label": e["label"],
                                "start": e["start"],
                                "end": e["end"],
                                "surface": rec["text"][e["start"] : e["end"]],
                            }
                            for e in rec.get("entities") or []
                        ],
                        "random_head_spans": spans[:12],
                        "n_pred": len(spans),
                    }
                )

        print(f"[gpu] {alias} frozen hidden states …")
        x_train, y_train, _packed_train = collect_encoder_features(
            model, tok, probe_train, device=device, max_length=max_length
        )
        x_eval, y_eval, packed_eval = collect_encoder_features(
            model, tok, probe_eval, device=device, max_length=max_length
        )
        del model
        gc.collect()
        torch.cuda.empty_cache()
        out[alias] = {
            "display": meta["display"],
            "random_head_is_not_ner": True,
            "random_head_demo": random_head_preds,
            "_probe_arrays": (x_train, y_train, x_eval, y_eval, packed_eval),
        }
    return out


def fit_frozen_linear_probes(gpu_models: dict[str, Any]) -> dict[str, Any]:
    """CPU logistic regression on frozen encoder states. Does not need the GPU lock."""
    from sklearn.linear_model import LogisticRegression

    cleaned: dict[str, Any] = {}
    for alias, payload in gpu_models.items():
        x_train, y_train, x_eval, y_eval, packed_eval = payload.pop("_probe_arrays")
        majority = Counter(y_train.tolist()).most_common(1)[0]
        majority_acc = float((y_eval == majority[0]).mean()) if len(y_eval) else 0.0
        probe_metrics: dict[str, Any] = {
            "n_train_tokens": len(y_train),
            "n_eval_tokens": len(y_eval),
            "majority_tag": ID_TO_TAG.get(int(majority[0]), str(majority[0])),
            "majority_token_acc": round(majority_acc, 4),
        }
        if len(y_train) and len(set(y_train.tolist())) > 1 and len(y_eval):
            print(f"[cpu] frozen linear probe {alias} …")
            clf = LogisticRegression(max_iter=200, class_weight="balanced", solver="lbfgs")
            clf.fit(x_train, y_train)
            pred_eval = clf.predict(x_eval)
            probe_metrics["frozen_linear_token_acc"] = round(float((pred_eval == y_eval).mean()), 4)
            cursor = 0
            pred_by_hash: dict[str, set[tuple[str, int, int]]] = {}
            gold_by_hash: dict[str, dict[str, Any]] = {}
            for item in packed_eval:
                rec = item["record"]
                tokens: list[tuple[int, int, str]] = []
                for start, end, _i, _lab in item["token_rows"]:
                    tag = ID_TO_TAG.get(int(pred_eval[cursor]), "O")
                    tokens.append((start, end, tag))
                    cursor += 1
                ents = decode_bio_tokens(tokens)
                pred_by_hash[rec["hash"]] = {(e["label"], e["start"], e["end"]) for e in ents}
                gold_by_hash[rec["hash"]] = {
                    "text": rec["text"],
                    "entities": {
                        (e["label"], e["start"], e["end"]) for e in rec.get("entities") or []
                    },
                }
            tp = fp = fn = 0
            for rec_hash, gold_rec in gold_by_hash.items():
                gold_set = gold_rec["entities"]
                pred_set = pred_by_hash.get(rec_hash, set())
                tp += len(gold_set & pred_set)
                fp += len(pred_set - gold_set)
                fn += len(gold_set - pred_set)
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec_v = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * prec * rec_v / (prec + rec_v) if prec + rec_v else 0.0
            probe_metrics["frozen_linear_exact_span_f1"] = round(f1, 4)
            probe_metrics["frozen_linear_tp_fp_fn"] = {"tp": tp, "fp": fp, "fn": fn}
        else:
            probe_metrics["frozen_linear_token_acc"] = None
            probe_metrics["note"] = "not enough class diversity to fit a probe"
        payload["frozen_linear_probe"] = probe_metrics
        cleaned[alias] = payload
    return cleaned


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=settings.official_train)
    parser.add_argument("--models-dir", type=Path, default=settings.models / "pretrained")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs" / "ema")
    parser.add_argument("--lock", type=Path, default=REPO_ROOT / "outputs" / ".gpu.lock")
    parser.add_argument("--lock-timeout-s", type=float, default=45 * 60)
    parser.add_argument("--probe-n", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--skip-gpu", action="store_true")
    parser.add_argument("--skip-remote", action="store_true")
    parser.add_argument(
        "--offline",
        action="store_true",
        default=os.environ.get("HF_HUB_OFFLINE") == "1",
        help="Do not hit the Hub (default if HF_HUB_OFFLINE=1)",
    )
    args = parser.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    print_section("Local MLM checkpoints (CPU)")
    local_report: dict[str, Any] = {}
    tokenizers: dict[str, Any] = {}
    for alias, meta in LOCAL_MODELS.items():
        path = args.models_dir / alias
        print(f"-- {alias} --")
        cfg = inspect_config(path)
        shapes = iter_weight_shapes(path)
        buckets = bucket_parameters(shapes)
        hidden = int(cfg["hidden_size"] or 768)
        print(
            f"  arch={cfg['architectures']} type={cfg['model_type']} L={cfg['num_hidden_layers']} H={hidden}"
        )
        print(f"  vocab={cfg['vocab_size']} files={list(cfg['disk_bytes'])}")
        print(
            f"  params_on_disk={buckets['n_params_on_disk']:,} buckets={buckets['buckets_params']}"
        )
        print("  converting MaskedLM → TokenClassification on CPU …")
        conversion = convert_mlm_to_token_classifier_cpu(path, hidden)
        print(
            f"  NER model params={conversion['token_classification_params']:,} "
            f"encoder={conversion['encoder_params_loaded']:,} "
            f"new_head={conversion['classifier_params']:,} "
            f"({conversion['reinit_pct_of_ner_model']}% reinit)"
        )
        tok = load_local_tokenizer(path)
        tokenizers[alias] = tok
        local_report[alias] = {
            **meta,
            "path": str(path),
            "config": cfg,
            "weights": buckets,
            "chop_head": conversion,
            "tokenizer_class": type(tok).__name__,
            "tokenizer_vocab_size": int(getattr(tok, "vocab_size", 0) or len(tok)),
        }

    print_section("Public NER heads (config / card; no large weights)")
    if args.skip_remote:
        remote_report: dict[str, Any] = {"skipped": True}
        print("skipped")
    else:
        remote_report = inspect_remote_ner(offline=args.offline)
        for name, rep in remote_report.items():
            print(f"-- {name} --")
            print(f"  gated={rep.get('gated')} verdict={rep.get('verdict')}")
            print(f"  map={rep.get('map_to_hackathon')} drop={rep.get('drop_types')}")
            if name.startswith("risqaliyevds"):
                print(f"  card example: {json.dumps(rep.get('card_example'), ensure_ascii=False)}")

    print_section("Official sample (demo sentences + extra fertility)")
    train = load_jsonl(args.train)
    rng = random.Random(args.seed)
    demo = pick_demo_records(train, n=8)
    demo_rows = demo_tokenization(tokenizers, demo)
    for row in demo_rows:
        print(f"* {row['why']}: {row['text_preview']!r}")
        gold_s = ", ".join(f"{g['label']}:{g['surface']}" for g in row["gold"][:6]) or "(empty)"
        print(f"  gold: {gold_s}")

    # Extra fertility on a 1500-doc slice (new vs EDA's 200-doc tok/word table).
    sample = train if len(train) <= 1500 else rng.sample(train, 1500)
    fertility: dict[str, Any] = {}
    splits: dict[str, Any] = {}
    for alias, tok in tokenizers.items():
        fertility[alias] = fertility_entity_surfaces(tok, sample)
        splits[alias] = probe_suffix_splits(tok)
        print(
            f"{alias} entity fertility (1500-doc sample): {json.dumps(fertility[alias], ensure_ascii=False)}"
        )
        print(f"  Toshkentda pieces: {splits[alias].get('Toshkentda')}")
        print(f"  Jungkookning pieces: {splits[alias].get('Jungkookning')}")

    gpu_report: dict[str, Any] = {"ran": False}
    if args.skip_gpu:
        gpu_report = {"ran": False, "reason": "skipped by flag"}
        print_section("GPU")
        print("skipped (--skip-gpu)")
    else:
        print_section("GPU lock")
        handle = try_acquire_gpu_lock(args.lock, args.lock_timeout_s)
        if handle is None:
            gpu_report = {
                "ran": False,
                "reason": f"lock wait exceeded {args.lock_timeout_s:.0f}s; CPU-only report",
            }
            print(gpu_report["reason"])
        else:
            raw_gpu: dict[str, Any] | None = None
            try:
                import torch

                if not torch.cuda.is_available():
                    gpu_report = {"ran": False, "reason": "CUDA not available"}
                    print("CUDA not available")
                else:
                    fit = sample[: args.probe_n] if len(sample) >= args.probe_n else sample
                    fit_hashes = {r["hash"] for r in fit}
                    rest = [r for r in sample if r["hash"] not in fit_hashes]
                    ev = rest[: args.probe_n]
                    raw_gpu = {
                        "ran": True,
                        "device": torch.cuda.get_device_name(0),
                        "probe_n_fit": len(fit),
                        "probe_n_eval": len(ev),
                        "models": gpu_random_head_and_features(
                            args.models_dir,
                            tokenizers,
                            demo,
                            fit,
                            ev,
                            args.max_length,
                        ),
                    }
            except Exception as exc:
                gpu_report = {"ran": False, "reason": f"{type(exc).__name__}: {exc}"}
                print(f"GPU probe failed: {gpu_report['reason']}")
                raw_gpu = None
            finally:
                release_gpu_lock(handle)
                print("released GPU lock")
            if raw_gpu is not None:
                raw_gpu["models"] = fit_frozen_linear_probes(raw_gpu["models"])
                gpu_report = raw_gpu

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "local_mlm": local_report,
        "remote_ner": remote_report,
        "demo_sentences": demo_rows,
        "fertility_extra": fertility,
        "suffix_token_splits": splits,
        "gpu": gpu_report,
        "hackathon_bio": HACKATHON_BIO,
        "recommendation": {
            "chop_mlm_head_and_finetune": True,
            "distill_public_ner_logits": False,
            "use_as_encoder_ft": [
                "exp0_xlm_roberta_base",
                "exp1_uztext_roberta",
                "exp2_mdeberta_v3_base",
            ],
            "ignore": [
                "jamshidahmadov/roberta-ner-uz",
                "risqaliyevds/xlm-roberta-large-ner",
            ],
        },
    }
    out_path = args.out_dir / "summary.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print_section("Artifacts")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

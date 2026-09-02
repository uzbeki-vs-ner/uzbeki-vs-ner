#!/usr/bin/env python3
# ruff: noqa: RUF001
"""EDA of the official Uzbek NER hackathon dataset vs public sources.

Computes class / span / script statistics on official JSONL, cheap ontology
stats on downloaded external datasets, and tokenizer fertility on local
BERT-family checkpoints. Does not train models.

Usage:
    uv run python scripts/eda_official.py
    uv run python scripts/eda_official.py --skip-tokenizers --skip-external
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from uzbek_ner.settings import get_settings  # noqa: E402

LATIN_RE = re.compile(r"[A-Za-zÀ-ÿ]")
CYR_RE = re.compile(r"[\u0400-\u04FF]")
WORD_RE = re.compile(r"\S+")

# Attached Uzbek case / possessive / plural endings seen in news + social text.
LATIN_SUFFIXES = (
    "lardagi",
    "laridan",
    "larining",
    "lariga",
    "larida",
    "laridan",
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
EDGE_PUNCT = set("«»\"'“”()[]{}#@.,;:!?…")

UZNER_MAP = {
    "PER": "NAME",
    "ORG": "ORG",
    "GPE": "GEO",
    "LOC": "GEO",
    "FAC": "GEO",  # locational facilities; mosques/churches are ORG in official
}
RISQ_MAP = {
    "PERSON": "NAME",
    "PER": "NAME",
    "ORG": "ORG",
    "LOC": "GEO",
    "GPE": "GEO",
    "FAC": "GEO",
    "FACILITY": "GEO",
}

LEGAL_V3_MAP = {"PER": "NAME", "ORG": "ORG", "LOC": "GEO"}
LEGAL_FULL_MAP = {
    "PER": "NAME",
    "ORG": "ORG",
    "LOC": "GEO",
    "COURT": "ORG",
    "BANK": "ORG",
}

TOKENIZER_ALIASES = {
    "exp0_xlm_roberta_base": "XLM-RoBERTa-base",
    "exp1_uztext_roberta": "uztext-3Gb-BPE-Roberta",
    "exp2_mdeberta_v3_base": "mDeBERTa-v3-base",
}


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


def summarize_numeric(values: list[int] | list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0, "min": 0, "p50": 0, "mean": 0, "p90": 0, "p95": 0, "p99": 0, "max": 0}
    return {
        "n": len(values),
        "min": float(min(values)),
        "p50": round(percentile(values, 0.50), 2),
        "mean": round(float(statistics.fmean(values)), 2),
        "p90": round(percentile(values, 0.90), 2),
        "p95": round(percentile(values, 0.95), 2),
        "p99": round(percentile(values, 0.99), 2),
        "max": float(max(values)),
    }


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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def iter_jsonl(path: Path, limit: int | None = None):
    n = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
            n += 1
            if limit is not None and n >= limit:
                return


def label_signature(labels: set[str]) -> str:
    if not labels:
        return "EMPTY"
    return "+".join(sorted(labels))


def analyze_official_split(name: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    n_docs = len(records)
    n_empty = 0
    class_counts: Counter[str] = Counter()
    signature_counts: Counter[str] = Counter()
    cooccur: Counter[str] = Counter()
    script_docs: Counter[str] = Counter()
    script_entities: Counter[str] = Counter()
    char_lens: list[int] = []
    word_lens: list[int] = []
    entity_char_lens: list[int] = []
    entity_word_lens: list[int] = []
    per_label_entity_chars: dict[str, list[int]] = defaultdict(list)
    surfaces: dict[str, Counter[str]] = defaultdict(Counter)
    surface_labels: dict[str, Counter[str]] = defaultdict(Counter)
    suffix_by_label: dict[str, Counter[str]] = defaultdict(Counter)
    n_with_attached_suffix = 0
    n_entities = 0

    overlap_pairs = 0
    nested_pairs = 0
    leading_ws = 0
    trailing_ws = 0
    edge_punct = 0
    invalid_offsets = 0
    duplicate_spans = 0
    empty_spans = 0

    for rec in records:
        text = rec.get("text") or ""
        entities = list(rec.get("entities") or [])
        char_lens.append(len(text))
        words = WORD_RE.findall(text)
        word_lens.append(len(words))
        script_docs[script_of(text)] += 1

        labels_in_doc: set[str] = set()
        spans: list[tuple[int, int, str]] = []
        seen_span: set[tuple[int, int, str]] = set()
        for ent in entities:
            label = ent.get("label")
            start = ent.get("start")
            end = ent.get("end")
            if label is None or not isinstance(start, int) or not isinstance(end, int):
                invalid_offsets += 1
                continue
            if not 0 <= start < end <= len(text):
                invalid_offsets += 1
                continue
            surface = text[start:end]
            if surface == "":
                empty_spans += 1
                continue
            key = (start, end, label)
            if key in seen_span:
                duplicate_spans += 1
                continue
            seen_span.add(key)
            spans.append((start, end, label))
            n_entities += 1
            class_counts[label] += 1
            labels_in_doc.add(label)
            entity_char_lens.append(len(surface))
            entity_word_lens.append(len(WORD_RE.findall(surface)))
            per_label_entity_chars[label].append(len(surface))
            surfaces[label][surface] += 1
            surface_labels[surface][label] += 1
            script_entities[script_of(surface)] += 1
            suffix = attached_suffix(surface)
            if suffix:
                n_with_attached_suffix += 1
                suffix_by_label[label][suffix] += 1
            if surface[:1].isspace():
                leading_ws += 1
            if surface[-1:].isspace():
                trailing_ws += 1
            if surface[:1] in EDGE_PUNCT or surface[-1:] in EDGE_PUNCT:
                edge_punct += 1

        if not labels_in_doc:
            n_empty += 1
        signature_counts[label_signature(labels_in_doc)] += 1
        for a in labels_in_doc:
            for b in labels_in_doc:
                if a < b:
                    cooccur[f"{a}+{b}"] += 1

        spans.sort()
        for i, (s1, e1, _l1) in enumerate(spans):
            for s2, e2, _l2 in spans[i + 1 :]:
                if s2 >= e1:
                    break
                overlap_pairs += 1
                if s1 <= s2 and e2 <= e1:
                    nested_pairs += 1

    ambiguity: list[dict[str, Any]] = []
    for surface, lab_counts in surface_labels.items():
        if len(lab_counts) > 1:
            ambiguity.append(
                {
                    "surface": surface,
                    "labels": dict(lab_counts),
                    "total": int(sum(lab_counts.values())),
                }
            )
    ambiguity.sort(key=lambda item: (-item["total"], -len(item["surface"])))

    agglutination_examples: list[dict[str, Any]] = []
    # Look for stem vs inflected pair inside the same class (Toshkent / Toshkentda).
    for label, counter in surfaces.items():
        stems: dict[str, int] = {}
        inflected: list[tuple[str, str, int]] = []
        for surface, count in counter.items():
            if " " in surface:
                continue
            suffix = attached_suffix(surface)
            if suffix:
                stem = surface[: -len(suffix)]
                inflected.append((stem, surface, count))
            else:
                stems[surface] = count
        for stem, surface, count in inflected:
            if stem in stems:
                agglutination_examples.append(
                    {
                        "label": label,
                        "stem": stem,
                        "stem_n": stems[stem],
                        "inflected": surface,
                        "inflected_n": count,
                    }
                )
    agglutination_examples.sort(key=lambda item: -(item["stem_n"] + item["inflected_n"]))

    top_surfaces = {label: counter.most_common(25) for label, counter in surfaces.items()}

    return {
        "split": name,
        "n_docs": n_docs,
        "n_docs_with_entities": n_docs - n_empty,
        "n_docs_empty": n_empty,
        "empty_doc_pct": round(100.0 * n_empty / n_docs, 2) if n_docs else 0,
        "n_entities": n_entities,
        "entities_per_doc_mean": round(n_entities / n_docs, 3) if n_docs else 0,
        "class_counts": dict(class_counts),
        "class_pct": {lab: round(100.0 * n / n_entities, 2) for lab, n in class_counts.items()}
        if n_entities
        else {},
        "label_signatures": dict(signature_counts),
        "label_cooccurrence_docs": dict(cooccur),
        "script_docs": dict(script_docs),
        "script_entities": dict(script_entities),
        "text_chars": summarize_numeric(char_lens),
        "text_words": summarize_numeric(word_lens),
        "n_docs_gt_300_words": sum(1 for n in word_lens if n > 300),
        "n_docs_gt_400_words": sum(1 for n in word_lens if n > 400),
        "entity_chars": summarize_numeric(entity_char_lens),
        "entity_words": summarize_numeric(entity_word_lens),
        "entity_chars_by_label": {
            lab: summarize_numeric(vals) for lab, vals in per_label_entity_chars.items()
        },
        "span_issues": {
            "overlapping_pairs": overlap_pairs,
            "nested_pairs": nested_pairs,
            "invalid_offsets": invalid_offsets,
            "duplicate_spans": duplicate_spans,
            "empty_spans": empty_spans,
            "leading_whitespace": leading_ws,
            "trailing_whitespace": trailing_ws,
            "edge_punctuation": edge_punct,
        },
        "n_entities_with_attached_suffix": n_with_attached_suffix,
        "attached_suffix_pct": round(100.0 * n_with_attached_suffix / n_entities, 2)
        if n_entities
        else 0,
        "suffix_counts_by_label": {
            lab: dict(counter.most_common(12)) for lab, counter in suffix_by_label.items()
        },
        "top_surfaces": dict(top_surfaces),
        "ambiguity_n_surfaces": len(ambiguity),
        "ambiguity_top": ambiguity[:40],
        "agglutination_pair_n": len(agglutination_examples),
        "agglutination_top": agglutination_examples[:40],
        "unique_surfaces": {lab: len(counter) for lab, counter in surfaces.items()},
    }


def official_surface_sets(records: list[dict[str, Any]]) -> dict[str, set[str]]:
    sets: dict[str, set[str]] = defaultdict(set)
    for rec in records:
        text = rec.get("text") or ""
        for ent in rec.get("entities") or []:
            start, end, label = ent["start"], ent["end"], ent["label"]
            if 0 <= start < end <= len(text):
                sets[label].add(text[start:end].casefold())
    return dict(sets)


def bio_type(tag: str) -> str | None:
    if not tag or tag in {"O", "o"}:
        return None
    parts = tag.replace("_", "-").split("-")
    if len(parts) == 1:
        return parts[0].upper()
    return parts[-1].upper()


def count_bio_tags(tags: list[Any], names: list[str] | None = None) -> Counter[str]:
    counts: Counter[str] = Counter()
    for tag in tags:
        if isinstance(tag, int) and names is not None and 0 <= tag < len(names):
            raw = names[tag]
        else:
            raw = str(tag)
        typ = bio_type(raw)
        if typ:
            counts[typ] += 1
    return counts


def map_counts(raw: Counter[str], mapping: dict[str, str]) -> dict[str, Any]:
    mapped: Counter[str] = Counter()
    rejected: Counter[str] = Counter()
    unmapped: Counter[str] = Counter()
    for lab, n in raw.items():
        if lab in mapping:
            mapped[mapping[lab]] += n
        else:
            rejected[lab] += n
            unmapped[lab] += n
    total = sum(raw.values())
    keep_n = sum(mapped.values())
    return {
        "raw_labels": dict(raw),
        "mapped_to_hackathon": dict(mapped),
        "rejected_or_extra": dict(rejected),
        "n_mentions_or_tokens": total,
        "keep_n": keep_n,
        "keep_pct": round(100.0 * keep_n / total, 2) if total else 0,
        "reject_pct": round(100.0 * (total - keep_n) / total, 2) if total else 0,
    }


def overlap_with_official(
    external_surfaces: dict[str, set[str]],
    official: dict[str, set[str]],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for lab in ("ORG", "NAME", "GEO"):
        a = official.get(lab, set())
        b = external_surfaces.get(lab, set())
        inter = a & b
        out[lab] = {
            "official_unique": len(a),
            "external_unique": len(b),
            "intersection": len(inter),
            "jaccard": round(len(inter) / len(a | b), 4) if (a or b) else 0,
            "recall_of_official": round(len(inter) / len(a), 4) if a else 0,
        }
    return out


def analyze_uzner(
    path: Path, official_sets: dict[str, set[str]], limit: int | None
) -> dict[str, Any]:
    labels: Counter[str] = Counter()
    n_docs = 0
    n_with_ent = 0
    scripts: Counter[str] = Counter()
    synthetic = 0
    surfaces: dict[str, set[str]] = defaultdict(set)
    suffix_in_span = 0
    n_ents = 0
    skipped = False
    for rec in iter_jsonl(path, limit=limit):
        n_docs += 1
        text = rec.get("text") or ""
        scripts[script_of(text)] += 1
        meta = rec.get("meta") or {}
        if meta.get("is_synthetic") or meta.get("source_tier") == "synthetic":
            synthetic += 1
        ents = rec.get("entities") or []
        if ents:
            n_with_ent += 1
        for ent in ents:
            lab = str(ent.get("label") or "")
            labels[lab] += 1
            n_ents += 1
            surf = ent.get("text") or (text[ent["start"] : ent["end"]] if "start" in ent else "")
            mapped = UZNER_MAP.get(lab)
            if mapped and surf:
                surfaces[mapped].add(str(surf).casefold())
            if surf and attached_suffix(str(surf)):
                suffix_in_span += 1
    mapped = map_counts(labels, UZNER_MAP)
    # FAC is a cautious GEO map; also report FAC separately.
    fac_n = labels.get("FAC", 0)
    return {
        "source": "uzner_100k",
        "path": str(path),
        "n_docs": n_docs,
        "n_docs_with_entities": n_with_ent,
        "n_synthetic_docs": synthetic,
        "script_docs": dict(scripts),
        "n_entities": n_ents,
        "entities_with_attached_suffix": suffix_in_span,
        "fac_n": fac_n,
        "ontology": mapped,
        "surface_overlap_official": overlap_with_official(surfaces, official_sets),
        "truncated": skipped or (limit is not None),
        "limit": limit,
        "notes": (
            "18-type BIOES. PER/ORG/GPE/LOC(+FAC) are the only mappable types. "
            "Guidelines put case suffixes OUTSIDE the span when tokenized separately "
            "(Toshkent + da), opposite of the hackathon rule (Toshkentda inside the span). "
            "FAC (airports, museums, stadiums) ≈ official GEO; mosques/churches are ORG "
            "in the official guide. PRODUCT/POSITION/NORP must not be mapped."
        ),
    }


def analyze_uznlp_gold(path: Path) -> dict[str, Any]:
    from datasets import load_from_disk

    ds = load_from_disk(str(path))["train"]
    tags: Counter[str] = Counter()
    n_tokens = len(ds)
    sentences: set[int] = set()
    for row in ds:
        sentences.add(int(row["Sentence"]))
        tag = str(row["NER_Tag"])
        typ = bio_type(tag)
        if typ:
            tags[typ] += 1
    mapping = {"PER": "NAME", "ORG": "ORG", "LOC": "GEO"}
    return {
        "source": "uznlp_uzbek_ner_gold",
        "n_tokens": n_tokens,
        "n_sentences": len(sentences),
        "ontology": map_counts(tags, mapping),
        "notes": (
            "Token BIO TSV (~4.2k sent). Keep PER/ORG/LOC only. Extra types "
            "(MISC/MONEY/TEMPORAL/...) are out of ontology."
        ),
    }


def analyze_risqaliyevds(
    path: Path, official_sets: dict[str, set[str]], max_docs: int
) -> dict[str, Any]:
    from datasets import load_from_disk

    ds = load_from_disk(str(path))["train"]
    n = min(len(ds), max_docs)
    labels: Counter[str] = Counter()
    n_docs = 0
    n_with = 0
    surfaces: dict[str, set[str]] = defaultdict(set)
    for i, row in enumerate(ds):
        if i >= n:
            break
        n_docs += 1
        ner = row.get("ner") or {}
        has = False
        if isinstance(ner, dict):
            for lab, mentions in ner.items():
                if not mentions:
                    continue
                has = True
                for mention in mentions:
                    labels[str(lab).upper()] += 1
                    mapped = RISQ_MAP.get(str(lab).upper())
                    if mapped and mention:
                        surfaces[mapped].add(str(mention).casefold())
        if has:
            n_with += 1
    return {
        "source": "risqaliyevds_uzbek_ner",
        "n_docs_seen": n_docs,
        "n_docs_total": len(ds),
        "n_docs_with_mentions": n_with,
        "ontology": map_counts(labels, RISQ_MAP),
        "surface_overlap_official": overlap_with_official(surfaces, official_sets),
        "notes": (
            "Mention lists, not char spans. Needs a string aligner before training. "
            "PERSON/PER→NAME, ORG→ORG, LOC/GPE/FAC→GEO; drop DATE/EVENT/MONEY/…"
        ),
    }


def analyze_ner_prepared(path: Path, max_docs: int) -> dict[str, Any]:
    from datasets import load_from_disk

    ds = load_from_disk(str(path))["train"]
    names = ds.features["ner_tags"].feature.names
    n = min(len(ds), max_docs)
    labels: Counter[str] = Counter()
    n_docs = 0
    for i, row in enumerate(ds):
        if i >= n:
            break
        n_docs += 1
        labels.update(count_bio_tags(row["ner_tags"], names))
    mapping = {
        "PERSON": "NAME",
        "ORG": "ORG",
        "GPE": "GEO",
        "LOC": "GEO",
    }
    return {
        "source": "ner_prepared_uzbek",
        "n_docs_seen": n_docs,
        "n_docs_total": len(ds),
        "label_names": names,
        "ontology": map_counts(labels, mapping),
        "notes": (
            "Token BIO derived from risqaliyevds. ClassLabel only has B-* tags "
            "(no I-*), so this is a noisy conversion — prefer the mention-list source "
            "plus a proper aligner, or treat as weak silver."
        ),
    }


def analyze_wikiann(path: Path) -> dict[str, Any]:
    from datasets import load_from_disk

    ds = load_from_disk(str(path))
    labels: Counter[str] = Counter()
    n_docs = 0
    names = None
    for split in ds:
        subset = ds[split]
        names = subset.features["ner_tags"].feature.names
        for row in subset:
            n_docs += 1
            labels.update(count_bio_tags(row["ner_tags"], names))
    mapping = {"PER": "NAME", "ORG": "ORG", "LOC": "GEO"}
    return {
        "source": "wikiann_uz",
        "n_docs": n_docs,
        "splits": {split: len(ds[split]) for split in ds},
        "ontology": map_counts(labels, mapping),
        "notes": "Weak Wikipedia NER, 1k x 3. Ontology maps cleanly but quality is low.",
    }


def analyze_rubai(path: Path, max_docs: int) -> dict[str, Any]:
    from datasets import load_from_disk

    ds = load_from_disk(str(path))["train"]
    n = min(len(ds), max_docs)
    types: Counter[str] = Counter()
    n_docs = 0
    n_with_name = 0
    domains: Counter[str] = Counter()
    for i, row in enumerate(ds):
        if i >= n:
            break
        n_docs += 1
        domains[str(row.get("domain") or "unknown")] += 1
        row_types = [str(t).upper() for t in (row.get("types") or row.get("denorm_types") or [])]
        types.update(row_types)
        if any(t in {"NAME", "PER", "PERSON", "FIO"} for t in row_types):
            n_with_name += 1
    name_like = sum(n for lab, n in types.items() if lab in {"NAME", "PER", "PERSON", "FIO"})
    total = sum(types.values()) or 1
    return {
        "source": "rubai_ner_150k_personal",
        "n_docs_seen": n_docs,
        "n_docs_total": len(ds),
        "n_docs_with_name_like": n_with_name,
        "type_counts": dict(types.most_common(30)),
        "name_like_pct_of_type_mentions": round(100.0 * name_like / total, 2),
        "domain_top": dict(domains.most_common(8)),
        "notes": (
            "Synthetic PII. Useful only as NAME (and maybe ORG for orgs-as-PII) silver. "
            "Not a GEO source. High domain/span-policy shift risk."
        ),
    }


def analyze_medner(path: Path) -> dict[str, Any]:
    from datasets import load_from_disk

    ds = load_from_disk(str(path))["train"]
    tags: Counter[str] = Counter()
    n_tokens = len(ds)
    docs: set[str] = set()
    for row in ds:
        docs.add(str(row.get("doc_id")))
        typ = bio_type(str(row.get("tag") or "O"))
        if typ:
            tags[typ] += 1
    return {
        "source": "uz_medner",
        "n_tokens": n_tokens,
        "n_docs": len(docs),
        "medical_labels": dict(tags),
        "keep_pct": 0.0,
        "notes": "Medical NER. No ORG/NAME/GEO. Reject for this hackathon.",
    }


def analyze_legal_v3(path: Path, official_sets: dict[str, set[str]]) -> dict[str, Any]:
    labels: Counter[str] = Counter()
    n_docs = 0
    n_with = 0
    suffix_in = 0
    suffix_excluded_examples: list[str] = []
    surfaces: dict[str, set[str]] = defaultdict(set)
    for rec in iter_jsonl(path):
        n_docs += 1
        ents = rec.get("entities") or []
        if ents:
            n_with += 1
        text = rec.get("text") or ""
        for ent in ents:
            lab = str(ent.get("label") or "")
            labels[lab] += 1
            surf = ent.get("text") or text[ent["start"] : ent["end"]]
            mapped = LEGAL_V3_MAP.get(lab)
            if mapped and surf:
                surfaces[mapped].add(str(surf).casefold())
            if surf and attached_suffix(str(surf)):
                suffix_in += 1
            # Detect Toshkentda-style: span is stem, next chars are a suffix.
            start, end = ent.get("start"), ent.get("end")
            if (
                isinstance(start, int)
                and isinstance(end, int)
                and end < len(text)
                and attached_suffix((surf or "") + text[end : end + 4])
                and not attached_suffix(str(surf))
                and len(suffix_excluded_examples) < 8
            ):
                suffix_excluded_examples.append(f"{surf}|{text[start : min(end + 6, len(text))]}")
    return {
        "source": "uzlegalner_v3",
        "n_docs": n_docs,
        "n_docs_with_entities": n_with,
        "ontology": map_counts(labels, LEGAL_V3_MAP),
        "entities_with_attached_suffix": suffix_in,
        "suffix_excluded_examples": suffix_excluded_examples,
        "surface_overlap_official": overlap_with_official(surfaces, official_sets),
        "notes": (
            "Legal contracts, 7 types. Map PER/ORG/LOC only; drop POSITION/DATE/MONEY/DOCNO. "
            "Example in the data dictionary labels Toshkentda as LOC='Toshkent' — suffix "
            "excluded, which is exact-span poison for the official policy."
        ),
    }


def analyze_legal_full(path: Path, max_rows: int) -> dict[str, Any]:
    labels: Counter[str] = Counter()
    gold: Counter[str] = Counter()
    n = 0
    missing_span = 0
    for rec in iter_jsonl(path, limit=max_rows):
        n += 1
        lab = str(rec.get("Label") or rec.get("label") or "")
        if lab:
            labels[lab] += 1
        status = str(rec.get("Gold_Status") or rec.get("gold_status") or "")
        gold[status] += 1
        start, end = rec.get("Start_Char"), rec.get("End_Char")
        if start in ("", None) or end in ("", None):
            missing_span += 1
    return {
        "source": "uzbek_legal_ner_full",
        "file": path.name,
        "n_rows_seen": n,
        "ontology": map_counts(labels, LEGAL_FULL_MAP),
        "gold_status": dict(gold),
        "missing_char_span": missing_span,
        "notes": (
            "One row ≈ one (often multi-mention) candidate, not a clean sentence NER set. "
            "Many silver/unverified rows and missing offsets. BANK/COURT→ORG is the only "
            "extra mapping that matches the official ORG definition. Do not mix into dev."
        ),
    }


def load_local_tokenizer(model_dir: Path):
    from transformers import AutoTokenizer, DebertaV2Tokenizer

    alias = model_dir.name
    try:
        return AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, use_fast=True)
    except Exception:
        pass
    if alias == "exp2_mdeberta_v3_base" or (model_dir / "spm.model").exists():
        spm = model_dir / "spm.model"
        return DebertaV2Tokenizer(vocab_file=str(spm), do_lower_case=False)
    return AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, use_fast=False)


def fertility_for_tokenizer(tokenizer, texts: list[str], entities: list[str]) -> dict[str, Any]:
    tok_per_word: list[float] = []
    tok_per_char: list[float] = []
    n_over_512 = 0
    n_over_384 = 0
    special = set(getattr(tokenizer, "all_special_ids", []) or [])
    for text in texts:
        words = max(len(WORD_RE.findall(text)), 1)
        chars = max(len(text), 1)
        ids = tokenizer.encode(text, add_special_tokens=False)
        n_tok = len([i for i in ids if i not in special]) if special else len(ids)
        n_tok = max(n_tok, 1)
        tok_per_word.append(n_tok / words)
        tok_per_char.append(n_tok / chars)
        # special tokens add ~2
        if n_tok + 2 > 512:
            n_over_512 += 1
        if n_tok + 2 > 384:
            n_over_384 += 1

    ent_fert: list[float] = []
    for surface in entities:
        words = max(len(WORD_RE.findall(surface)), 1)
        ids = tokenizer.encode(surface, add_special_tokens=False)
        n_tok = max(len(ids), 1)
        ent_fert.append(n_tok / words)

    vocab = getattr(tokenizer, "vocab_size", None) or len(tokenizer)
    return {
        "vocab_size": int(vocab),
        "n_texts": len(texts),
        "mean_tokens_per_word": round(statistics.fmean(tok_per_word), 3),
        "median_tokens_per_word": round(percentile(tok_per_word, 0.5), 3),
        "mean_tokens_per_char": round(statistics.fmean(tok_per_char), 4),
        "mean_tokens_per_entity_word": round(statistics.fmean(ent_fert), 3) if ent_fert else None,
        "n_texts_over_384_tokens": n_over_384,
        "n_texts_over_512_tokens": n_over_512,
        "by_script": {},
    }


def fertility_by_script(tokenizer, records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for rec in records:
        grouped[script_of(rec["text"])].append(rec["text"])
    out: dict[str, Any] = {}
    special = set(getattr(tokenizer, "all_special_ids", []) or [])
    for script, texts in grouped.items():
        ratios: list[float] = []
        for text in texts:
            words = max(len(WORD_RE.findall(text)), 1)
            ids = tokenizer.encode(text, add_special_tokens=False)
            n_tok = len([i for i in ids if i not in special]) if special else len(ids)
            ratios.append(max(n_tok, 1) / words)
        out[script] = {
            "n": len(texts),
            "mean_tokens_per_word": round(statistics.fmean(ratios), 3),
        }
    return out


def analyze_tokenizers(models_dir: Path, sample_records: list[dict[str, Any]]) -> dict[str, Any]:
    texts = [r["text"] for r in sample_records]
    entity_surfaces: list[str] = []
    for rec in sample_records:
        text = rec["text"]
        for ent in rec.get("entities") or []:
            entity_surfaces.append(text[ent["start"] : ent["end"]])
    results: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for alias, pretty in TOKENIZER_ALIASES.items():
        path = models_dir / alias
        if not path.exists():
            errors[alias] = f"missing {path}"
            continue
        try:
            tok = load_local_tokenizer(path)
            stats = fertility_for_tokenizer(tok, texts, entity_surfaces[:800])
            stats["by_script"] = fertility_by_script(tok, sample_records)
            stats["display_name"] = pretty
            results[alias] = stats
        except Exception as exc:
            errors[alias] = f"{type(exc).__name__}: {exc}"
    return {"models": results, "errors": errors, "n_sample_texts": len(texts)}


def maybe_plot(out_dir: Path, official: dict[str, Any]) -> list[str]:
    """Write tiny optional plots; skip silently if matplotlib is absent."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    written: list[str] = []
    train = official["train"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    labels = ["ORG", "NAME", "GEO"]
    axes[0].bar(labels, [train["class_counts"].get(lab, 0) for lab in labels], color="#3b6ea5")
    axes[0].set_title("Train entity counts")
    scripts = train["script_docs"]
    axes[1].bar(list(scripts.keys()), list(scripts.values()), color="#c47b2b")
    axes[1].set_title("Train docs by script")
    fig.tight_layout()
    path = out_dir / "official_overview.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    written.append(str(path))
    return written


def print_section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def fmt_counter(d: dict, limit: int = 12) -> str:
    items = sorted(
        d.items(),
        key=lambda kv: (-kv[1] if isinstance(kv[1], int | float) else 0, str(kv[0])),
    )
    parts = [f"{k}={v}" for k, v in items[:limit]]
    return ", ".join(parts)


def run_external(
    external_root: Path,
    official_sets: dict[str, set[str]],
    uzner_limit: int | None,
    hf_max: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    reports: list[dict[str, Any]] = []
    skipped: list[str] = []

    uzner_train = external_root / "zenodo" / "uzner_100k" / "extracted" / "uzner_train_bioes.jsonl"
    if uzner_train.exists():
        print("[external] uzner_100k train jsonl …")
        reports.append(analyze_uzner(uzner_train, official_sets, uzner_limit))
        for extra in (
            "uzner_dev_bioes.jsonl",
            "uzner_test_bioes.jsonl",
            "uzner_hard_eval_bioes.jsonl",
        ):
            p = uzner_train.parent / extra
            if p.exists():
                skipped.append(f"{extra} (same schema as train; not fully counted)")
        skipped.append("uzner_*.conll (duplicate of jsonl)")
        skipped.append("uzner_gold_candidate_bioes.* (candidate dump, not the train split)")
    else:
        skipped.append("uzner_100k train jsonl missing")

    hf_root = external_root / "hf"
    hf_jobs = [
        ("uznlp_uzbek_ner_gold", lambda p: analyze_uznlp_gold(p)),
        ("wikiann_uz", lambda p: analyze_wikiann(p)),
        ("uz_medner", lambda p: analyze_medner(p)),
        ("risqaliyevds_uzbek_ner", lambda p: analyze_risqaliyevds(p, official_sets, hf_max)),
        ("ner_prepared_uzbek", lambda p: analyze_ner_prepared(p, hf_max)),
        ("rubai_ner_150k_personal", lambda p: analyze_rubai(p, hf_max)),
    ]
    for key, fn in hf_jobs:
        path = hf_root / key / "hf_dataset"
        if not path.exists():
            skipped.append(f"{key} not on disk")
            continue
        print(f"[external] {key} …")
        try:
            reports.append(fn(path))
        except Exception as exc:
            skipped.append(f"{key}: {type(exc).__name__}: {exc}")

    legal_v3 = (
        external_root
        / "zenodo"
        / "uzlegalner_v3"
        / "extracted"
        / "UzLegalNER_v3_fixed_zenodo"
        / "data"
        / "uz_legal_ner_master_v3_fixed.jsonl"
    )
    if legal_v3.exists():
        print("[external] uzlegalner_v3 …")
        reports.append(analyze_legal_v3(legal_v3, official_sets))
        skipped.append("uzlegalner_v3 CoNLL splits (same sentences as jsonl)")
    else:
        skipped.append("uzlegalner_v3 jsonl missing")

    legal_full = (
        external_root
        / "zenodo"
        / "uzbek_legal_ner_full"
        / "extracted"
        / "Uzbek_Legal_NER_Core_Gold_v1.jsonl"
    )
    goldready = legal_full.parent / "Uzbek_Legal_NER_GoldReady_Augmented_v3_final.jsonl"
    if legal_full.exists():
        print("[external] uzbek_legal_ner_full core gold …")
        reports.append(analyze_legal_full(legal_full, max_rows=50_000))
        skipped.append("Uzbek_Legal_NER_Extended_Augmented_v1.* (duplicate formats)")
        skipped.append("Uzbek_Legal_NER_*.json/*.csv (same content as jsonl)")
        if goldready.exists():
            # GoldReady is 12MB / ~same schema; count labels only, cap rows.
            print("[external] uzbek_legal_ner_full goldready (label counts, capped) …")
            reports.append(analyze_legal_full(goldready, max_rows=20_000))
    else:
        skipped.append("uzbek_legal_ner_full jsonl missing")

    return reports, skipped


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=settings.official_train)
    parser.add_argument("--dev", type=Path, default=settings.official_dev)
    parser.add_argument("--external-root", type=Path, default=settings.data_external)
    parser.add_argument("--models-dir", type=Path, default=settings.models / "pretrained")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs" / "eda")
    parser.add_argument("--sample-n", type=int, default=200, help="Texts for tokenizer fertility")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-external", action="store_true")
    parser.add_argument("--skip-tokenizers", action="store_true")
    parser.add_argument(
        "--uzner-limit",
        type=int,
        default=None,
        help="Cap UzNER-100k train lines (default: all ~100k)",
    )
    parser.add_argument(
        "--hf-max",
        type=int,
        default=25_000,
        help="Max HF rows to scan per dataset (rubai is 142k; default scans 25k)",
    )
    args = parser.parse_args()

    print_section("Official JSONL")
    print(f"train: {args.train}")
    print(f"dev:   {args.dev}")
    train = load_jsonl(args.train)
    dev = load_jsonl(args.dev)
    print(f"loaded train={len(train)} dev={len(dev)}")

    train_stats = analyze_official_split("train", train)
    dev_stats = analyze_official_split("dev", dev)
    official_sets = official_surface_sets(train + dev)

    for stats in (train_stats, dev_stats):
        print_section(f"Official {stats['split']}")
        print(
            f"docs={stats['n_docs']} with_ent={stats['n_docs_with_entities']} "
            f"empty={stats['n_docs_empty']} ({stats['empty_doc_pct']}%)"
        )
        print(f"entities={stats['n_entities']}  by class: {fmt_counter(stats['class_counts'])}")
        print(f"signatures: {fmt_counter(stats['label_signatures'])}")
        print(f"cooccur docs: {fmt_counter(stats['label_cooccurrence_docs'])}")
        print(f"script docs: {fmt_counter(stats['script_docs'])}")
        print(f"script entities: {fmt_counter(stats['script_entities'])}")
        print(f"text chars: {stats['text_chars']}")
        print(f"text words: {stats['text_words']}")
        print(
            f"long docs: >300 words={stats['n_docs_gt_300_words']} "
            f">400 words={stats['n_docs_gt_400_words']}"
        )
        print(f"entity chars: {stats['entity_chars']}")
        print(f"span issues: {stats['span_issues']}")
        print(
            f"attached suffix entities: {stats['n_entities_with_attached_suffix']} "
            f"({stats['attached_suffix_pct']}%)"
        )
        print(f"suffixes by label: {stats['suffix_counts_by_label']}")
        print(f"unique surfaces: {stats['unique_surfaces']}")
        print(f"ambiguity surfaces (same string, >1 label): {stats['ambiguity_n_surfaces']}")
        for item in stats["ambiguity_top"][:12]:
            print(f"  {item['surface']!r}: {item['labels']} n={item['total']}")
        print(f"agglutination stem/inflected pairs: {stats['agglutination_pair_n']}")
        for item in stats["agglutination_top"][:10]:
            print(
                f"  [{item['label']}] {item['stem']} ({item['stem_n']}) vs "
                f"{item['inflected']} ({item['inflected_n']})"
            )
        print("top surfaces:")
        for lab, rows in stats["top_surfaces"].items():
            preview = ", ".join(f"{s} x{c}" for s, c in rows[:8])
            print(f"  {lab}: {preview}")

    external_reports: list[dict[str, Any]] = []
    skipped: list[str] = []
    if not args.skip_external:
        print_section("External datasets")
        external_reports, skipped = run_external(
            args.external_root, official_sets, args.uzner_limit, args.hf_max
        )
        for rep in external_reports:
            print(f"\n-- {rep.get('source')} --")
            for key, value in rep.items():
                if key in {"surface_overlap_official", "ontology", "type_counts", "medical_labels"}:
                    print(f"  {key}: {json.dumps(value, ensure_ascii=False)}")
                elif key != "notes":
                    print(f"  {key}: {value}")
            if "notes" in rep:
                print(f"  notes: {rep['notes']}")
        print("\nSkipped / not fully parsed:")
        for item in skipped:
            print(f"  - {item}")

    tok_report: dict[str, Any] = {}
    if not args.skip_tokenizers:
        print_section("Tokenizer fertility")
        rng = random.Random(args.seed)
        sample = train[:] if len(train) <= args.sample_n else rng.sample(train, args.sample_n)
        tok_report = analyze_tokenizers(args.models_dir, sample)
        for alias, stats in tok_report.get("models", {}).items():
            print(f"{alias} ({stats.get('display_name')}): {json.dumps(stats, ensure_ascii=False)}")
        if tok_report.get("errors"):
            print(f"errors: {tok_report['errors']}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "official": {"train": train_stats, "dev": dev_stats},
        "external": external_reports,
        "skipped": skipped,
        "tokenizers": tok_report,
    }
    # Drop bulky top lists from the on-disk JSON? Keep them — they are small.
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print_section("Artifacts")
    print(f"wrote {summary_path}")
    plots = maybe_plot(args.out_dir, summary["official"])
    for p in plots:
        print(f"wrote {p}")


if __name__ == "__main__":
    main()

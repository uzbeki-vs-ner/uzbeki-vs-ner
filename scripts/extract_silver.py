#!/usr/bin/env python3
# ruff: noqa: RUF001
"""Extract mapped silver NER records from downloaded external Uzbek datasets.

Converts keep-list sources into the official hackathon schema
``{hash, text, entities:[{label, start, end}]}`` with exclusive end offsets
and official suffix-inclusive span policy. CPU only — does not use CUDA.

Usage:
    uv run python scripts/extract_silver.py
    uv run python scripts/extract_silver.py --workers 20
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Another agent may be using the GPU; never allocate CUDA from this script.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from uzbek_ner.settings import get_settings  # noqa: E402

JsonDict = dict[str, Any]
Span = tuple[int, int, str]

LABELS = ("ORG", "NAME", "GEO")
LABEL_PRIORITY = {"NAME": 0, "ORG": 1, "GEO": 2}

# Core types only. FAC is mapped for risqaliyevds (EDA keep table) but NOT for
# UzNER-100k (mosque/church FAC vs official ORG conflict; Mix B says optional).
RISQ_MAP = {
    "PERSON": "NAME",
    "PER": "NAME",
    "ORG": "ORG",
    "LOC": "GEO",
    "GPE": "GEO",
    "FAC": "GEO",
    "FACILITY": "GEO",
}
UZNER_MAP = {
    "PER": "NAME",
    "ORG": "ORG",
    "GPE": "GEO",
    "LOC": "GEO",
}
UZNLP_MAP = {
    "PER": "NAME",
    "ORG": "ORG",
    "LOC": "GEO",
}

# Longest-first Uzbek case / plural+case endings (Latin + Cyrillic).
# Short stems like -da/-ga/-ni are only accepted at a following word boundary.
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
SUFFIXES: tuple[str, ...] = tuple(sorted(set(LATIN_SUFFIXES + CYR_SUFFIXES), key=len, reverse=True))
APOSTROPHES = ("'", "ʼ", "ʻ", "‘", "’", "`", "´")
APOSTROPHE_PREFIXES: tuple[str, ...] = ("", *APOSTROPHES)
QUOTE_OPENERS = frozenset('«»"“”()[]{}')
REJECTED_SOURCES = (
    "uz_medner",
    "rubai_ner_150k_personal",
    "uzlegalner_v3",
    "uzbek_legal_ner_full",
    "ner_prepared_uzbek",
    "wikiann_uz",
)


def silver_hash(source: str, native_id: str, text: str) -> str:
    payload = f"{source}\n{native_id}\n{text}".encode()
    return "sv" + hashlib.sha1(payload).hexdigest()


def fold_apostrophes(text: str) -> str:
    out = text
    for mark in APOSTROPHES:
        out = out.replace(mark, "'")
    return out


def normalize_text(text: str) -> str:
    return " ".join(fold_apostrophes(text).casefold().split())


@dataclass
class Stats:
    source: str
    docs_in: int = 0
    docs_real_kept_in: int = 0
    docs_synthetic_dropped: int = 0
    docs_with_keep_mentions: int = 0
    docs_kept: int = 0
    docs_dropped_empty: int = 0
    docs_dropped_official: int = 0
    docs_dropped_cross_source: int = 0
    mentions_keep_type: int = 0
    mentions_other_type: int = 0
    mentions_unfound: int = 0
    mentions_type_conflict: int = 0
    spans_aligned: int = 0
    spans_overlap_dropped: int = 0
    entities_emitted: int = 0
    entities_by_label: dict[str, int] = field(default_factory=dict)
    other_type_counts: dict[str, int] = field(default_factory=dict)
    suffix_extended: int = 0
    suffix_skipped_space: int = 0
    suffix_skipped_punct_or_quote: int = 0
    suffix_by_form: dict[str, int] = field(default_factory=dict)

    def add_label(self, label: str, n: int = 1) -> None:
        self.entities_by_label[label] = self.entities_by_label.get(label, 0) + n

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "docs_in": self.docs_in,
            "docs_real_kept_in": self.docs_real_kept_in,
            "docs_synthetic_dropped": self.docs_synthetic_dropped,
            "docs_with_keep_mentions": self.docs_with_keep_mentions,
            "docs_kept": self.docs_kept,
            "docs_dropped_empty": self.docs_dropped_empty,
            "docs_dropped_official": self.docs_dropped_official,
            "docs_dropped_cross_source": self.docs_dropped_cross_source,
            "mentions_keep_type": self.mentions_keep_type,
            "mentions_other_type": self.mentions_other_type,
            "mentions_unfound": self.mentions_unfound,
            "mentions_type_conflict": self.mentions_type_conflict,
            "spans_aligned": self.spans_aligned,
            "spans_overlap_dropped": self.spans_overlap_dropped,
            "entities_emitted": self.entities_emitted,
            "entities_by_label": dict(self.entities_by_label),
            "other_type_counts": dict(
                sorted(self.other_type_counts.items(), key=lambda kv: (-kv[1], kv[0]))
            ),
            "suffix_extended": self.suffix_extended,
            "suffix_skipped_space": self.suffix_skipped_space,
            "suffix_skipped_punct_or_quote": self.suffix_skipped_punct_or_quote,
            "suffix_by_form": dict(
                sorted(self.suffix_by_form.items(), key=lambda kv: (-kv[1], kv[0]))
            ),
        }


def merge_stats(dst: Stats, src: Stats) -> None:
    for name in (
        "docs_in",
        "docs_real_kept_in",
        "docs_synthetic_dropped",
        "docs_with_keep_mentions",
        "docs_kept",
        "docs_dropped_empty",
        "docs_dropped_official",
        "docs_dropped_cross_source",
        "mentions_keep_type",
        "mentions_other_type",
        "mentions_unfound",
        "mentions_type_conflict",
        "spans_aligned",
        "spans_overlap_dropped",
        "entities_emitted",
        "suffix_extended",
        "suffix_skipped_space",
        "suffix_skipped_punct_or_quote",
    ):
        setattr(dst, name, getattr(dst, name) + getattr(src, name))
    for key, value in src.entities_by_label.items():
        dst.entities_by_label[key] = dst.entities_by_label.get(key, 0) + value
    for key, value in src.other_type_counts.items():
        dst.other_type_counts[key] = dst.other_type_counts.get(key, 0) + value
    for key, value in src.suffix_by_form.items():
        dst.suffix_by_form[key] = dst.suffix_by_form.get(key, 0) + value


def stats_from_dict(payload: dict[str, Any]) -> Stats:
    stats = Stats(source=str(payload["source"]))
    for name, value in payload.items():
        if name == "source":
            continue
        setattr(stats, name, value)
    return stats


def match_attached_suffix(rest: str) -> str | None:
    """Return the original-cased suffix slice if `rest` starts with a known ending."""

    if not rest:
        return None
    rest_cf = rest.casefold()
    best: str | None = None
    for suffix in SUFFIXES:
        for prefix in APOSTROPHE_PREFIXES:
            candidate = prefix + suffix
            n = len(candidate)
            if n > len(rest) or not rest_cf.startswith(candidate.casefold()):
                continue
            after = rest[n:]
            if after and after[0].isalpha():
                continue
            if best is None or n > len(best):
                best = rest[:n]
    return best


def rewrite_span_end(text: str, start: int, end: int) -> tuple[int, str | None, str]:
    """Extend `end` when an official-style case ending is attached.

    Conservative: do not extend across whitespace, after closing quotes, or
    when the current span already ends in punctuation (``Abror J.`` + ``ga``).
    """

    if not 0 <= start < end <= len(text):
        return end, None, "invalid"
    rest = text[end:]
    if not rest:
        return end, None, "eos"
    if rest[0].isspace():
        return end, None, "space"
    if rest[0] in QUOTE_OPENERS:
        return end, None, "quote"
    last = text[end - 1]
    if not (last.isalnum() or last in APOSTROPHES or last in "-‑"):
        return end, None, "punct"
    matched = match_attached_suffix(rest)
    if not matched:
        return end, None, "none"
    return end + len(matched), matched, "extended"


def resolve_flat_spans(spans: list[Span]) -> tuple[list[JsonDict], int]:
    """Keep a non-overlapping subset; longer spans win (official is flat)."""

    ordered = sorted(
        spans,
        key=lambda item: (
            -(item[1] - item[0]),
            item[0],
            LABEL_PRIORITY.get(item[2], 9),
            item[2],
        ),
    )
    accepted: list[Span] = []
    dropped = 0
    for start, end, label in ordered:
        if not 0 <= start < end or label not in LABEL_PRIORITY:
            dropped += 1
            continue
        if any(start < other_end and other_start < end for other_start, other_end, _ in accepted):
            dropped += 1
            continue
        accepted.append((start, end, label))
    accepted.sort(key=lambda item: (item[0], item[1], item[2]))
    entities = [{"label": lab, "start": start, "end": end} for start, end, lab in accepted]
    return entities, dropped


def find_word_starts(text: str, needle: str) -> list[int]:
    if not needle:
        return []
    hits: list[int] = []
    start = 0
    while True:
        index = text.find(needle, start)
        if index < 0:
            break
        if index == 0 or not text[index - 1].isalpha():
            hits.append(index)
        start = index + 1
    return hits


def locate_mention(text: str, mention: str) -> list[int]:
    """Word-start hits; apostrophe-fold then case-insensitive fallback."""

    if not mention or not text:
        return []
    hits = find_word_starts(text, mention)
    if hits:
        return hits
    folded_text = fold_apostrophes(text)
    folded_mention = fold_apostrophes(mention)
    if folded_mention != mention or folded_text != text:
        folded_hits = find_word_starts(folded_text, folded_mention)
        if folded_hits and len(folded_text) == len(text):
            return folded_hits
    mention_cf = mention.casefold()
    n = len(mention)
    hits = []
    limit = len(text) - n
    index = 0
    while index <= limit:
        window = text[index : index + n]
        if window.casefold() == mention_cf and (index == 0 or not text[index - 1].isalpha()):
            hits.append(index)
        index += 1
    return hits


def apply_suffix(text: str, start: int, end: int, stats: Stats) -> int:
    new_end, matched, reason = rewrite_span_end(text, start, end)
    if reason == "extended" and matched is not None:
        stats.suffix_extended += 1
        stats.suffix_by_form[matched.casefold()] = (
            stats.suffix_by_form.get(matched.casefold(), 0) + 1
        )
        return new_end
    if reason == "space":
        stats.suffix_skipped_space += 1
    elif reason in {"quote", "punct"}:
        stats.suffix_skipped_punct_or_quote += 1
    return end


def finalize_record(
    source: str,
    native_id: str,
    text: str,
    spans: list[Span],
    stats: Stats,
) -> JsonDict | None:
    rewritten: list[Span] = []
    for start, end, label in spans:
        end = apply_suffix(text, start, end, stats)
        if 0 <= start < end <= len(text) and text[start:end].strip() == text[start:end]:
            rewritten.append((start, end, label))
    entities, dropped = resolve_flat_spans(rewritten)
    stats.spans_overlap_dropped += dropped
    if not entities:
        stats.docs_dropped_empty += 1
        return None
    stats.docs_kept += 1
    stats.entities_emitted += len(entities)
    for ent in entities:
        stats.add_label(str(ent["label"]))
    return {
        "hash": silver_hash(source, native_id, text),
        "text": text,
        "entities": entities,
    }


def _mention_list(raw: Any) -> list[str]:
    if raw is None or isinstance(raw, float):
        return []
    if isinstance(raw, str):
        return [raw] if raw else []
    if isinstance(raw, dict):
        return []
    try:
        items = list(raw)
    except TypeError:
        return []
    out: list[str] = []
    for item in items:
        if item is None or isinstance(item, float):
            continue
        text = str(item)
        if text and text != "nan":
            out.append(text)
    return out


def convert_risq_row(row: JsonDict) -> tuple[JsonDict | None, Stats]:
    stats = Stats(source="risqaliyevds_uzbek_ner")
    stats.docs_in = 1
    text = row.get("text") or ""
    ner = row.get("ner") or {}
    mention_to_labels: dict[str, set[str]] = defaultdict(set)
    if isinstance(ner, dict):
        for raw_label, mentions in ner.items():
            mapped = RISQ_MAP.get(str(raw_label).upper())
            mention_list = _mention_list(mentions)
            if not mention_list:
                continue
            if mapped is None:
                n = len(mention_list)
                stats.mentions_other_type += n
                stats.other_type_counts[str(raw_label).upper()] = (
                    stats.other_type_counts.get(str(raw_label).upper(), 0) + n
                )
                continue
            for mention in mention_list:
                stats.mentions_keep_type += 1
                mention_to_labels[str(mention)].add(mapped)
    if not mention_to_labels:
        stats.docs_dropped_empty += 1
        return None, stats
    stats.docs_with_keep_mentions = 1
    spans: list[Span] = []
    for mention, labels in mention_to_labels.items():
        if len(labels) != 1:
            stats.mentions_type_conflict += 1
            continue
        label = next(iter(labels))
        hits = locate_mention(text, mention)
        if not hits:
            stats.mentions_unfound += 1
            continue
        width = len(mention)
        for start in hits:
            end = start + width
            if 0 <= start < end <= len(text):
                spans.append((start, end, label))
                stats.spans_aligned += 1
    native = str(row.get("id") or hashlib.sha1(text.encode()).hexdigest()[:16])
    return finalize_record("risqaliyevds_uzbek_ner", native, text, spans, stats), stats


def _risq_chunk(rows: list[JsonDict]) -> dict[str, Any]:
    records: list[JsonDict] = []
    stats = Stats(source="risqaliyevds_uzbek_ner")
    for row in rows:
        record, row_stats = convert_risq_row(row)
        merge_stats(stats, row_stats)
        if record is not None:
            records.append(record)
    return {"records": records, "stats": stats.as_dict()}


def is_uzner_synthetic(meta: JsonDict) -> bool:
    if meta.get("is_synthetic") is True:
        return True
    return str(meta.get("source_tier") or "").lower() == "synthetic"


def convert_uzner_row(row: JsonDict) -> tuple[JsonDict | None, Stats]:
    stats = Stats(source="uzner_100k_real")
    stats.docs_in = 1
    meta = row.get("meta") or {}
    if not isinstance(meta, dict):
        meta = {}
    if is_uzner_synthetic(meta):
        stats.docs_synthetic_dropped = 1
        return None, stats
    stats.docs_real_kept_in = 1
    text = row.get("text") or ""
    raw_entities = row.get("entities") or []
    spans: list[Span] = []
    had_keep = False
    for ent in raw_entities:
        raw_label = str(ent.get("label") or "").upper()
        mapped = UZNER_MAP.get(raw_label)
        if mapped is None:
            stats.mentions_other_type += 1
            stats.other_type_counts[raw_label] = stats.other_type_counts.get(raw_label, 0) + 1
            continue
        stats.mentions_keep_type += 1
        had_keep = True
        start = ent.get("start")
        end = ent.get("end")
        if not isinstance(start, int) or not isinstance(end, int):
            stats.mentions_unfound += 1
            continue
        if not 0 <= start < end <= len(text):
            stats.mentions_unfound += 1
            continue
        surface = text[start:end]
        claimed = ent.get("text")
        if claimed is not None and str(claimed) != surface:
            stats.mentions_unfound += 1
            continue
        spans.append((start, end, mapped))
        stats.spans_aligned += 1
    if not had_keep and not spans:
        stats.docs_dropped_empty += 1
        return None, stats
    stats.docs_with_keep_mentions = 1 if had_keep else 0
    native = str(row.get("id") or meta.get("id") or "")
    return finalize_record("uzner_100k_real", native, text, spans, stats), stats


def _uzner_chunk(rows: list[JsonDict]) -> dict[str, Any]:
    records: list[JsonDict] = []
    stats = Stats(source="uzner_100k_real")
    for row in rows:
        record, row_stats = convert_uzner_row(row)
        merge_stats(stats, row_stats)
        if record is not None:
            records.append(record)
    return {"records": records, "stats": stats.as_dict()}


def bio_type(tag: str) -> str | None:
    if not tag or tag in {"O", "o"}:
        return None
    parts = tag.replace("_", "-").split("-")
    return parts[-1].upper() if parts else None


def detokenize(tokens: list[str]) -> tuple[str, list[tuple[int, int]]]:
    no_space_before = set(",.!?;:)]}”’»…%")
    no_space_after = set("([{“‘«#@$")
    parts: list[str] = []
    offsets: list[tuple[int, int]] = []
    pos = 0
    prev: str | None = None
    for token in tokens:
        need_space = prev is not None
        if need_space and (token in no_space_before or (prev in no_space_after)):
            need_space = False
        if need_space and token[:1] in APOSTROPHES:
            need_space = False
        if need_space:
            parts.append(" ")
            pos += 1
        start = pos
        parts.append(token)
        pos += len(token)
        offsets.append((start, pos))
        prev = token
    return "".join(parts), offsets


def bio_tokens_to_spans(
    tags: list[str],
    offsets: list[tuple[int, int]],
    mapping: dict[str, str],
    stats: Stats,
) -> list[Span]:
    spans: list[Span] = []
    current: Span | None = None
    for (start, end), tag in zip(offsets, tags, strict=True):
        raw_type = bio_type(tag)
        mapped = mapping.get(raw_type) if raw_type else None
        if raw_type and mapped is None:
            stats.mentions_other_type += 1
            stats.other_type_counts[raw_type] = stats.other_type_counts.get(raw_type, 0) + 1
        if mapped is None:
            if current is not None:
                spans.append(current)
                current = None
            continue
        prefix = tag.split("-", 1)[0]
        if prefix == "B" or current is None or current[2] != mapped:
            if current is not None:
                spans.append(current)
            current = (start, end, mapped)
            stats.mentions_keep_type += 1
            stats.spans_aligned += 1
        else:
            current = (current[0], end, mapped)
    if current is not None:
        spans.append(current)
    return spans


def convert_uznlp_sentence(
    sentence_id: int,
    tokens: list[str],
    tags: list[str],
) -> tuple[JsonDict | None, Stats]:
    stats = Stats(source="uznlp_uzbek_ner_gold")
    stats.docs_in = 1
    text, offsets = detokenize(tokens)
    spans = bio_tokens_to_spans(tags, offsets, UZNLP_MAP, stats)
    if spans:
        stats.docs_with_keep_mentions = 1
    return finalize_record("uznlp_uzbek_ner_gold", str(sentence_id), text, spans, stats), stats


def chunked(items: list[Any], size: int) -> list[list[Any]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def run_pool(
    fn: Any,
    chunks: list[Any],
    workers: int,
) -> tuple[list[JsonDict], Stats]:
    if not chunks:
        source = "unknown"
        return [], Stats(source=source)
    records: list[JsonDict] = []
    merged: Stats | None = None
    workers = max(1, min(workers, len(chunks)))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fn, chunk) for chunk in chunks]
        for future in as_completed(futures):
            payload = future.result()
            records.extend(payload["records"])
            part = stats_from_dict(payload["stats"])
            if merged is None:
                merged = part
            else:
                merge_stats(merged, part)
    assert merged is not None
    return records, merged


def load_official_index(paths: list[Path]) -> tuple[set[str], set[str]]:
    exact: set[str] = set()
    normed: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                text = rec.get("text") or ""
                exact.add(text)
                normed.add(normalize_text(text))
    return exact, normed


def drop_overlaps(
    records: list[JsonDict],
    stats: Stats,
    official_exact: set[str],
    official_norm: set[str],
    seen_norm: set[str],
) -> list[JsonDict]:
    kept: list[JsonDict] = []
    for rec in records:
        text = rec["text"]
        if text in official_exact or normalize_text(text) in official_norm:
            stats.docs_dropped_official += 1
            continue
        key = normalize_text(text)
        if key in seen_norm:
            stats.docs_dropped_cross_source += 1
            continue
        seen_norm.add(key)
        kept.append(rec)
    stats.entities_by_label = {}
    stats.entities_emitted = 0
    stats.docs_kept = len(kept)
    for rec in kept:
        stats.entities_emitted += len(rec["entities"])
        for ent in rec["entities"]:
            stats.add_label(str(ent["label"]))
    return kept


def official_suffix_selfcheck(paths: list[Path]) -> dict[str, Any]:
    """Estimate rewrite precision against official gold (lemma→full span)."""

    recover_ok = 0
    recover_fail = 0
    no_suffix = 0
    false_extend = 0
    false_examples: list[str] = []
    fail_examples: list[str] = []
    n_entities = 0
    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                rec = json.loads(line)
                text = rec.get("text") or ""
                for ent in rec.get("entities") or []:
                    start, end = ent["start"], ent["end"]
                    n_entities += 1
                    surface = text[start:end]
                    gold_suffix = None
                    for suffix in SUFFIXES:
                        if len(surface) > len(suffix) + 1 and surface.casefold().endswith(suffix):
                            gold_suffix = suffix
                            break
                    if gold_suffix:
                        stem_end = end - len(gold_suffix)
                        while stem_end > start and text[stem_end - 1] in APOSTROPHES:
                            stem_end -= 1
                        new_end, matched, reason = rewrite_span_end(text, start, stem_end)
                        if new_end == end and reason == "extended":
                            recover_ok += 1
                        else:
                            recover_fail += 1
                            if len(fail_examples) < 8:
                                fail_examples.append(
                                    f"{surface!r} suf={gold_suffix!r} got={matched!r} reason={reason}"
                                )
                    else:
                        no_suffix += 1
                        new_end, matched, reason = rewrite_span_end(text, start, end)
                        if reason == "extended":
                            false_extend += 1
                            if len(false_examples) < 8:
                                false_examples.append(
                                    f"{surface!r} + {matched!r} → {text[start:new_end]!r}"
                                )
    recover_n = recover_ok + recover_fail
    return {
        "official_entities": n_entities,
        "gold_spans_with_allowlisted_suffix": recover_n,
        "rewrite_recovers_gold_n": recover_ok,
        "rewrite_recovers_gold_pct": round(100.0 * recover_ok / recover_n, 3) if recover_n else 0.0,
        "rewrite_fail_n": recover_fail,
        "gold_spans_without_suffix": no_suffix,
        "false_extend_n": false_extend,
        "false_extend_pct_of_no_suffix": round(100.0 * false_extend / no_suffix, 4)
        if no_suffix
        else 0.0,
        "fail_examples": fail_examples,
        "false_extend_examples": false_examples,
        "policy": (
            "Extend only when the following characters are an allowlisted Uzbek "
            "case/plural ending, attached (no space), not after a closing quote, "
            "and the current span does not already end in punctuation. "
            "A following letter blocks the match so -da does not eat dastur."
        ),
    }


def write_jsonl(path: Path, records: list[JsonDict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def extract_risqaliyevds(path: Path, workers: int, chunk_size: int) -> tuple[list[JsonDict], Stats]:
    from datasets import load_from_disk

    print(f"[risqaliyevds] loading {path}", flush=True)
    dataset = load_from_disk(str(path))["train"]
    frame = dataset.to_pandas()
    rows: list[JsonDict] = []
    for rec in frame[["text", "ner"]].to_dict("records"):
        ner = rec.get("ner") or {}
        if not isinstance(ner, dict):
            ner = {}
        rows.append(
            {
                "text": rec.get("text") or "",
                "ner": {str(key): _mention_list(value) for key, value in ner.items()},
            }
        )
    print(f"[risqaliyevds] {len(rows)} docs, {workers} workers", flush=True)
    chunks = chunked(rows, chunk_size)
    del frame, dataset, rows
    return run_pool(_risq_chunk, chunks, workers)


def extract_uzner(path: Path, workers: int, chunk_size: int) -> tuple[list[JsonDict], Stats]:
    print(f"[uzner] reading {path}", flush=True)
    rows: list[JsonDict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"[uzner] {len(rows)} train rows", flush=True)
    chunks = chunked(rows, chunk_size)
    del rows
    return run_pool(_uzner_chunk, chunks, workers)


def extract_uznlp(path: Path) -> tuple[list[JsonDict], Stats]:
    import pandas as pd
    from datasets import load_from_disk

    print(f"[uznlp] loading {path}", flush=True)
    dataset = load_from_disk(str(path))["train"]
    frame: pd.DataFrame = dataset.to_pandas()
    frame = frame.sort_values(["Sentence", "TokenOrder"], kind="mergesort")
    records: list[JsonDict] = []
    stats = Stats(source="uznlp_uzbek_ner_gold")
    grouped = frame.groupby("Sentence", sort=True)
    print(f"[uznlp] {grouped.ngroups} sentences / {len(frame)} tokens", flush=True)
    for sentence_id, group in grouped:
        tokens = [str(tok) for tok in group["Token"].tolist()]
        tags = [str(tag) for tag in group["NER_Tag"].tolist()]
        record, row_stats = convert_uznlp_sentence(int(sentence_id), tokens, tags)
        merge_stats(stats, row_stats)
        if record is not None:
            records.append(record)
    return records, stats


def render_markdown(
    out_dir: Path,
    per_source: dict[str, Stats],
    combined: dict[str, Any],
    suffix_check: dict[str, Any],
    official_overlap: dict[str, Any],
) -> str:
    def fmt_labels(stats: Stats) -> str:
        parts = [f"{lab} {stats.entities_by_label.get(lab, 0):,}" for lab in LABELS]
        return ", ".join(parts)

    rel_out = out_dir
    try:
        rel_out = out_dir.resolve().relative_to(REPO_ROOT)
    except ValueError:
        rel_out = out_dir
    lines: list[str] = [
        "# Silver extract — mapped external Uzbek NER",
        "",
        "Generated by `uv run python scripts/extract_silver.py`. Blobs under "
        "`data/processed/silver/*.jsonl` are gitignored; this note is the committed "
        "record of counts, mappings, and caveats.",
        "",
        "Do **not** mix these rows into official `dev.jsonl` scoring. Silver is Mix B "
        "augmentation only (`docs/EDA_official.md`).",
        "",
        "## Command",
        "",
        "```bash",
        "uv run python scripts/extract_silver.py",
        "```",
        "",
        "CPU only (`CUDA_VISIBLE_DEVICES` is cleared). Uses `datasets`, `pandas`, and "
        "a process pool (default: all logical cores).",
        "",
        "## Keep / map / reject",
        "",
        "| Source | Action | Map |",
        "|---|---|---|",
        "| `risqaliyevds_uzbek_ner` | **KEEP** (primary) | PERSON/PER→NAME, ORG→ORG, "
        "GPE/LOC/FAC/FACILITY→GEO |",
        "| `uzner_100k` train, `meta.is_synthetic==false` | **KEEP** (optional real) | "
        "PER→NAME, ORG→ORG, GPE/LOC→GEO. FAC dropped |",
        "| `uznlp_uzbek_ner_gold` | **KEEP** (small BIO) | PER→NAME, ORG→ORG, LOC→GEO |",
        "| official train/dev | gold only — **not** copied into silver | native |",
        "| `ner_prepared_uzbek` | **REJECT** | B-only duplicate of risqaliyevds |",
        "| `uz_medner` | **REJECT** | medical types |",
        "| `rubai_ner_150k_personal` | **REJECT** | synthetic PII |",
        "| `uzlegalner_v3`, `uzbek_legal_ner_full` | **REJECT** | legal extra types + "
        "suffix-excluding spans |",
        "| `wikiann_uz` | **REJECT** | weak Wikipedia silver, tiny |",
        "",
        "Rejected paths were not read into the jsonl outputs: "
        + ", ".join(f"`{name}`" for name in REJECTED_SOURCES)
        + ".",
        "",
        "## Output files",
        "",
        f"Directory: `{rel_out}`",
        "",
        "| File | Role |",
        "|---|---|",
        "| `risqaliyevds.jsonl` | primary mention-aligned news silver |",
        "| `uzner_100k_real.jsonl` | UzNER-100k real subset after type filter |",
        "| `uznlp_gold.jsonl` | small BIO gold |",
        "| `all.jsonl` | concatenation in source order (risq → uzner → uznlp) |",
        "| `stats.json` | machine-readable copy of these counts |",
        "| `README.md` | how to regenerate; gitignore note |",
        "",
        "Schema matches official: `hash`, `text`, `entities[{label,start,end}]` "
        "with exclusive `end`. Silver hashes are `sv` + SHA-1 and do not collide "
        "with organizer hashes.",
        "",
        "## Counts after filtering",
        "",
        "| Source | Docs in | Docs kept | Entities | ORG | NAME | GEO |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    order = ("risqaliyevds_uzbek_ner", "uzner_100k_real", "uznlp_uzbek_ner_gold")
    for key in order:
        stats = per_source[key]
        lines.append(
            f"| `{key}` | {stats.docs_in:,} | {stats.docs_kept:,} | "
            f"{stats.entities_emitted:,} | {stats.entities_by_label.get('ORG', 0):,} | "
            f"{stats.entities_by_label.get('NAME', 0):,} | "
            f"{stats.entities_by_label.get('GEO', 0):,} |"
        )
    lines.extend(
        [
            f"| **all (deduped concat)** | — | {combined['docs']:,} | "
            f"{combined['entities']:,} | {combined['by_label'].get('ORG', 0):,} | "
            f"{combined['by_label'].get('NAME', 0):,} | "
            f"{combined['by_label'].get('GEO', 0):,} |",
            "",
        ]
    )
    for key in order:
        stats = per_source[key]
        lines.extend(
            [
                f"### `{key}`",
                "",
                f"- Labels kept: {fmt_labels(stats)}",
                f"- Keep-type mentions seen: {stats.mentions_keep_type:,}",
                f"- Other types dropped: {stats.mentions_other_type:,}",
                f"- Unfound / invalid offsets: {stats.mentions_unfound:,}",
                f"- Type-conflict mention surfaces: {stats.mentions_type_conflict:,}",
                f"- Overlap/nested spans dropped: {stats.spans_overlap_dropped:,}",
                f"- Empty after filter: {stats.docs_dropped_empty:,}",
                f"- Dropped as official train/dev overlap: {stats.docs_dropped_official:,}",
                f"- Dropped as cross-source duplicate: {stats.docs_dropped_cross_source:,}",
                f"- Synthetic docs dropped: {stats.docs_synthetic_dropped:,}",
                f"- Suffix spans extended: {stats.suffix_extended:,}",
                f"- Suffix skipped (space / separate particle): {stats.suffix_skipped_space:,}",
                f"- Suffix skipped (quote or trailing punct): {stats.suffix_skipped_punct_or_quote:,}",
                "",
            ]
        )
        if stats.other_type_counts:
            top = list(stats.other_type_counts.items())[:12]
            pretty = ", ".join(f"{name} {n:,}" for name, n in top)
            lines.extend([f"Dropped extra types (top): {pretty}", ""])
        if stats.suffix_by_form:
            top_s = list(stats.suffix_by_form.items())[:12]
            pretty_s = ", ".join(f"{name!r} × {n:,}" for name, n in top_s)
            lines.extend([f"Suffix forms added (top): {pretty_s}", ""])

    lines.extend(
        [
            "## Mention → span alignment (risqaliyevds)",
            "",
            "The HF set stores **sets of mention strings**, not offsets. Alignment:",
            "",
            "1. Keep PERSON/PER, ORG, GPE/LOC/FAC/FACILITY; drop DATE/EVENT/MONEY/…",
            "2. If the same surface is listed under two mapped types in one doc "
            "(typical PERSON vs GPE on a long honorific string), drop it.",
            "3. Find word-start occurrences (no mid-token hit). Try exact, then "
            "apostrophe folding (`O‘`/`Oʻ`/`O'`), then case-insensitive windows "
            "of the original mention length.",
            "4. Mention lists are de-duplicated in the source, so an unambiguous "
            "surface is expanded to **all** word-start hits. This matches official "
            "rule 4 (each occurrence has its own coordinates). A unique-match-only "
            "policy would throw away repeated `AQSh` / `Toshkent` mentions.",
            "5. Ghost mentions that never occur in `text` (very common: `O‘zbekiston` "
            "on a foreign-news article) are dropped.",
            "6. After suffix rewrite, overlapping/nested spans are resolved by keeping "
            "the **longer** span (flat ontology).",
            "",
            "## Suffix rewrite",
            "",
            "UzNER §2.4 leaves case endings outside the span when they are a separate "
            "token (`Toshkent` + `da`). Official gold includes them (`Toshkentda`). "
            "The rewriter extends `end` only when:",
            "",
            "- `text[end:]` starts with an allowlisted ending "
            "(longest of `lardagi/laridan/ning/dagi/dan/ga/ni/da/lar` and Cyrillic mirrors),",
            "- optionally after an apostrophe (`Mobiuz'dan`),",
            "- the suffix is **attached** (no leading space — `KFC da` stays `KFC`),",
            "- the span does not already end in punctuation (`Abror J.` + `ga` is not extended),",
            "- a closing quote/bracket is not between stem and suffix (`«EVOS»ga` stays `EVOS`),",
            "- the character after the suffix is a non-letter (or EOS), so `-da` cannot "
            "consume `dastur`.",
            "",
            "Self-check on official train+dev (strip a detected gold suffix, re-apply):",
            "",
            f"- Official entities: {suffix_check['official_entities']:,}",
            f"- Gold spans with allowlisted suffix: {suffix_check['gold_spans_with_allowlisted_suffix']:,}",
            f"- Recovered exact gold `end`: {suffix_check['rewrite_recovers_gold_n']:,} "
            f"({suffix_check['rewrite_recovers_gold_pct']}%)",
            f"- Failures: {suffix_check['rewrite_fail_n']:,}",
            f"- Gold spans without suffix: {suffix_check['gold_spans_without_suffix']:,}",
            f"- False extends on those: {suffix_check['false_extend_n']:,} "
            f"({suffix_check['false_extend_pct_of_no_suffix']}% of no-suffix spans)",
            "",
            suffix_check["policy"],
            "",
        ]
    )
    if suffix_check.get("false_extend_examples"):
        lines.append("False-extend examples: " + "; ".join(suffix_check["false_extend_examples"]))
        lines.append("")
    if suffix_check.get("fail_examples"):
        lines.append("Recover-fail examples: " + "; ".join(suffix_check["fail_examples"]))
        lines.append("")

    lines.extend(
        [
            "## Overlap with official texts",
            "",
            "Dedup is exact text or whitespace/apostrophe/case-normalized text. "
            "Official rows are never copied into silver.",
            "",
            f"- Official docs indexed: {official_overlap['official_docs']:,}",
            f"- Silver docs dropped as official overlap: {official_overlap['dropped_official']:,}",
            f"- Silver docs dropped as cross-source duplicates: {official_overlap['dropped_cross']:,}",
            "",
            "## Caveats",
            "",
            "- **Ontology shift.** Wiki/news ORG is mostly institutions; official ORG also "
            "covers platforms, QSR brands, clubs, mosques. GEO collapse of GPE+LOC(+FAC on "
            "risqaliyevds) still mis-tags some sports clubs the way a gazetteer would.",
            "- **FAC on UzNER is dropped.** Official GEO includes airports/stadiums/markets, "
            "but UzNER FAC also includes mosques/churches which the labeling guide marks ORG.",
            "- **Risqaliyevds mention lists are noisy.** They include ghost toponyms, "
            "honorifics inside PERSON, and generic nouns (`kompaniyasi`) inside ORG. "
            "Alignment + conflict drop removes the worst, not all of it.",
            "- **Occurrence expansion** can over-tag a surface that changes role in the "
            "same document (`Andijon` city vs club) because the source has no per-hit type.",
            "- **UzNER real partition** uses paper-aligned `meta.is_synthetic` / "
            "`source_tier` (70k real / 30k synthetic). Original provenance fields are ignored "
            "for the keep/drop decision. Dev/test/gold_candidate/hard_eval are not ingested "
            "(EDA skip list; would duplicate jsonl/conll).",
            "- **uznlp detokenization** reconstructs text from tokens (space except around "
            "punctuation). Offsets are consistent with that reconstructed string, not a "
            "missing original document.",
            "- **Empty official docs are 18% by design.** This extract emits only docs that "
            "still have ≥1 mapped entity after filtering, so it is dense NER silver — cap "
            "or downsample before concatenating with the 13k gold (EDA Mix B).",
            "- **Do not evaluate on a merged dev.** Leaderboard analogue is official `dev.jsonl` only.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=settings.data_processed / "silver",
        help="Output directory for jsonl + stats",
    )
    parser.add_argument(
        "--docs-out",
        type=Path,
        default=REPO_ROOT / "docs" / "SILVER_EXTRACT.md",
        help="Markdown report path (committed)",
    )
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 8)
    parser.add_argument("--chunk-size", type=int, default=400)
    parser.add_argument(
        "--skip-risq",
        action="store_true",
        help="Skip primary risqaliyevds extract",
    )
    parser.add_argument("--skip-uzner", action="store_true")
    parser.add_argument("--skip-uznlp", action="store_true")
    parser.add_argument(
        "--skip-docs",
        action="store_true",
        help="Do not rewrite docs/SILVER_EXTRACT.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    workers = max(1, int(args.workers))
    print(
        f"CPU workers={workers} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}",
        flush=True,
    )

    official_paths = [settings.official_train, settings.official_dev]
    print("[official] suffix self-check + text index", flush=True)
    suffix_check = official_suffix_selfcheck(official_paths)
    official_exact, official_norm = load_official_index(official_paths)
    print(
        f"[official] {len(official_exact)} texts; "
        f"suffix recover {suffix_check['rewrite_recovers_gold_pct']}%",
        flush=True,
    )

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    external = settings.data_external
    risq_path = external / "hf" / "risqaliyevds_uzbek_ner" / "hf_dataset"
    uzner_path = external / "zenodo" / "uzner_100k" / "extracted" / "uzner_train_bioes.jsonl"
    uznlp_path = external / "hf" / "uznlp_uzbek_ner_gold" / "hf_dataset"

    per_source: dict[str, Stats] = {}
    written: dict[str, list[JsonDict]] = {}
    seen_norm: set[str] = set()

    if not args.skip_risq:
        records, stats = extract_risqaliyevds(risq_path, workers, args.chunk_size)
        records = drop_overlaps(records, stats, official_exact, official_norm, seen_norm)
        write_jsonl(out_dir / "risqaliyevds.jsonl", records)
        per_source[stats.source] = stats
        written[stats.source] = records
        print(
            f"[risqaliyevds] kept {stats.docs_kept} docs / {stats.entities_emitted} ents",
            flush=True,
        )
    else:
        per_source["risqaliyevds_uzbek_ner"] = Stats(source="risqaliyevds_uzbek_ner")
        written["risqaliyevds_uzbek_ner"] = []

    if not args.skip_uzner:
        records, stats = extract_uzner(uzner_path, workers, args.chunk_size)
        records = drop_overlaps(records, stats, official_exact, official_norm, seen_norm)
        write_jsonl(out_dir / "uzner_100k_real.jsonl", records)
        per_source[stats.source] = stats
        written[stats.source] = records
        print(
            f"[uzner] kept {stats.docs_kept} docs / {stats.entities_emitted} ents",
            flush=True,
        )
    else:
        per_source["uzner_100k_real"] = Stats(source="uzner_100k_real")
        written["uzner_100k_real"] = []

    if not args.skip_uznlp:
        records, stats = extract_uznlp(uznlp_path)
        records = drop_overlaps(records, stats, official_exact, official_norm, seen_norm)
        write_jsonl(out_dir / "uznlp_gold.jsonl", records)
        per_source[stats.source] = stats
        written[stats.source] = records
        print(
            f"[uznlp] kept {stats.docs_kept} docs / {stats.entities_emitted} ents",
            flush=True,
        )
    else:
        per_source["uznlp_uzbek_ner_gold"] = Stats(source="uznlp_uzbek_ner_gold")
        written["uznlp_uzbek_ner_gold"] = []

    all_records: list[JsonDict] = []
    for key in ("risqaliyevds_uzbek_ner", "uzner_100k_real", "uznlp_uzbek_ner_gold"):
        all_records.extend(written.get(key, []))
    write_jsonl(out_dir / "all.jsonl", all_records)

    combined_labels: Counter[str] = Counter()
    for rec in all_records:
        for ent in rec["entities"]:
            combined_labels[str(ent["label"])] += 1
    combined = {
        "docs": len(all_records),
        "entities": sum(combined_labels.values()),
        "by_label": dict(combined_labels),
    }
    official_overlap = {
        "official_docs": len(official_exact),
        "dropped_official": sum(s.docs_dropped_official for s in per_source.values()),
        "dropped_cross": sum(s.docs_dropped_cross_source for s in per_source.values()),
    }

    stats_payload = {
        "combined": combined,
        "sources": {key: stats.as_dict() for key, stats in per_source.items()},
        "suffix_selfcheck_official": suffix_check,
        "official_overlap": official_overlap,
        "rejected_sources": list(REJECTED_SOURCES),
        "maps": {
            "risqaliyevds_uzbek_ner": RISQ_MAP,
            "uzner_100k_real": UZNER_MAP,
            "uznlp_uzbek_ner_gold": UZNLP_MAP,
        },
        "workers": workers,
    }
    (out_dir / "stats.json").write_text(
        json.dumps(stats_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    if not args.skip_docs:
        markdown = render_markdown(out_dir, per_source, combined, suffix_check, official_overlap)
        args.docs_out.parent.mkdir(parents=True, exist_ok=True)
        args.docs_out.write_text(markdown, encoding="utf-8")
        print(f"wrote {args.docs_out}", flush=True)

    print("=== silver summary ===", flush=True)
    print(json.dumps(combined, ensure_ascii=False), flush=True)
    for key, stats in per_source.items():
        print(
            f"{key}: docs={stats.docs_kept} ents={stats.entities_emitted} "
            f"labels={stats.entities_by_label}",
            flush=True,
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Download public Uzbek NER / span-annotation datasets into data/external/."""

from __future__ import annotations

import argparse
import json
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.request import urlretrieve

from datasets import DownloadConfig, load_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "data" / "external"


@dataclass
class DatasetSpec:
    key: str
    source: str
    license: str
    mapping_to_hackathon: str
    notes: str


HF_DATASETS: list[tuple[str, str | None, DatasetSpec]] = [
    (
        "uznlp-uz/uzbek_NER",
        None,
        DatasetSpec(
            key="uznlp_uzbek_ner_gold",
            source="https://huggingface.co/datasets/uznlp-uz/uzbek_NER",
            license="CC-BY-4.0",
            mapping_to_hackathon="PER→NAME, ORG→ORG, LOC→GEO (+ MISC/MONEY/TEMPORAL/...)",
            notes="Token-level BIO TSV, ~4.2k sentences / 59k tokens.",
        ),
    ),
    (
        "risqaliyevds/uzbek_ner",
        None,
        DatasetSpec(
            key="risqaliyevds_uzbek_ner",
            source="https://huggingface.co/datasets/risqaliyevds/uzbek_ner",
            license="open (see HF card)",
            mapping_to_hackathon="PERSON/PER→NAME, ORG→ORG, LOC/GPE→GEO",
            notes="~19.6k docs, entity mention lists (needs span alignment).",
        ),
    ),
    (
        "ShakhzoDavronov/ner-prepared-uzbek",
        None,
        DatasetSpec(
            key="ner_prepared_uzbek",
            source="https://huggingface.co/datasets/ShakhzoDavronov/ner-prepared-uzbek",
            license="open (see HF card)",
            mapping_to_hackathon="PERSON→NAME, ORG→ORG, GPE/LOC→GEO",
            notes="Token BIO derived from risqaliyevds, 19.6k sentences.",
        ),
    ),
    (
        "unimelb-nlp/wikiann",
        "uz",
        DatasetSpec(
            key="wikiann_uz",
            source="https://huggingface.co/datasets/unimelb-nlp/wikiann",
            license="CC-BY-SA-4.0",
            mapping_to_hackathon="PER→NAME, ORG→ORG, LOC→GEO",
            notes="Weakly supervised Wiki NER, 1k train/dev/test each.",
        ),
    ),
    (
        "islomov/rubai-NER-150K-Personal",
        None,
        DatasetSpec(
            key="rubai_ner_150k_personal",
            source="https://huggingface.co/datasets/islomov/rubai-NER-150K-Personal",
            license="Apache-2.0",
            mapping_to_hackathon="NAME only (+ PII types); useful for NAME class augmentation",
            notes="Synthetic PII, uz+ru, ~142k examples.",
        ),
    ),
    (
        "uznlp-uz/uz_medner",
        None,
        DatasetSpec(
            key="uz_medner",
            source="https://huggingface.co/datasets/uznlp-uz/uz_medner",
            license="see HF card",
            mapping_to_hackathon="Not directly ORG/NAME/GEO — medical domain auxiliary",
            notes="Token-level medical entities, ~20k tokens.",
        ),
    ),
]

ZENODO_ARCHIVES: list[tuple[str, str, DatasetSpec]] = [
    (
        "https://zenodo.org/api/records/18903080/files/UzNER-100K_paper_aligned_70k_real_30k_synthetic_v1.zip/content",
        "uzner_100k.zip",
        DatasetSpec(
            key="uzner_100k",
            source="https://doi.org/10.5281/zenodo.18903080",
            license="see Zenodo record",
            mapping_to_hackathon="18 fine-grained types → map PER→NAME, ORG→ORG, LOC/GPE→GEO",
            notes="114k sentences, BIOES, main academic benchmark (Mar 2026).",
        ),
    ),
    (
        "https://zenodo.org/api/records/18816402/files/UzLegalNER_Zenodo.zip/content",
        "uzlegalner_v3.zip",
        DatasetSpec(
            key="uzlegalner_v3",
            source="https://doi.org/10.5281/zenodo.18816402",
            license="see Zenodo record",
            mapping_to_hackathon="PER→NAME, ORG→ORG, LOC→GEO (+ legal-specific types)",
            notes="Legal contracts NER, JSONL + CoNLL splits.",
        ),
    ),
    (
        "https://zenodo.org/api/records/19682709/files/Uzbek_Legal_NER_Full_Release_Package_FINAL.zip/content",
        "uzbek_legal_ner_full.zip",
        DatasetSpec(
            key="uzbek_legal_ner_full",
            source="https://doi.org/10.5281/zenodo.19682709",
            license="see Zenodo record",
            mapping_to_hackathon="PER→NAME, ORG→ORG, LOC→GEO (+ legal types)",
            notes="Core gold + augmented legal NER, multi-format.",
        ),
    ),
]


def save_hf_dataset(repo_id: str, config: str | None, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    dl_cfg = DownloadConfig(cache_dir=str(out_dir / ".cache"))
    ds = (
        load_dataset(repo_id, config, download_config=dl_cfg)
        if config
        else load_dataset(repo_id, download_config=dl_cfg)
    )
    ds.save_to_disk(str(out_dir / "hf_dataset"))
    stats = {split: len(ds[split]) for split in ds}
    return {"splits": stats, "path": str(out_dir / "hf_dataset")}


def download_zip(url: str, archive_path: Path, extract_dir: Path) -> dict:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if not archive_path.exists():
        print(f"Downloading {url}")
        urlretrieve(url, archive_path)
    extract_dir.mkdir(parents=True, exist_ok=True)
    marker = extract_dir / ".extracted"
    if not marker.exists():
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(extract_dir)
        marker.write_text("ok", encoding="utf-8")
    size_mb = sum(f.stat().st_size for f in extract_dir.rglob("*") if f.is_file()) / 1024 / 1024
    return {
        "archive": str(archive_path),
        "extracted_to": str(extract_dir),
        "size_mb": round(size_mb, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--skip-hf", action="store_true")
    parser.add_argument("--skip-zenodo", action="store_true")
    parser.add_argument("--only", nargs="*", help="Dataset keys to download")
    args = parser.parse_args()

    manifest: dict = {"hf": {}, "zenodo": {}, "catalog": {}}
    wanted = set(args.only or [])

    if not args.skip_hf:
        for repo_id, config, spec in HF_DATASETS:
            if wanted and spec.key not in wanted:
                continue
            target = args.out / "hf" / spec.key
            print(f"[HF] {spec.key} ({repo_id})")
            try:
                manifest["hf"][spec.key] = {
                    "meta": asdict(spec),
                    **save_hf_dataset(repo_id, config, target),
                }
            except Exception as exc:
                manifest["hf"][spec.key] = {"meta": asdict(spec), "error": str(exc)}
                print(f"  ERROR: {exc}")

    if not args.skip_zenodo:
        for url, filename, spec in ZENODO_ARCHIVES:
            if wanted and spec.key not in wanted:
                continue
            target = args.out / "zenodo" / spec.key
            archive = target / filename
            print(f"[Zenodo] {spec.key}")
            try:
                manifest["zenodo"][spec.key] = {
                    "meta": asdict(spec),
                    **download_zip(url, archive, target / "extracted"),
                }
            except Exception as exc:
                manifest["zenodo"][spec.key] = {"meta": asdict(spec), "error": str(exc)}
                print(f"  ERROR: {exc}")

    for _, _, spec in HF_DATASETS:
        manifest["catalog"][spec.key] = asdict(spec)
    for _, _, spec in ZENODO_ARCHIVES:
        manifest["catalog"][spec.key] = asdict(spec)

    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()

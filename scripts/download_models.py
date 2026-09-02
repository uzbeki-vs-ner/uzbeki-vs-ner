#!/usr/bin/env python3
"""Download BERT-family checkpoints for Uzbek NER experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = REPO_ROOT / "models" / "pretrained"

MODELS: dict[str, str] = {
    "exp0_xlm_roberta_base": "FacebookAI/xlm-roberta-base",
    "exp1_uztext_roberta": "rifkat/uztext-3Gb-BPE-Roberta",
    "exp2_mdeberta_v3_base": "microsoft/mdeberta-v3-base",
}


def download_model(model_id: str, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {model_id} -> {target_dir}")
    path = snapshot_download(
        repo_id=model_id,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
    )
    print(f"Done: {path}")
    return Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE,
        help="Root directory for downloaded checkpoints",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        choices=list(MODELS.keys()) + list(MODELS.values()),
        help="Download only selected models (alias or HF id)",
    )
    args = parser.parse_args()

    selected = MODELS.items()
    if args.only:
        wanted = set(args.only)
        selected = [
            (alias, mid) for alias, mid in MODELS.items() if alias in wanted or mid in wanted
        ]

    for alias, model_id in selected:
        download_model(model_id, args.cache_dir / alias)


if __name__ == "__main__":
    main()

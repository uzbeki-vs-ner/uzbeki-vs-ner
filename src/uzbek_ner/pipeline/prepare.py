"""Stage: prepare — validate official organizer data and emit processed manifest."""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger
from omegaconf import DictConfig, OmegaConf

from uzbek_ner.settings import Settings, get_settings
from uzbek_ner.tracking import RESEARCH_EXPERIMENT, log_json, log_params, setup_experiment

OFFICIAL_FILES = ("train.jsonl", "dev.jsonl", "dataset_manifest.json")


def run_prepare(cfg: DictConfig, settings: Settings | None = None) -> Path:
    runtime = settings or get_settings()
    runtime.data_processed.mkdir(parents=True, exist_ok=True)
    official = runtime.data_official

    missing = [name for name in OFFICIAL_FILES if not (official / name).exists()]
    stats: dict[str, object] = {}

    if not missing:
        manifest_src = json.loads((official / "dataset_manifest.json").read_text(encoding="utf-8"))
        stats = {
            "source": "official",
            "priority": 1,
            "train_records": manifest_src["splits"]["train"]["records"],
            "dev_records": manifest_src["splits"]["dev"]["records"],
            "labels": manifest_src["schema"]["labels"],
        }

    manifest = {
        "status": "ready" if not missing else "waiting_for_data",
        "source": "official",
        "priority": 1,
        "official_dir": str(official),
        "expected_files": list(OFFICIAL_FILES),
        "missing_files": missing,
        "stats": stats,
        "external_dir": str(runtime.data_external),
        "external_note": "augmentation only — do not mix with official dev for reporting",
        "prepare": OmegaConf.to_container(cfg.prepare, resolve=True),
    }
    manifest_path = runtime.data_processed / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    with setup_experiment(RESEARCH_EXPERIMENT, run_name="prepare"):
        log_params({"seed": str(cfg.seed), **OmegaConf.to_container(cfg.prepare, resolve=True)})
        log_json(manifest, "prepare", "manifest.json")

    if missing:
        logger.warning(
            "Official dataset incomplete ({}). Expected under {}.",
            ", ".join(missing),
            official,
        )
    else:
        logger.info(
            "Official data OK: train={}, dev={}", stats["train_records"], stats["dev_records"]
        )

    return manifest_path

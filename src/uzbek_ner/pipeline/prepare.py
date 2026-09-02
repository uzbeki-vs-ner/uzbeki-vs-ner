"""Stage: prepare — validate raw data layout and emit processed placeholder."""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger
from omegaconf import DictConfig, OmegaConf

from uzbek_ner.settings import Settings, get_settings
from uzbek_ner.tracking import RESEARCH_EXPERIMENT, log_json, log_params, setup_experiment


def _expected_raw_files(data_raw: Path) -> list[str]:
    return ["train.json", "val.json"]


def run_prepare(cfg: DictConfig, settings: Settings | None = None) -> Path:
    runtime = settings or get_settings()
    runtime.data_processed.mkdir(parents=True, exist_ok=True)

    raw_files = _expected_raw_files(runtime.data_raw)
    missing = [name for name in raw_files if not (runtime.data_raw / name).exists()]

    manifest = {
        "status": "ready" if not missing else "waiting_for_data",
        "raw_dir": str(runtime.data_raw),
        "expected_files": raw_files,
        "missing_files": missing,
        "prepare": OmegaConf.to_container(cfg.prepare, resolve=True),
    }
    manifest_path = runtime.data_processed / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    with setup_experiment(RESEARCH_EXPERIMENT, run_name="prepare"):
        log_params({"seed": str(cfg.seed), **OmegaConf.to_container(cfg.prepare, resolve=True)})
        log_json(manifest, "prepare", "manifest.json")

    if missing:
        logger.warning(
            "Raw dataset not found yet ({}). Place files into {} and rerun.",
            ", ".join(missing),
            runtime.data_raw,
        )
    else:
        logger.info("Prepare stage finished: {}", manifest_path)

    return manifest_path

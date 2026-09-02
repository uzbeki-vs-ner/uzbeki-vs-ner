"""Stage: train — stub that records hyperparameters until training is implemented."""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger
from omegaconf import DictConfig, OmegaConf

from uzbek_ner.settings import Settings, get_settings
from uzbek_ner.tracking import RESEARCH_EXPERIMENT, log_json, log_params, setup_experiment


def run_train(cfg: DictConfig, settings: Settings | None = None) -> Path:
    runtime = settings or get_settings()
    runtime.checkpoints.mkdir(parents=True, exist_ok=True)

    manifest_path = runtime.data_processed / "manifest.json"
    if not manifest_path.exists():
        msg = f"Missing {manifest_path}. Run `ner prepare` or `dvc repro prepare` first."
        raise FileNotFoundError(msg)

    stub = {
        "status": "stub",
        "message": "Training not implemented yet — hyperparameters logged to MLflow.",
        "train": OmegaConf.to_container(cfg.train, resolve=True),
    }
    checkpoint_meta = runtime.checkpoints / "run_meta.json"
    checkpoint_meta.write_text(json.dumps(stub, indent=2, ensure_ascii=False), encoding="utf-8")

    with setup_experiment(RESEARCH_EXPERIMENT, run_name="train"):
        log_params({"seed": str(cfg.seed), **OmegaConf.to_container(cfg.train, resolve=True)})
        log_json(stub, "train", "run_meta.json")

    logger.info("Train stub finished: {}", checkpoint_meta)
    return checkpoint_meta

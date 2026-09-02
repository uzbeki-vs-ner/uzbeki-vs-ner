"""Stage: evaluate — stub metrics file for DVC metrics tracking."""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger
from omegaconf import DictConfig, OmegaConf

from uzbek_ner.settings import Settings, get_settings
from uzbek_ner.tracking import (
    RESEARCH_EXPERIMENT,
    log_json,
    log_metrics,
    log_params,
    setup_experiment,
)


def run_evaluate(cfg: DictConfig, settings: Settings | None = None) -> Path:
    runtime = settings or get_settings()
    checkpoint_meta = runtime.checkpoints / "run_meta.json"
    if not checkpoint_meta.exists():
        msg = f"Missing {checkpoint_meta}. Run `ner train` or `dvc repro train` first."
        raise FileNotFoundError(msg)

    metrics = {
        "micro_f1": 0.0,
        "micro_precision": 0.0,
        "micro_recall": 0.0,
        "org_f1": 0.0,
        "name_f1": 0.0,
        "geo_f1": 0.0,
        "status": "stub",
    }
    metrics_path = runtime.metrics_path
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    with setup_experiment(RESEARCH_EXPERIMENT, run_name="evaluate"):
        log_params({"seed": str(cfg.seed), **OmegaConf.to_container(cfg.evaluate, resolve=True)})
        log_metrics({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))})
        log_json(metrics, "evaluate", "metrics.json")

    logger.info("Evaluate stub finished: {}", metrics_path)
    return metrics_path

"""Stage: evaluate — score prediction JSONL vs official gold (CPU, no model load)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from loguru import logger
from omegaconf import DictConfig, OmegaConf

from uzbek_ner.evaldash.registry import evaluate_and_register
from uzbek_ner.settings import REPO_ROOT, Settings, get_settings
from uzbek_ner.tracking import (
    RESEARCH_EXPERIMENT,
    log_json,
    log_metrics,
    log_params,
    setup_experiment,
)

WAITING_METRICS: dict[str, float | str] = {
    "micro_f1": 0.0,
    "micro_precision": 0.0,
    "micro_recall": 0.0,
    "org_f1": 0.0,
    "name_f1": 0.0,
    "geo_f1": 0.0,
    "status": "waiting_for_predictions",
}


def _as_path(raw: object, fallback: Path) -> Path:
    if raw is None or raw == "":
        return fallback
    path = Path(str(raw))
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _flatten_dvc_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    by_label = cast(dict[str, Any], metrics["by_label"])
    micro = cast(dict[str, Any], metrics["micro"])
    macro = cast(dict[str, Any], metrics["macro"])
    return {
        "micro_f1": float(micro["f1"]),
        "micro_precision": float(micro["precision"]),
        "micro_recall": float(micro["recall"]),
        "macro_f1": float(macro["f1"]),
        "org_f1": float(by_label["ORG"]["f1"]),
        "name_f1": float(by_label["NAME"]["f1"]),
        "geo_f1": float(by_label["GEO"]["f1"]),
    }


def run_evaluate(cfg: DictConfig, settings: Settings | None = None) -> Path:
    runtime = settings or get_settings()
    eval_cfg = cfg.evaluate
    gold_path = _as_path(eval_cfg.get("gold"), runtime.official_dev)
    pred_path = _as_path(
        eval_cfg.get("predictions"),
        REPO_ROOT / "outputs" / "official" / "dev_predictions.jsonl",
    )
    metrics_out = _as_path(
        eval_cfg.get("metrics_out"),
        REPO_ROOT / "outputs" / "official" / "dev_metrics.json",
    )
    run_id = str(eval_cfg.get("run_id") or "dvc_evaluate")
    split = str(eval_cfg.get("split") or "official_dev")
    model = str(cfg.train.get("model_name") or "unknown")
    checkpoint = str(runtime.checkpoints)
    hyperparams = cast(dict[str, Any], OmegaConf.to_container(cfg.train, resolve=True))

    metrics_path = runtime.metrics_path
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    eval_params = cast(dict[str, Any], OmegaConf.to_container(eval_cfg, resolve=True))

    if gold_path.is_file() and pred_path.is_file():
        run = evaluate_and_register(
            gold_path,
            pred_path,
            run_id=run_id,
            model=model,
            metrics_out=metrics_out,
            checkpoint=checkpoint,
            split=split,
            hyperparams=hyperparams,
        )
        dvc_metrics: dict[str, float | str] = {
            **_flatten_dvc_metrics(run.metrics.model_dump(mode="json")),
            "status": "ok",
        }
        logger.info(
            "evaluate {} micro-F1 {:.4f} ({} records)",
            run_id,
            run.metrics.micro.f1,
            run.metrics.records,
        )
        mlflow_payload: dict[str, Any] = run.model_dump(mode="json")
    else:
        dvc_metrics = dict(WAITING_METRICS)
        logger.warning(
            "evaluate skipped scoring (missing gold={} exists={} pred={} exists={})",
            gold_path,
            gold_path.is_file(),
            pred_path,
            pred_path.is_file(),
        )
        mlflow_payload = {
            "status": "waiting_for_predictions",
            "gold": str(gold_path),
            "predictions": str(pred_path),
        }

    metrics_path.write_text(json.dumps(dvc_metrics, indent=2) + "\n", encoding="utf-8")

    with setup_experiment(RESEARCH_EXPERIMENT, run_name="evaluate"):
        log_params({"seed": str(cfg.seed), **{k: str(v) for k, v in eval_params.items()}})
        log_metrics({k: float(v) for k, v in dvc_metrics.items() if isinstance(v, (int, float))})
        log_json(mlflow_payload, "evaluate", "metrics.json")

    logger.info("Evaluate finished: {}", metrics_path)
    return metrics_path

"""Stage: calibrate — entity-confidence τ on cached official-dev logits.

Training does not pick τ. This stage (1) encodes eval if the cache is missing,
(2) sweeps mean-token confidence, (3) writes the τ chosen on the analysis fold.
Faster than a training epoch once the cache exists; still a separate job.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from loguru import logger
from omegaconf import DictConfig, OmegaConf

from uzbek_ner.decode.threshold import (
    DEFAULT_TAUS,
    analysis_held_split,
    gold_lookup,
    sweep_confidence_gate,
)
from uzbek_ner.io.jsonl import read_jsonl_records
from uzbek_ner.modeling.eval_cache import fill_cache, load_merged_npz, merged_ready
from uzbek_ner.settings import REPO_ROOT, Settings, get_settings
from uzbek_ner.tracking import (
    RESEARCH_EXPERIMENT,
    log_json,
    log_metrics,
    log_params,
    setup_experiment,
)

WAITING_METRICS: dict[str, float | str] = {
    "selected_tau": 0.0,
    "held_out_f1": 0.0,
    "held_out_delta_f1": 0.0,
    "analysis_f1": 0.0,
    "status": "waiting_for_cache",
}


def _as_path(raw: object, fallback: Path) -> Path:
    if raw is None or raw == "":
        return fallback
    path = Path(str(raw))
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _taus(raw: object) -> tuple[float, ...]:
    if raw is None:
        return DEFAULT_TAUS
    container: Any = OmegaConf.to_container(raw) if OmegaConf.is_config(raw) else raw
    if not isinstance(container, list):
        raise TypeError(f"calibrate.taus must be a list, got {type(raw)}")
    return tuple(float(value) for value in container)


def run_calibrate(cfg: DictConfig, settings: Settings | None = None) -> Path:
    runtime = settings or get_settings()
    cal_cfg = cfg.calibrate
    gold_path = _as_path(cal_cfg.get("gold"), runtime.official_dev)
    checkpoint = _as_path(cal_cfg.get("checkpoint"), runtime.checkpoints / "uztext_smoke")
    cache_dir = _as_path(
        cal_cfg.get("cache_dir"), REPO_ROOT / "outputs" / "cache" / "uztext_smoke_official_dev"
    )
    report_path = _as_path(cal_cfg.get("output"), REPO_ROOT / "outputs" / "eval" / "threshold.json")
    metrics_path = _as_path(cal_cfg.get("metrics_out"), REPO_ROOT / "threshold_metrics.json")
    fill = bool(cal_cfg.get("fill_cache", True))
    taus = _taus(cal_cfg.get("taus"))
    seed = int(cal_cfg.get("seed", cfg.seed))
    k = int(cal_cfg.get("k", 5))
    batch_size = int(cal_cfg.get("batch_size", cfg.evaluate.get("batch_size", 8)))
    max_length = int(cal_cfg.get("max_length", cfg.prepare.get("max_length", 512)))
    stride = int(cal_cfg.get("stride", cfg.prepare.get("stride", 128)))

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    cal_params = cast(dict[str, Any], OmegaConf.to_container(cal_cfg, resolve=True))

    if not gold_path.is_file():
        dvc_metrics: dict[str, float | str] = dict(WAITING_METRICS)
        payload: dict[str, Any] = {
            "status": "waiting_for_cache",
            "reason": "gold missing",
            "gold": str(gold_path),
            "cache_dir": str(cache_dir),
        }
        logger.warning("calibrate skipped (missing gold={})", gold_path)
    else:
        records = read_jsonl_records(gold_path, require_entities=True)
        if not merged_ready(cache_dir):
            if fill and checkpoint.is_dir():
                logger.info("filling eval cache from {} → {}", checkpoint, cache_dir)
                fill_cache(
                    checkpoint=checkpoint,
                    records=records,
                    cache_dir=cache_dir,
                    batch_size=batch_size,
                    max_length=max_length,
                    stride=stride,
                )
            else:
                dvc_metrics = dict(WAITING_METRICS)
                payload = {
                    "status": "waiting_for_cache",
                    "reason": "merged logits missing; set fill_cache true and a checkpoint",
                    "gold": str(gold_path),
                    "checkpoint": str(checkpoint),
                    "cache_dir": str(cache_dir),
                }
                logger.warning("calibrate skipped (no cache at {})", cache_dir)
                metrics_path.write_text(json.dumps(dvc_metrics, indent=2) + "\n", encoding="utf-8")
                report_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
                with setup_experiment(RESEARCH_EXPERIMENT, run_name="calibrate"):
                    log_params(
                        {
                            "seed": str(cfg.seed),
                            **{key: str(value) for key, value in cal_params.items()},
                        }
                    )
                    log_metrics(
                        {
                            key: float(value)
                            for key, value in dvc_metrics.items()
                            if isinstance(value, (int, float))
                        }
                    )
                    log_json(payload, "calibrate", "threshold.json")
                return metrics_path

        meta_path = cache_dir / "index.json"
        merged_name = "merged_mean_logits.npz"
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            merged_name = str(meta.get("files", {}).get("merged") or merged_name)
        docs = load_merged_npz(cache_dir / merged_name, records)
        gold = gold_lookup(records)
        hashes = [doc.record_hash for doc in docs]
        analysis_hashes, held_out_hashes = analysis_held_split(hashes, k=k, seed=seed)
        sweep = sweep_confidence_gate(
            docs,
            gold,
            analysis_hashes=analysis_hashes,
            held_out_hashes=held_out_hashes,
            taus=taus,
        )
        dvc_metrics = {
            "selected_tau": float(sweep["selected_tau"]),
            "held_out_f1": float(sweep["held_out"]["f1"]),
            "held_out_delta_f1": float(sweep["held_out_delta_f1"]),
            "analysis_f1": float(sweep["analysis"]["f1"]),
            "status": "ok",
        }
        payload = {
            "schema_version": 1,
            "status": "ok",
            "created_at": datetime.now(UTC).isoformat(),
            "checkpoint": str(checkpoint),
            "gold": str(gold_path),
            "cache_dir": str(cache_dir),
            "split": {
                "method": "k-fold round-robin, fold 0 = analysis, rest = held-out",
                "k": k,
                "seed": seed,
                "analysis_docs": len(analysis_hashes),
                "held_out_docs": len(held_out_hashes),
            },
            "decode": ("argmax BIO → drop span if mean predicted-class token p < τ → word snap"),
            **sweep,
        }
        logger.info(
            "calibrate τ={} analysis F1={:.4f} held-out F1={:.4f} (Δ {:+.4f})",
            dvc_metrics["selected_tau"],
            dvc_metrics["analysis_f1"],
            dvc_metrics["held_out_f1"],
            dvc_metrics["held_out_delta_f1"],
        )

    metrics_path.write_text(json.dumps(dvc_metrics, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    with setup_experiment(RESEARCH_EXPERIMENT, run_name="calibrate"):
        log_params(
            {"seed": str(cfg.seed), **{key: str(value) for key, value in cal_params.items()}}
        )
        log_metrics(
            {
                key: float(value)
                for key, value in dvc_metrics.items()
                if isinstance(value, (int, float))
            }
        )
        log_json(payload, "calibrate", "threshold.json")

    logger.info("Calibrate finished: {}", metrics_path)
    return metrics_path

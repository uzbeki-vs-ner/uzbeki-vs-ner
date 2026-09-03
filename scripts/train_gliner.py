#!/usr/bin/env python3
"""Fine-tune urchade/gliner_multi-v2.1 on official JSONL (2080 Ti / fp16).

Does not change the uztext+BIO product default. Pin the physical card from the
shell with CUDA_DEVICE_ORDER=PCI_BUS_ID and the 2080 Ti UUID; this script
refuses to run if cuda:0 is not a 2080 Ti (the 3060 is a different job).
Turing has no bf16. GLiNER's Trainer skips CUDA-OOM batches and returns zero
loss — those skips are counted and written into the metrics JSON.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from gliner import GLiNER
from loguru import logger

from uzbek_ner.gliner_data import convert_record
from uzbek_ner.gliner_windows import (
    DEFAULT_MAX_WORDS,
    DEFAULT_STRIDE,
    entities_for_submission,
    predict_records_windowed,
)
from uzbek_ner.io.jsonl import read_jsonl_records
from uzbek_ner.io.mix import load_train_mix
from uzbek_ner.metrics.error_analysis import analyze_prediction_files
from uzbek_ner.metrics.exact_span import evaluate_prediction_files, write_metrics
from uzbek_ner.settings import REPO_ROOT, get_settings

JsonObject = dict[str, Any]
REQUIRED_GPU = "2080 Ti"


class OomSkipCounter(logging.Handler):
    """Count GLiNER Trainer's silent CUDA-OOM batch skips."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.count = 0

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if "Skipping batch due" in message:
            self.count += 1


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=settings.models / "pretrained" / "gliner_multi_v2.1",
    )
    parser.add_argument("--train", type=Path, default=settings.official_train)
    parser.add_argument(
        "--extra-train",
        type=Path,
        action="append",
        default=[],
        help="Additional JSONL (silver). Never scored on dev.",
    )
    parser.add_argument(
        "--extra-cap",
        type=int,
        default=None,
        help="Max extra records after shuffle (Mix B: cap to official train size).",
    )
    parser.add_argument("--dev", type=Path, default=settings.official_dev)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/gliner_multi_v21"))
    parser.add_argument("--run-id", default="gliner_multi_v21")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--max-words", type=int, default=DEFAULT_MAX_WORDS)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--others-lr", type=float, default=5e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-any-gpu", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    return parser.parse_args()


def assert_2080_ti(*, allow_any: bool) -> str:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    name = torch.cuda.get_device_name(0)
    if REQUIRED_GPU not in name and not allow_any:
        raise SystemExit(
            f"refusing cuda:0={name!r}; expected {REQUIRED_GPU}. "
            "Set CUDA_DEVICE_ORDER=PCI_BUS_ID and CUDA_VISIBLE_DEVICES to the "
            "2080 Ti UUID, or pass --allow-any-gpu."
        )
    logger.info("cuda:0={} device_count={}", name, torch.cuda.device_count())
    return name


def enable_encoder_checkpointing(model: GLiNER) -> str:
    try:
        backbone = model.model.token_rep_layer.bert_layer.model
    except AttributeError as error:
        return f"unavailable ({error})"
    if hasattr(backbone, "gradient_checkpointing_enable"):
        backbone.gradient_checkpointing_enable()
        if hasattr(backbone, "config"):
            backbone.config.use_cache = False
        return type(backbone).__name__
    return "no gradient_checkpointing_enable on encoder"


def write_prediction_jsonl(
    path: Path,
    records: list[JsonObject],
    predictions: dict[str, list[JsonObject]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(
                json.dumps(
                    {
                        "hash": record["hash"],
                        "text": record["text"],
                        "entities": predictions[record["hash"]],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def predict_dev(
    model: GLiNER,
    records: list[JsonObject],
    *,
    threshold: float,
    max_words: int,
    stride: int,
    batch_size: int = 8,
) -> dict[str, list[JsonObject]]:
    model.eval()
    with torch.inference_mode():
        raw, _meta = predict_records_windowed(
            model,
            records,
            max_words=max_words,
            stride=stride,
            threshold=threshold,
            batch_size=batch_size,
        )
    return {record_hash: entities_for_submission(rows) for record_hash, rows in raw.items()}


def main() -> None:
    args = parse_args()
    gpu_name = assert_2080_ti(allow_any=args.allow_any_gpu)
    if not args.model.is_dir():
        raise FileNotFoundError(f"missing GLiNER weights at {args.model}")

    train_records, mix_meta = load_train_mix(
        args.train,
        list(args.extra_train),
        extra_cap=args.extra_cap,
        seed=args.seed,
    )
    train_data = [
        row
        for record in train_records
        if (row := convert_record(record, max_words=args.max_words)) is not None
    ]
    logger.info(
        "train examples {} / {} docs (max_words={}) mix={}",
        len(train_data),
        len(train_records),
        args.max_words,
        mix_meta,
    )

    oom_counter = OomSkipCounter()
    logging.getLogger("gliner.training.trainer").addHandler(oom_counter)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = GLiNER.from_pretrained(str(args.model), local_files_only=True)
    model.to("cuda")
    ckpt_note = enable_encoder_checkpointing(model)
    logger.info("encoder checkpointing={}", ckpt_note)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    params = {
        "model": str(args.model),
        "train": str(args.train),
        "mix": mix_meta,
        "gpu": gpu_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "max_words": args.max_words,
        "stride": args.stride,
        "learning_rate": args.learning_rate,
        "others_lr": args.others_lr,
        "fp16": True,
        "bf16": False,
        "optim": "adamw_torch",
        "threshold": args.threshold,
        "seed": args.seed,
        "checkpointing": ckpt_note,
    }
    (output_dir / "train_config.json").write_text(
        json.dumps(params, indent=2) + "\n", encoding="utf-8"
    )

    model.train_model(
        train_dataset=train_data,
        eval_dataset=None,
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        max_steps=-1,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        others_lr=args.others_lr,
        fp16=True,
        bf16=False,
        optim="adamw_torch",
        dataloader_num_workers=0,
        logging_steps=20,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        warmup_ratio=args.warmup_ratio,
        seed=args.seed,
    )
    if oom_counter.count:
        logger.warning("Trainer skipped {} CUDA-OOM batches (zero loss)", oom_counter.count)

    model.save_pretrained(str(output_dir))

    if not args.skip_eval:
        dev_records = read_jsonl_records(args.dev)
        predictions = predict_dev(
            model,
            dev_records,
            threshold=args.threshold,
            max_words=args.max_words,
            stride=args.stride,
        )
        pred_path = Path(f"outputs/official/{args.run_id}_dev_predictions.jsonl")
        write_prediction_jsonl(pred_path, dev_records, predictions)
        metrics = evaluate_prediction_files(args.dev, pred_path)
        metrics["oom_skipped_batches"] = oom_counter.count
        metrics_path = Path(f"outputs/official/{args.run_id}_dev_metrics.json")
        write_metrics(metrics_path, metrics)
        diagnostics = analyze_prediction_files(args.dev, pred_path)
        eval_run = {
            "schema_version": 1,
            "run_id": args.run_id,
            "model": "urchade/gliner_multi-v2.1",
            "checkpoint": str(output_dir),
            "created_at": datetime.now(UTC).isoformat(),
            "split": "official_dev",
            "hyperparams": params,
            "metrics": metrics,
            "diagnostics": diagnostics,
            "paths": {
                "gold": str(args.dev),
                "predictions": str(pred_path.resolve()),
                "metrics": str(metrics_path.resolve()),
            },
        }
        eval_run_path = REPO_ROOT / "outputs" / "eval" / "runs" / f"{args.run_id}.json"
        eval_run_path.parent.mkdir(parents=True, exist_ok=True)
        eval_run_path.write_text(
            json.dumps(eval_run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        logger.info(
            "dev micro-F1 {:.4f} P={:.4f} R={:.4f} oom_skips={} → {}",
            metrics["micro"]["f1"],
            metrics["micro"]["precision"],
            metrics["micro"]["recall"],
            oom_counter.count,
            metrics_path,
        )


if __name__ == "__main__":
    main()

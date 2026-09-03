#!/usr/bin/env python3
"""Fine-tune a TokenClassification head on official JSONL (local checkpoints)."""

from __future__ import annotations

import argparse
import json
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    get_linear_schedule_with_warmup,
)

from uzbek_ner.io.jsonl import read_jsonl_records
from uzbek_ner.io.mix import load_train_mix
from uzbek_ner.labels import ID_TO_TAG, TAG_TO_ID, TAGS
from uzbek_ner.metrics.error_analysis import analyze_prediction_files
from uzbek_ner.metrics.exact_span import evaluate_prediction_files, write_metrics
from uzbek_ner.modeling.heads import load_token_classifier, write_head_spec
from uzbek_ner.modeling.predict import predict_records
from uzbek_ner.modeling.windows import clamp_max_length, labeled_windows
from uzbek_ner.settings import get_settings
from uzbek_ner.tracking import SMOKE_EXPERIMENT, log_json, log_metrics, log_params, setup_experiment


class WindowDataset(Dataset):
    def __init__(self, features: list[dict[str, Any]]) -> None:
        self.features = features

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict:
        row = self.features[index]
        return {
            "input_ids": row["input_ids"],
            "attention_mask": row["attention_mask"],
            "labels": row["labels"],
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=settings.models / "pretrained" / "exp1_uztext_roberta",
    )
    parser.add_argument("--train", type=Path, default=settings.official_train)
    parser.add_argument(
        "--extra-train",
        type=Path,
        action="append",
        default=[],
        help="Additional JSONL (silver). Never used as the scored split.",
    )
    parser.add_argument(
        "--extra-cap",
        type=int,
        default=None,
        help="Max extra records after shuffle (EDA Mix B: cap to official size).",
    )
    parser.add_argument("--dev", type=Path, default=settings.official_dev)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/uztext_smoke"))
    parser.add_argument("--run-id", default="uztext_smoke")
    parser.add_argument("--run-name", default=None, help="MLflow run name (default: --run-id)")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument(
        "--head",
        choices=("linear", "mlp"),
        default="linear",
        help="Token head: Linear(H,7) or Linear(H,H)→GELU→Dropout→Linear(H,7).",
    )
    parser.add_argument(
        "--mlp-hidden",
        type=int,
        default=None,
        help="MLP hidden width (default: encoder hidden_size).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = args.model.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, use_fast=True)
    model = load_token_classifier(
        model_dir,
        num_labels=len(TAGS),
        id2label=ID_TO_TAG,
        label2id=TAG_TO_ID,
        head=args.head,
        mlp_hidden=args.mlp_hidden,
    )
    logger.info(
        "token head={} classifier_params={}",
        args.head,
        sum(param.numel() for param in model.classifier.parameters()),
    )
    max_length = clamp_max_length(
        args.max_length,
        max_position_embeddings=int(model.config.max_position_embeddings),
        pad_token_id=int(model.config.pad_token_id or 0),
        model_type=str(model.config.model_type),
    )
    if max_length != args.max_length:
        logger.warning(
            "clamped max_length {} → {} ({} position table {})",
            args.max_length,
            max_length,
            model.config.model_type,
            model.config.max_position_embeddings,
        )
    model.to(device)

    train_records, mix_meta = load_train_mix(
        args.train,
        list(args.extra_train),
        extra_cap=args.extra_cap,
        seed=args.seed,
    )
    features: list[dict[str, Any]] = []
    for record in tqdm(train_records, desc="tokenize train"):
        features.extend(
            labeled_windows(
                tokenizer,
                record,
                max_length=max_length,
                stride=args.stride,
            )
        )
    logger.info(
        "Train windows: {} from {} docs (gold {} extra {}) on {}",
        len(features),
        mix_meta["train_docs"],
        mix_meta["gold_docs"],
        mix_meta["extra_docs"],
        device,
    )

    collator = DataCollatorForTokenClassification(tokenizer)
    loader = DataLoader(
        WindowDataset(features),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    steps_per_epoch = max(len(loader), 1)
    total_steps = steps_per_epoch * args.epochs
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    run_id = args.run_id
    run_name = args.run_name or run_id
    params = {
        "model": str(model_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_length": max_length,
        "stride": args.stride,
        "lr": args.learning_rate,
        "seed": args.seed,
        "windows": len(features),
        "device": str(device),
        "run_id": run_id,
        "head": args.head,
        "mlp_hidden": args.mlp_hidden,
        **mix_meta,
    }
    with setup_experiment(SMOKE_EXPERIMENT, run_name=run_name):
        log_params(params)
        global_step = 0
        for epoch in range(args.epochs):
            model.train()
            running = 0.0
            progress = tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}")
            for batch in progress:
                batch = {key: value.to(device) for key, value in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    loss = model(**batch).loss
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= scale_before:
                    scheduler.step()
                running += float(loss.item())
                global_step += 1
                progress.set_postfix(loss=f"{loss.item():.4f}")
            epoch_loss = running / steps_per_epoch
            log_metrics({"train_loss": epoch_loss}, step=epoch)
            logger.info("epoch {} loss {:.4f}", epoch + 1, epoch_loss)

        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        write_head_spec(
            output_dir,
            kind=args.head,
            mlp_hidden=int(getattr(model.config, "mlp_hidden", 0)) or None,
        )
        (output_dir / "train_config.json").write_text(
            json.dumps(params, indent=2),
            encoding="utf-8",
        )

        if not args.skip_eval:
            dev_records = read_jsonl_records(args.dev)
            predictions = predict_records(
                model,
                tokenizer,
                dev_records,
                max_length=max_length,
                stride=args.stride,
                batch_size=args.batch_size,
                device=device,
            )
            pred_path = Path(f"outputs/official/{run_id}_dev_predictions.jsonl")
            pred_path.parent.mkdir(parents=True, exist_ok=True)
            with pred_path.open("w", encoding="utf-8") as stream:
                for row in predictions:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            metrics = evaluate_prediction_files(args.dev, pred_path)
            metrics_path = Path(f"outputs/official/{run_id}_dev_metrics.json")
            write_metrics(metrics_path, metrics)
            diagnostics = analyze_prediction_files(args.dev, pred_path)
            eval_run = {
                "schema_version": 1,
                "run_id": run_id,
                "model": "rifkat/uztext-3Gb-BPE-Roberta",
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
            eval_run_path = Path(f"outputs/eval/runs/{run_id}.json")
            eval_run_path.parent.mkdir(parents=True, exist_ok=True)
            eval_run_path.write_text(
                json.dumps(eval_run, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            log_json(eval_run, "evaluate", "eval_run.json")
            log_metrics(
                {
                    "micro_f1": float(metrics["micro"]["f1"]),
                    "micro_precision": float(metrics["micro"]["precision"]),
                    "micro_recall": float(metrics["micro"]["recall"]),
                    "org_f1": float(metrics["by_label"]["ORG"]["f1"]),
                    "name_f1": float(metrics["by_label"]["NAME"]["f1"]),
                    "geo_f1": float(metrics["by_label"]["GEO"]["f1"]),
                }
            )
            logger.info("dev micro-F1 {:.4f} → {}", metrics["micro"]["f1"], metrics_path)

    logger.info("saved {}", output_dir)


if __name__ == "__main__":
    main()

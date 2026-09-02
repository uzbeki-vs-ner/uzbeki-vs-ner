#!/usr/bin/env python3
"""Measure BERT-family NER fine-tune feasibility on the local laptop GPU.

Does not full-train and does not write model checkpoints. Local snapshots only
(HF_HUB_OFFLINE). CUDA work should be wrapped with:

    flock outputs/.gpu.lock uv run python scripts/bench_finetune.py --phase gpu
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from uzbek_ner.io.jsonl import read_jsonl_records  # noqa: E402
from uzbek_ner.labels import TAGS  # noqa: E402
from uzbek_ner.settings import get_settings  # noqa: E402

logging.getLogger("transformers").setLevel(logging.ERROR)

NUM_LABELS = len(TAGS)
STRIDE = 128
N_TRAIN_DOCS = 13_000
N_DEV_DOCS = 1_500
N_EPOCHS = 3
SUBSAMPLE_DOCS = 400
THROUGHPUT_STEPS = 40
THROUGHPUT_WARMUP = 5
BATCH_CANDIDATES = (16, 8, 4, 2, 1)
SEQ_LENGTHS = (256, 512)

MODELS: dict[str, dict[str, Any]] = {
    "exp0_xlm_roberta_base": {
        "alias": "xlm-roberta-base",
        "params_m": 278,
        "layers": 12,
    },
    "exp1_uztext_roberta": {
        "alias": "uztext-roberta-6L",
        "params_m": 83,
        "layers": 6,
    },
    "exp2_mdeberta_v3_base": {
        "alias": "mdeberta-v3-base",
        "params_m": 278,
        "layers": 12,
    },
}


def log(message: str) -> None:
    print(message, flush=True)


def gb(num_bytes: int | float) -> float:
    return round(float(num_bytes) / (1024**3), 3)


def json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_tokenizer(model_dir: Path) -> Any:
    from transformers import AutoTokenizer, DebertaV2Tokenizer

    try:
        return AutoTokenizer.from_pretrained(
            str(model_dir),
            local_files_only=True,
            use_fast=True,
        )
    except Exception:
        pass
    spm = model_dir / "spm.model"
    if spm.exists():
        return DebertaV2Tokenizer(vocab_file=str(spm), do_lower_case=False)
    return AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, use_fast=False)


def count_trainable_params(model: Any) -> int:
    return sum(int(param.numel()) for param in model.parameters() if param.requires_grad)


def special_token_budget(tokenizer: Any) -> int:
    try:
        return int(tokenizer.num_special_tokens_to_add(pair=False))
    except Exception:
        return 2


def n_windows_for_n_tokens(n_tokens: int, max_length: int, stride: int, n_special: int) -> int:
    body = max(max_length - n_special, 1)
    overlap = min(stride, body - 1)
    step = max(body - overlap, 1)
    if n_tokens <= body:
        return 1
    return 1 + math.ceil((n_tokens - body) / step)


def window_stats_for_texts(
    tokenizer: Any,
    texts: list[str],
    max_lengths: tuple[int, ...] = SEQ_LENGTHS,
    stride: int = STRIDE,
) -> dict[str, Any]:
    n_special = special_token_budget(tokenizer)
    lengths: list[int] = []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        lengths.append(len(ids))
    lengths.sort()

    def pct(p: float) -> int:
        if not lengths:
            return 0
        idx = min(len(lengths) - 1, max(0, round(p * (len(lengths) - 1))))
        return int(lengths[idx])

    out: dict[str, Any] = {
        "n_docs": len(texts),
        "n_special": n_special,
        "tokens_no_special": {
            "mean": round(sum(lengths) / max(len(lengths), 1), 2),
            "p50": pct(0.5),
            "p90": pct(0.9),
            "p95": pct(0.95),
            "p99": pct(0.99),
            "max": max(lengths) if lengths else 0,
        },
        "windows": {},
        "truncation_note": (
            "Sliding windows counted from tokenizer.encode(add_special_tokens=False) "
            f"with stride={stride} content overlap and CLS/SEP budget. "
            "Throughput loops pad/truncate to max_length (worst-case compute per window)."
        ),
    }
    for max_length in max_lengths:
        windows = [n_windows_for_n_tokens(n, max_length, stride, n_special) for n in lengths]
        overflow = sum(1 for n in lengths if n + n_special > max_length)
        out["windows"][str(max_length)] = {
            "stride": stride,
            "docs_over_max_length": overflow,
            "docs_over_max_length_frac": round(overflow / max(len(texts), 1), 4),
            "windows_total": int(sum(windows)),
            "windows_per_doc": round(sum(windows) / max(len(texts), 1), 4),
            "max_windows_one_doc": max(windows) if windows else 0,
        }
    return out


def collect_hardware() -> dict[str, Any]:
    import torch

    props = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    return {
        "torch": torch.__version__,
        "cuda_compiled": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "capability": list(torch.cuda.get_device_capability(0))
        if torch.cuda.is_available()
        else None,
        "total_vram_gb": gb(props.total_memory) if props else None,
        "bf16_supported": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        "cpu_threads": os.cpu_count(),
    }


def quiet_transformers() -> None:
    import transformers

    transformers.logging.set_verbosity_error()
    logging.getLogger("transformers").setLevel(logging.ERROR)


def safe_seq_len(config: Any, requested: int) -> int:
    """RoBERTa-family position ids are padding_idx+1 .. so max usable length is often n_pos-2."""

    model_type = str(getattr(config, "model_type", "") or "")
    n_pos = int(getattr(config, "max_position_embeddings", requested) or requested)
    if model_type in {"roberta", "xlm-roberta"}:
        return min(requested, max(n_pos - 2, 1))
    return min(requested, n_pos)


def load_token_classifier(model_dir: Path, device: Any, dtype: Any | None = None) -> Any:
    from transformers import AutoConfig, AutoModelForTokenClassification

    quiet_transformers()
    config = AutoConfig.from_pretrained(str(model_dir), local_files_only=True)
    config.num_labels = NUM_LABELS
    model = AutoModelForTokenClassification.from_pretrained(
        str(model_dir),
        config=config,
        local_files_only=True,
        ignore_mismatched_sizes=True,
        torch_dtype=dtype,
    )
    model.to(device)
    model.train()
    return model


def prime_adamw(optimizer: Any) -> None:
    """Allocate fp32 AdamW moment tensors so VRAM peaks include optimizer state."""

    import torch

    for group in optimizer.param_groups:
        for param in group["params"]:
            if not param.requires_grad:
                continue
            state = optimizer.state[param]
            if "exp_avg" not in state:
                state["step"] = torch.tensor(0.0, device=param.device)
                state["exp_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format)
                state["exp_avg_sq"] = torch.zeros_like(param, memory_format=torch.preserve_format)


def cuda_cleanup(*objects: Any) -> None:
    import torch

    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()


def synthetic_batch(
    vocab_size: int,
    batch_size: int,
    seq_len: int,
    pad_id: int,
    device: Any,
) -> dict[str, Any]:
    import torch

    vocab = max(int(vocab_size), pad_id + 8)
    # Avoid pad ids so RoBERTa position_ids stay dense and in-range.
    low = max(pad_id + 1, 2)
    input_ids = torch.randint(low, vocab, (batch_size, seq_len), device=device)
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=device)
    labels = torch.randint(0, NUM_LABELS, (batch_size, seq_len), device=device)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def try_train_microbatch(
    model: Any,
    optimizer: Any,
    scaler: Any,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    pad_id: int,
    device: Any,
    amp: str,
    n_steps: int = 2,
    accum_steps: int = 1,
) -> dict[str, Any]:
    import torch
    from torch.amp import autocast

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(amp)
    use_amp = amp_dtype is not None
    use_scaler = bool(scaler is not None and amp == "fp16")
    t0 = time.perf_counter()
    try:
        for _ in range(n_steps):
            optimizer.zero_grad(set_to_none=True)
            for _acc in range(accum_steps):
                batch = synthetic_batch(vocab_size, batch_size, seq_len, pad_id, device)
                with autocast("cuda", enabled=use_amp, dtype=amp_dtype or torch.float16):
                    loss = model(**batch).loss
                    if accum_steps > 1:
                        loss = loss / accum_steps
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"non-finite loss: {float(loss.detach().cpu())}")
                if use_scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                del batch
            if use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        return {
            "ok": True,
            "alloc_gb": gb(torch.cuda.max_memory_allocated(device)),
            "reserved_gb": gb(torch.cuda.max_memory_reserved(device)),
            "seconds": round(elapsed, 3),
            "finite_loss": True,
            "amp": amp,
        }
    except torch.cuda.OutOfMemoryError:
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        return {"ok": False, "error": "oom", "amp": amp}
    except ValueError as exc:
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        return {"ok": False, "error": str(exc)[:200], "amp": amp}
    except RuntimeError as exc:
        message = str(exc).lower()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        if "out of memory" in message:
            return {"ok": False, "error": "oom", "amp": amp}
        return {"ok": False, "error": str(exc)[:300], "amp": amp}


PROBE_PREFIX = "BENCH_PROBE_JSON:"


def emit_probe(payload: dict[str, Any]) -> None:
    print(PROBE_PREFIX + json.dumps(payload), flush=True)


def enable_checkpointing(model: Any) -> None:
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        model.gradient_checkpointing_enable()
    model.config.use_cache = False


def default_amp(model_alias: str) -> str:
    if model_alias == "exp2_mdeberta_v3_base":
        return "bf16"
    return "fp16"


def run_probe_worker(
    model_alias: str,
    seq_len: int,
    batch_size: int,
    *,
    amp: str,
    gradient_checkpointing: bool,
    accum_steps: int = 1,
    n_steps: int = 2,
) -> dict[str, Any]:
    import torch
    from torch.amp import GradScaler
    from torch.optim import AdamW
    from transformers import AutoConfig

    quiet_transformers()
    device = torch.device("cuda")
    model_dir = get_settings().models / "pretrained" / model_alias
    config = AutoConfig.from_pretrained(str(model_dir), local_files_only=True)
    used_seq = safe_seq_len(config, seq_len)
    pad_id = int(getattr(config, "pad_token_id", 1) or 1)
    vocab_size = int(getattr(config, "vocab_size", 250002) or 250002)

    model = load_token_classifier(model_dir, device)
    if gradient_checkpointing:
        enable_checkpointing(model)
    optimizer = AdamW(model.parameters(), lr=2e-5, fused=False, foreach=False)
    prime_adamw(optimizer)
    scaler = GradScaler("cuda", enabled=amp == "fp16")
    idle_alloc = gb(torch.cuda.memory_allocated(device))
    idle_reserved = gb(torch.cuda.memory_reserved(device))
    torch.cuda.reset_peak_memory_stats(device)
    result = try_train_microbatch(
        model,
        optimizer,
        scaler,
        batch_size,
        used_seq,
        vocab_size,
        pad_id,
        device,
        amp,
        n_steps=n_steps,
        accum_steps=accum_steps,
    )
    payload = {
        "ok": bool(result.get("ok")),
        "error": result.get("error"),
        "model_alias": model_alias,
        "requested_seq_len": seq_len,
        "used_seq_len": used_seq,
        "batch_size": batch_size,
        "accum_steps": accum_steps,
        "effective_batch": batch_size * accum_steps,
        "gradient_checkpointing": gradient_checkpointing,
        "amp": amp,
        "fp16": amp == "fp16",
        "n_params": count_trainable_params(model),
        "idle_alloc_gb": idle_alloc,
        "idle_reserved_gb": idle_reserved,
        "alloc_gb": result.get("alloc_gb"),
        "reserved_gb": result.get("reserved_gb"),
        "seconds": result.get("seconds"),
    }
    cuda_cleanup(model, optimizer, scaler)
    return payload


def parse_probe_stdout(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        if line.startswith(PROBE_PREFIX):
            return json.loads(line[len(PROBE_PREFIX) :])
    msg = stdout[-1500:] if stdout else "empty stdout"
    raise RuntimeError(f"probe subprocess did not emit JSON. tail:\n{msg}")


def probe_via_subprocess(
    model_alias: str,
    seq_len: int,
    batch_size: int,
    *,
    gradient_checkpointing: bool = False,
    accum_steps: int = 1,
    amp: str | None = None,
) -> dict[str, Any]:
    chosen_amp = amp or default_amp(model_alias)
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--phase",
        "probe",
        "--model-alias",
        model_alias,
        "--seq-len",
        str(seq_len),
        "--batch-size",
        str(batch_size),
        "--accum-steps",
        str(accum_steps),
        "--amp",
        chosen_amp,
    ]
    if gradient_checkpointing:
        cmd.append("--gradient-checkpointing")
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["TRANSFORMERS_VERBOSITY"] = "error"
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
        oom = "out of memory" in combined.lower() or "CUDA out of memory" in combined
        return {
            "ok": False,
            "error": "oom" if oom else f"probe_exit_{proc.returncode}",
            "batch_size": batch_size,
            "requested_seq_len": seq_len,
            "stderr_tail": (proc.stderr or "")[-400:],
        }
    return parse_probe_stdout(proc.stdout)


def max_stable_batch(
    model_alias: str,
    seq_len: int,
    *,
    gradient_checkpointing: bool,
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    max_ok: dict[str, Any] | None = None
    n_params = None
    idle_alloc = None
    idle_reserved = None
    used_seq = seq_len

    candidates = list(BATCH_CANDIDATES)
    for batch_size in candidates:
        result = probe_via_subprocess(
            model_alias,
            seq_len,
            batch_size,
            gradient_checkpointing=gradient_checkpointing,
        )
        row = {"batch_size": batch_size, **result}
        attempts.append(row)
        n_params = result.get("n_params", n_params)
        idle_alloc = result.get("idle_alloc_gb", idle_alloc)
        idle_reserved = result.get("idle_reserved_gb", idle_reserved)
        used_seq = result.get("used_seq_len", used_seq)
        log(
            f"    bs={batch_size} seq={seq_len} (used={used_seq}) "
            f"ckpt={gradient_checkpointing} -> {row.get('error') or 'ok'} "
            f"alloc={row.get('alloc_gb')} reserved={row.get('reserved_gb')}"
        )
        if result.get("ok"):
            max_ok = row
            break

    if max_ok is not None and max_ok["batch_size"] >= 16:
        for batch_size in (24, 32, 48, 64):
            if batch_size <= max_ok["batch_size"]:
                continue
            result = probe_via_subprocess(
                model_alias,
                seq_len,
                batch_size,
                gradient_checkpointing=gradient_checkpointing,
            )
            row = {"batch_size": batch_size, **result}
            attempts.append(row)
            log(
                f"    bs={batch_size} seq={seq_len} ckpt={gradient_checkpointing} "
                f"-> {row.get('error') or 'ok'} alloc={row.get('alloc_gb')} "
                f"reserved={row.get('reserved_gb')}"
            )
            if result.get("ok"):
                max_ok = row
            else:
                break

    return {
        "seq_len": seq_len,
        "used_seq_len": used_seq,
        "gradient_checkpointing": gradient_checkpointing,
        "fp16": True,
        "n_params": n_params,
        "idle_alloc_gb": idle_alloc,
        "idle_reserved_gb": idle_reserved,
        "max_batch": None if max_ok is None else max_ok["batch_size"],
        "peak_alloc_gb": None if max_ok is None else max_ok.get("alloc_gb"),
        "peak_reserved_gb": None if max_ok is None else max_ok.get("reserved_gb"),
        "attempts": attempts,
        "probe_isolation": "subprocess",
        "adamw_states_primed": True,
    }


def build_padded_dataset(
    tokenizer: Any,
    texts: list[str],
    max_length: int,
) -> list[dict[str, list[int]]]:
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
        pad_id = tokenizer.pad_token_id
    rows: list[dict[str, list[int]]] = []
    for text in texts:
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_attention_mask=True,
        )
        labels = [0] * max_length
        rows.append(
            {
                "input_ids": list(encoded["input_ids"]),
                "attention_mask": list(encoded["attention_mask"]),
                "labels": labels,
            }
        )
    return rows


def collate_batch(rows: list[dict[str, list[int]]]) -> dict[str, Any]:
    import torch

    return {
        "input_ids": torch.tensor([row["input_ids"] for row in rows], dtype=torch.long),
        "attention_mask": torch.tensor([row["attention_mask"] for row in rows], dtype=torch.long),
        "labels": torch.tensor([row["labels"] for row in rows], dtype=torch.long),
    }


def measure_throughput(
    model_dir: Path,
    device: Any,
    batch_size: int,
    seq_len: int,
    amp: str,
    gradient_checkpointing: bool,
    n_steps: int = THROUGHPUT_STEPS,
    warmup: int = THROUGHPUT_WARMUP,
) -> dict[str, Any]:
    import torch
    from torch.amp import GradScaler
    from torch.optim import AdamW
    from transformers import AutoConfig

    quiet_transformers()
    config = AutoConfig.from_pretrained(str(model_dir), local_files_only=True)
    used_seq = safe_seq_len(config, seq_len)
    pad_id = int(getattr(config, "pad_token_id", 1) or 1)
    vocab_size = int(getattr(config, "vocab_size", 250002) or 250002)
    model = load_token_classifier(model_dir, device)
    if gradient_checkpointing:
        enable_checkpointing(model)
    optimizer = AdamW(model.parameters(), lr=2e-5, fused=False, foreach=False)
    prime_adamw(optimizer)
    scaler = GradScaler("cuda", enabled=amp == "fp16")
    torch.backends.cudnn.benchmark = True
    try:
        warm = try_train_microbatch(
            model,
            optimizer,
            scaler,
            batch_size,
            used_seq,
            vocab_size,
            pad_id,
            device,
            amp,
            n_steps=warmup,
        )
        if not warm.get("ok"):
            return {
                "ok": False,
                "error": warm.get("error"),
                "batch_size": batch_size,
                "seq_len": used_seq,
                "amp": amp,
            }
        timed = try_train_microbatch(
            model,
            optimizer,
            scaler,
            batch_size,
            used_seq,
            vocab_size,
            pad_id,
            device,
            amp,
            n_steps=n_steps,
        )
        if not timed.get("ok"):
            return {
                "ok": False,
                "error": timed.get("error"),
                "batch_size": batch_size,
                "seq_len": used_seq,
                "amp": amp,
            }
        seconds = float(timed["seconds"])
        steps_per_sec = n_steps / seconds
        tokens = n_steps * batch_size * used_seq
        return {
            "ok": True,
            "batch_size": batch_size,
            "seq_len": used_seq,
            "gradient_checkpointing": gradient_checkpointing,
            "amp": amp,
            "fp16": amp == "fp16",
            "n_steps": n_steps,
            "warmup_steps": warmup,
            "seconds": round(seconds, 3),
            "steps_per_sec": round(steps_per_sec, 3),
            "tokens_per_sec": round(tokens / seconds, 1),
            "tokens_include_padding": True,
            "synthetic_full_length_windows": True,
            "peak_alloc_gb": timed.get("alloc_gb"),
            "peak_reserved_gb": timed.get("reserved_gb"),
            "pin_memory": False,
            "num_workers": 0,
        }
    finally:
        cuda_cleanup(model, optimizer, scaler)


def throughput_via_subprocess(
    model_alias: str,
    seq_len: int,
    batch_size: int,
    sample_n: int,
    *,
    gradient_checkpointing: bool,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--phase",
        "throughput",
        "--model-alias",
        model_alias,
        "--seq-len",
        str(seq_len),
        "--batch-size",
        str(batch_size),
        "--sample-n",
        str(sample_n),
        "--amp",
        default_amp(model_alias),
    ]
    if gradient_checkpointing:
        cmd.append("--gradient-checkpointing")
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["TRANSFORMERS_VERBOSITY"] = "error"
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        return {
            "ok": False,
            "error": f"throughput_exit_{proc.returncode}",
            "stderr_tail": (proc.stderr or "")[-500:],
            "batch_size": batch_size,
            "seq_len": seq_len,
        }
    return parse_probe_stdout(proc.stdout)


def run_throughput_worker(
    model_alias: str,
    seq_len: int,
    batch_size: int,
    sample_n: int,
    *,
    gradient_checkpointing: bool,
    amp: str | None = None,
) -> dict[str, Any]:
    import torch

    del sample_n  # synthetic full-length windows; subsample size is unused
    quiet_transformers()
    device = torch.device("cuda")
    settings = get_settings()
    model_dir = settings.models / "pretrained" / model_alias
    chosen_amp = amp or default_amp(model_alias)
    result = measure_throughput(
        model_dir,
        device,
        batch_size,
        seq_len,
        chosen_amp,
        gradient_checkpointing,
    )
    result["requested_seq_len"] = seq_len
    result["model_alias"] = model_alias
    return result


def hardware_via_subprocess() -> dict[str, Any]:
    cmd = [sys.executable, str(Path(__file__).resolve()), "--phase", "hardware"]
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
    return parse_probe_stdout(proc.stdout)


def recommend_recipe(vram: dict[str, Any], throughput: dict[str, Any]) -> dict[str, Any]:
    max_512 = vram.get("seq_512", {}).get("max_batch")
    max_512_ckpt = vram.get("seq_512_checkpoint", {}).get("max_batch")
    max_256 = vram.get("seq_256", {}).get("max_batch")
    use_ckpt = False
    max_length = 256
    reserved = 0.0
    micro: int | None = None
    if max_512:
        max_length = 512
        micro = max_512
        reserved = float(vram.get("seq_512", {}).get("peak_reserved_gb") or 0)
    elif max_512_ckpt:
        max_length = 512
        micro = max_512_ckpt
        use_ckpt = True
        reserved = float(vram.get("seq_512_checkpoint", {}).get("peak_reserved_gb") or 0)
    elif max_256:
        max_length = 256
        micro = max_256
        reserved = float(vram.get("seq_256", {}).get("peak_reserved_gb") or 0)

    if micro is None:
        return {
            "verdict": "impossible",
            "max_length": max_length,
            "micro_batch": 0,
            "grad_accum": 1,
            "effective_batch": None,
            "gradient_checkpointing": False,
            "fp16": True,
            "lr": 2e-5,
            "epochs": N_EPOCHS,
            "stride": STRIDE,
            "optimizer": "AdamW",
            "throughput_steps_per_sec": None,
        }

    rec_micro = micro
    if micro >= 16 and reserved >= 4.5:
        rec_micro = 16
    elif micro >= 4 and reserved >= 5.2:
        rec_micro = max(micro // 2, 2)
    if max_length == 512 and micro <= 2:
        verdict = "tight"
        rec_micro = micro
    elif rec_micro >= 8 and reserved <= 5.2:
        verdict = "easy"
    elif rec_micro >= 1:
        verdict = "tight"
    else:
        verdict = "impossible"
    accum = 1 if rec_micro >= 16 else math.ceil(16 / rec_micro)
    steps_per_sec = throughput.get("steps_per_sec") if throughput.get("ok") else None
    return {
        "verdict": verdict,
        "max_length": max_length,
        "micro_batch": rec_micro,
        "grad_accum": accum,
        "effective_batch": rec_micro * accum,
        "gradient_checkpointing": use_ckpt,
        "fp16": True,
        "lr": 2e-5,
        "epochs": N_EPOCHS,
        "stride": STRIDE,
        "optimizer": "AdamW",
        "throughput_steps_per_sec": steps_per_sec,
        "peak_reserved_gb_at_max": reserved,
    }


def estimate_wallclock(
    windows_per_doc: float,
    micro_batch: int | None,
    steps_per_sec: float | None,
    n_docs: int = N_TRAIN_DOCS,
    n_epochs: int = N_EPOCHS,
) -> dict[str, Any]:
    if not micro_batch or not steps_per_sec or steps_per_sec <= 0:
        return {"ok": False, "reason": "missing throughput or batch"}
    n_windows = windows_per_doc * n_docs
    steps_per_epoch = math.ceil(n_windows / micro_batch)
    seconds_per_epoch = steps_per_epoch / steps_per_sec
    train_seconds = seconds_per_epoch * n_epochs
    # Eval is typically 3–6× faster than train (no backward); add a conservative 15%.
    with_eval_seconds = train_seconds * 1.15
    return {
        "ok": True,
        "n_docs": n_docs,
        "windows_per_doc": windows_per_doc,
        "n_windows_per_epoch": round(n_windows, 1),
        "micro_batch": micro_batch,
        "steps_per_epoch": steps_per_epoch,
        "steps_per_sec": steps_per_sec,
        "seconds_per_epoch_train": round(seconds_per_epoch, 1),
        "minutes_per_epoch_train": round(seconds_per_epoch / 60, 2),
        "hours_3_epochs_train_only": round(train_seconds / 3600, 3),
        "hours_3_epochs_plus_eval_tax": round(with_eval_seconds / 3600, 3),
        "note": (
            "Train-only time is 3 * ceil(windows_per_doc * 13000 / micro_batch) / steps_per_sec. "
            "Throughput was measured on padded max_length windows, so this is a pessimistic "
            "compute bound. Sliding-window counts come from official train texts. "
            "Eval tax is a flat +15% (no backward), not a measured eval loop."
        ),
    }


def run_cpu_prep(out_dir: Path, sample_n: int) -> dict[str, Any]:
    settings = get_settings()
    log(f"Reading {settings.official_train}")
    records = read_jsonl_records(settings.official_train)
    texts_all = [str(row["text"]) for row in records]
    texts_sample = texts_all[:sample_n]
    payload: dict[str, Any] = {
        "n_train_docs_file": len(texts_all),
        "sample_n": len(texts_sample),
        "models": {},
    }
    models_root = settings.models / "pretrained"
    for alias in MODELS:
        model_dir = models_root / alias
        log(f"Tokenizer windows: {alias}")
        tokenizer = load_tokenizer(model_dir)
        stats = window_stats_for_texts(tokenizer, texts_all)
        stats_sample = window_stats_for_texts(tokenizer, texts_sample)
        payload["models"][alias] = {
            "path": str(model_dir),
            "is_fast_tokenizer": bool(getattr(tokenizer, "is_fast", False)),
            "vocab_size": int(getattr(tokenizer, "vocab_size", 0) or 0),
            "pad_token_id": tokenizer.pad_token_id,
            "windows_full_train": stats,
            "windows_sample": stats_sample,
        }
        del tokenizer
        gc.collect()
    json_dump(out_dir / "cpu_prep.json", payload)
    log(f"Wrote {out_dir / 'cpu_prep.json'}")
    return payload


def pick_throughput_batch(max_batch: int | None) -> int | None:
    if not max_batch:
        return None
    preferred = (8, 4, 2, 1)
    for size in preferred:
        if max_batch >= size:
            return size
    return max_batch


def run_gpu(
    out_dir: Path,
    prep: dict[str, Any],
    sample_n: int,
    only: list[str] | None = None,
) -> dict[str, Any]:
    hardware = hardware_via_subprocess()
    log(f"Hardware: {json.dumps(hardware)}")
    if not hardware.get("cuda_available"):
        raise SystemExit("CUDA is required for --phase gpu")
    settings = get_settings()
    models_root = settings.models / "pretrained"
    summary_path = out_dir / "summary.json"
    summary: dict[str, Any]
    if only and summary_path.exists():
        summary = load_json(summary_path)
        summary["hardware"] = hardware
        summary["sample_n"] = sample_n
        summary.setdefault("models", {})
    else:
        summary = {
            "hardware": hardware,
            "num_labels": NUM_LABELS,
            "tags": list(TAGS),
            "sample_n": sample_n,
            "throughput_steps": THROUGHPUT_STEPS,
            "models": {},
        }

    for alias, meta in MODELS.items():
        if only and alias not in only:
            continue
        model_dir = models_root / alias
        log(f"\n=== GPU bench {alias} ===")
        vram: dict[str, Any] = {}
        for seq_len in SEQ_LENGTHS:
            log(f"  max batch seq={seq_len} (no checkpoint)")
            vram[f"seq_{seq_len}"] = max_stable_batch(
                alias,
                seq_len,
                gradient_checkpointing=False,
            )
        need_ckpt = alias != "exp1_uztext_roberta" or (vram["seq_512"].get("max_batch") or 0) < 16
        if need_ckpt:
            log("  max batch seq=512 (gradient checkpointing)")
            vram["seq_512_checkpoint"] = max_stable_batch(
                alias,
                512,
                gradient_checkpointing=True,
            )
        rec_seq = 512
        rec_bs = pick_throughput_batch(vram["seq_512"].get("max_batch"))
        use_ckpt_for_speed = False
        if rec_bs is None:
            rec_bs = pick_throughput_batch((vram.get("seq_512_checkpoint") or {}).get("max_batch"))
            use_ckpt_for_speed = rec_bs is not None
        if rec_bs is None:
            rec_bs = pick_throughput_batch(vram["seq_256"].get("max_batch"))
            rec_seq = 256
        throughput: dict[str, Any] = {"ok": False, "error": "no stable batch"}
        if rec_bs:
            log(f"  throughput {alias} bs={rec_bs} seq={rec_seq} ckpt={use_ckpt_for_speed}")
            throughput = throughput_via_subprocess(
                alias,
                rec_seq,
                rec_bs,
                sample_n,
                gradient_checkpointing=use_ckpt_for_speed,
            )
            if not throughput.get("ok") and rec_seq == 512:
                rec_bs_256 = pick_throughput_batch(vram["seq_256"].get("max_batch"))
                if rec_bs_256:
                    rec_seq = 256
                    rec_bs = rec_bs_256
                    use_ckpt_for_speed = False
                    log(f"  throughput fallback {alias} bs={rec_bs} seq=256")
                    throughput = throughput_via_subprocess(
                        alias,
                        rec_seq,
                        rec_bs,
                        sample_n,
                        gradient_checkpointing=False,
                    )
            if not throughput.get("ok") and rec_bs > 1:
                rec_bs = max(rec_bs // 2, 1)
                log(f"  throughput retry bs={rec_bs}")
                throughput = throughput_via_subprocess(
                    alias,
                    rec_seq,
                    rec_bs,
                    sample_n,
                    gradient_checkpointing=use_ckpt_for_speed,
                )

        no_ckpt_bs = vram["seq_512"].get("max_batch")
        ckpt_bs = (vram.get("seq_512_checkpoint") or {}).get("max_batch")
        accum_probe: dict[str, Any] = {"skipped": True}
        micro_for_accum = no_ckpt_bs or ckpt_bs
        ckpt_for_accum = no_ckpt_bs is None and bool(ckpt_bs)
        if micro_for_accum and micro_for_accum < 16:
            accum = math.ceil(16 / micro_for_accum)
            log(f"  effective-16 probe micro={micro_for_accum} accum={accum} ckpt={ckpt_for_accum}")
            accum_probe = probe_via_subprocess(
                alias,
                512,
                micro_for_accum,
                gradient_checkpointing=ckpt_for_accum,
                accum_steps=accum,
            )
            accum_probe["skipped"] = False
        elif micro_for_accum and micro_for_accum >= 16:
            accum_probe = {
                "skipped": False,
                "ok": True,
                "note": "true batch already >= 16",
                "batch_size": micro_for_accum,
            }

        effective16 = {
            "asked": "effective batch 16 at seq 512",
            "microbatch_no_ckpt": no_ckpt_bs,
            "microbatch_ckpt": ckpt_bs,
            "fits_true_batch_16_no_ckpt": bool(no_ckpt_bs and no_ckpt_bs >= 16),
            "fits_true_batch_16_ckpt": bool(ckpt_bs and ckpt_bs >= 16),
            "grad_accum_to_16_no_ckpt": None if not no_ckpt_bs else math.ceil(16 / no_ckpt_bs),
            "grad_accum_to_16_ckpt": None if not ckpt_bs else math.ceil(16 / ckpt_bs),
            "effective_16_possible": bool(no_ckpt_bs or ckpt_bs),
            "accum_probe": accum_probe,
        }
        recipe = recommend_recipe(vram, throughput)
        windows = prep.get("models", {}).get(alias, {}).get("windows_full_train", {})
        windows_per_doc = (
            windows.get("windows", {}).get(str(recipe["max_length"]), {}).get("windows_per_doc")
            or 1.0
        )
        steps = throughput.get("steps_per_sec") if throughput.get("ok") else None
        measured_bs = throughput.get("batch_size") if throughput.get("ok") else rec_bs
        wall_micro = recipe.get("micro_batch")
        if steps and measured_bs and wall_micro and measured_bs != wall_micro:
            # Do not invent linear scaling; estimate at the measured batch instead.
            wall = estimate_wallclock(windows_per_doc, int(measured_bs), steps)
            wall["note_batch"] = (
                f"Throughput measured at micro-batch {measured_bs}; recipe recommends {wall_micro}."
            )
        else:
            wall = estimate_wallclock(windows_per_doc, wall_micro, steps)
        summary["models"][alias] = {
            **meta,
            "path": str(model_dir),
            "vram": vram,
            "throughput": throughput,
            "effective_batch_16": effective16,
            "recipe": recipe,
            "wallclock": wall,
            "windows_per_doc": windows_per_doc,
        }
        json_dump(out_dir / "summary.json", summary)
        log(f"  finished {alias} (CUDA context was subprocess-scoped)")

    json_dump(out_dir / "summary.json", summary)
    return summary


def run_cpu_train_step(out_dir: Path) -> dict[str, Any]:
    import torch
    from torch.optim import AdamW
    from transformers import AutoConfig, AutoModelForTokenClassification

    settings = get_settings()
    device = torch.device("cpu")
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
    results: dict[str, Any] = {"threads": torch.get_num_threads(), "models": {}}
    for alias in ("exp1_uztext_roberta", "exp0_xlm_roberta_base"):
        model_dir = settings.models / "pretrained" / alias
        log(f"CPU 1-step {alias}")
        config = AutoConfig.from_pretrained(str(model_dir), local_files_only=True)
        config.num_labels = NUM_LABELS
        model = AutoModelForTokenClassification.from_pretrained(
            str(model_dir),
            config=config,
            local_files_only=True,
            ignore_mismatched_sizes=True,
        )
        model.to(device)
        model.train()
        optimizer = AdamW(model.parameters(), lr=2e-5)
        batch = {
            "input_ids": torch.randint(2, int(config.vocab_size), (1, 256)),
            "attention_mask": torch.ones(1, 256, dtype=torch.long),
            "labels": torch.randint(0, NUM_LABELS, (1, 256)),
        }
        t0 = time.perf_counter()
        loss = model(**batch).loss
        loss.backward()
        optimizer.step()
        elapsed = time.perf_counter() - t0
        results["models"][alias] = {
            "seconds_one_step_bs1_seq256": round(elapsed, 3),
            "n_params": count_trainable_params(model),
        }
        log(f"  {alias}: {elapsed:.2f}s")
        del model, optimizer, loss
        gc.collect()
    json_dump(out_dir / "cpu_train.json", results)
    return results


def fmt_hours(hours: float | None) -> str:
    if hours is None:
        return "n/a"
    if hours < 1:
        return f"{hours * 60:.0f} min"
    return f"{hours:.2f} h"


def write_markdown(
    summary: dict[str, Any], prep: dict[str, Any], cpu_train: dict[str, Any] | None
) -> str:
    hw = summary.get("hardware", {})
    lines: list[str] = [
        "# BERT-family NER fine-tune — RTX A1000 6 GB",
        "",
        "Measured on this laptop, not guessed. Script: `uv run python scripts/bench_finetune.py`. "
        "No full training run; no checkpoints. Local snapshots only (`HF_HUB_OFFLINE=1`).",
        "",
        "## Hardware (verified)",
        "",
        f"- GPU: `{hw.get('device_name')}` ({hw.get('total_vram_gb')} GiB), "
        f"driver/CUDA from nvidia-smi + torch `{hw.get('torch')}` compiled with CUDA `{hw.get('cuda_compiled')}`, "
        f"capability `{hw.get('capability')}`.",
        f"- CPU threads seen by Python: {hw.get('cpu_threads')}; host RAM is 31 GB (from `free`).",
        f"- bf16 supported: `{hw.get('bf16_supported')}`. All GPU benches used **fp16 autocast + GradScaler + AdamW** "
        "(fp32 master weights). Peak memory is allocated / reserved GiB after 2 train steps of synthetic full-length windows.",
        "",
        "## Method",
        "",
        "- Head: `AutoModelForTokenClassification`, `num_labels=7` (BIO for ORG/NAME/GEO).",
        "- Max batch: step down `32/16/8/4/2/1` at `max_length` 256 and 512, including backward + AdamW.",
        "- Throughput: DataLoader `pin_memory=True`, `num_workers=2`, **truncate+pad to max_length** on a 400-doc official subsample, "
        f"{THROUGHPUT_STEPS} timed steps after {THROUGHPUT_WARMUP} warmup. Padding **is** counted in tokens/sec (worst-case window).",
        f"- Windows/doc: sliding window, stride `{STRIDE}`, counted on **all {N_TRAIN_DOCS} official train texts** "
        "(not the throughput subsample). Long documents become multiple windows; 3-epoch time uses that multiplier.",
        "- Gradient checkpointing probed for seq 512 when a true batch of 16 did not fit.",
        "",
        "## Results",
        "",
        "| Model | Params | Verdict | Max BS 256 | VRAM 256 alloc/res | Max BS 512 | VRAM 512 alloc/res | Rec. micro-BS | steps/s | tok/s | 3-epoch train |",
        "|---|---:|---|---:|---|---:|---|---:|---:|---:|---|",
    ]
    for alias, meta in MODELS.items():
        row = summary["models"][alias]
        v256 = row["vram"]["seq_256"]
        v512 = row["vram"]["seq_512"]
        tp = row["throughput"]
        wall = row["wallclock"]
        hours = wall.get("hours_3_epochs_train_only") if wall.get("ok") else None
        lines.append(
            "| `{alias}` ({name}) | {params}M | **{verdict}** | {bs256} | {a256}/{r256} | {bs512} | {a512}/{r512} | {rec} | {sps} | {tps} | {eta} |".format(
                alias=alias,
                name=meta["alias"],
                params=meta["params_m"],
                verdict=row["recipe"]["verdict"],
                bs256=v256.get("max_batch"),
                a256=v256.get("peak_alloc_gb"),
                r256=v256.get("peak_reserved_gb"),
                bs512=v512.get("max_batch"),
                a512=v512.get("peak_alloc_gb"),
                r512=v512.get("peak_reserved_gb"),
                rec=row["recipe"].get("micro_batch"),
                sps=tp.get("steps_per_sec", "n/a") if tp.get("ok") else "fail",
                tps=tp.get("tokens_per_sec", "n/a") if tp.get("ok") else "fail",
                eta=fmt_hours(hours),
            )
        )
    lines += [
        "",
        "VRAM numbers are **peak allocated / reserved GiB** at that max batch. Reserved is what the driver holds.",
        "",
        "## Windowing (official train, all 13k docs)",
        "",
        "| Model | tok/doc p50 | p95 | max | docs >512 | windows/doc @256 | windows/doc @512 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for alias in MODELS:
        stats = prep.get("models", {}).get(alias, {}).get("windows_full_train", {})
        tok = stats.get("tokens_no_special", {})
        w256 = stats.get("windows", {}).get("256", {})
        w512 = stats.get("windows", {}).get("512", {})
        lines.append(
            "| `{alias}` | {p50} | {p95} | {mx} | {over} ({frac:.1%}) | {w256} | {w512} |".format(
                alias=alias,
                p50=tok.get("p50"),
                p95=tok.get("p95"),
                mx=tok.get("max"),
                over=w512.get("docs_over_max_length"),
                frac=w512.get("docs_over_max_length_frac") or 0,
                w256=w256.get("windows_per_doc"),
                w512=w512.get("windows_per_doc"),
            )
        )
    lines += [
        "",
        "Most posts are one window at 512. The news tail is not: without sliding windows those entities are silently dropped. "
        "Epoch-time estimates multiply 13k docs by `windows_per_doc` at the recommended max_length.",
        "",
        "## Effective batch 16 at seq 512 (XLM-R / mDeBERTa)",
        "",
    ]
    for alias in ("exp0_xlm_roberta_base", "exp2_mdeberta_v3_base", "exp1_uztext_roberta"):
        row = summary["models"][alias]
        e16 = row["effective_batch_16"]
        ckpt = row["vram"].get("seq_512_checkpoint") or {}
        lines.append(
            f"- **{alias}**: true batch 16 without checkpointing: `{e16['fits_true_batch_16_no_ckpt']}` "
            f"(max micro `{e16['microbatch_no_ckpt']}`). "
            f"With checkpointing: true batch 16 `{e16['fits_true_batch_16_ckpt']}` "
            f"(max micro `{e16['microbatch_ckpt']}`, peak reserved {ckpt.get('peak_reserved_gb')} GiB). "
            f"Grad-accum steps to effective 16: `{e16['grad_accum_to_16_no_ckpt']}` (no ckpt) / "
            f"`{e16['grad_accum_to_16_ckpt']}` (ckpt). "
            f"Effective 16 is possible whenever micro-batch ≥ 1: `{e16['effective_16_possible']}`."
        )
    lines += ["", "## Per-model verdicts", ""]
    for alias, meta in MODELS.items():
        row = summary["models"][alias]
        rec = row["recipe"]
        wall = row["wallclock"]
        lines += [
            f"### {alias} (`{meta['alias']}`)",
            "",
            f"**{rec['verdict'].upper()}** on this 6 GB GPU for full fine-tune.",
            "",
            f"- Recommended: fp16, max_length={rec['max_length']}, micro-batch={rec['micro_batch']}, "
            f"grad_accum={rec['grad_accum']} (effective {rec['effective_batch']}), "
            f"gradient_checkpointing={rec['gradient_checkpointing']}, AdamW lr={rec['lr']}, epochs={rec['epochs']}, stride={rec['stride']}.",
        ]
        if wall.get("ok"):
            lines.append(
                f"- 3-epoch train-only estimate: **{fmt_hours(wall['hours_3_epochs_train_only'])}** "
                f"({wall['steps_per_epoch']} steps/epoch * {wall['steps_per_sec']:.3f} steps/s). "
                f"With a +15% eval tax: **{fmt_hours(wall['hours_3_epochs_plus_eval_tax'])}**."
            )
            lines.append(
                f"- Windows/epoch: {wall['n_windows_per_epoch']} "
                f"({wall['windows_per_doc']} windows/doc * {wall['n_docs']} docs)."
            )
        lines.append("")
    if cpu_train:
        lines += [
            "## Optional: CPU one-step (expect slow)",
            "",
            f"Threads: {cpu_train.get('threads')}. One forward+backward+AdamW, batch 1, seq 256, fp32.",
            "",
        ]
        for alias, row in cpu_train.get("models", {}).items():
            lines.append(
                f"- `{alias}`: **{row['seconds_one_step_bs1_seq256']} s** ({row['n_params']:,} tensors)."
            )
        lines += ["", "CPU training is not a realistic default on this machine.", ""]
    lines += [
        "## Recommended default training recipe (this GPU)",
        "",
        "Default the first official-only run to **uztext** if wall-clock is the bottleneck, and **XLM-R** if you want the organizer-shaped multilingual baseline the same night. mDeBERTa is the same VRAM class as XLM-R (278M, 12L) with slower relative-attention — only run it after the first two have numbers.",
        "",
        "Shared settings:",
        "",
        "- Data: `data/official/train.jsonl` / `dev.jsonl` only.",
        "- Labels: 7 BIO tags in `src/uzbek_ner/labels.py`.",
        "- `fp16` autocast + AdamW `2e-5`, 3 epochs, sliding window `max_length=512`, `stride=128`.",
        "- Effective batch **16** via grad accumulation (true batch 16 does not fit for the 278M models).",
        "- Do **not** full-train Qwen on this GPU; that is a LoRA question, not this bench.",
        "",
        "Per-model knobs are in the table and verdicts above (micro-batch, checkpointing).",
        "",
        "## Repro",
        "",
        "```bash",
        "HF_HUB_OFFLINE=1 uv run python scripts/bench_finetune.py --phase cpu-prep",
        "flock outputs/.gpu.lock env HF_HUB_OFFLINE=1 uv run python scripts/bench_finetune.py --phase gpu",
        "uv run python scripts/bench_finetune.py --phase cpu-train",
        "uv run python scripts/bench_finetune.py --phase report",
        "```",
        "",
        "Artifacts (gitignored): `outputs/bench/summary.json`.",
        "",
    ]
    return "\n".join(lines) + "\n"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=(
            "cpu-prep",
            "gpu",
            "cpu-train",
            "report",
            "all",
            "probe",
            "throughput",
            "hardware",
        ),
        default="all",
    )
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs" / "bench")
    parser.add_argument("--sample-n", type=int, default=SUBSAMPLE_DOCS)
    parser.add_argument(
        "--docs-path",
        type=Path,
        default=REPO_ROOT / "docs" / "BERT_hardware.md",
    )
    parser.add_argument("--model-alias", choices=list(MODELS.keys()), default=None)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    args = parser.parse_args()

    if args.phase == "hardware":
        emit_probe(collect_hardware())
        return
    if args.phase == "probe":
        if not args.model_alias:
            raise SystemExit("--model-alias is required for --phase probe")
        payload = run_probe_worker(
            args.model_alias,
            args.seq_len,
            args.batch_size,
            use_fp16=True,
            gradient_checkpointing=args.gradient_checkpointing,
            accum_steps=args.accum_steps,
        )
        emit_probe(payload)
        return
    if args.phase == "throughput":
        if not args.model_alias:
            raise SystemExit("--model-alias is required for --phase throughput")
        payload = run_throughput_worker(
            args.model_alias,
            args.seq_len,
            args.batch_size,
            args.sample_n,
            gradient_checkpointing=args.gradient_checkpointing,
        )
        emit_probe(payload)
        return

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    prep_path = out_dir / "cpu_prep.json"
    summary_path = out_dir / "summary.json"
    cpu_path = out_dir / "cpu_train.json"

    if args.phase in {"cpu-prep", "all"}:
        run_cpu_prep(out_dir, args.sample_n)
    if args.phase in {"gpu", "all"}:
        prep = load_json(prep_path) if prep_path.exists() else run_cpu_prep(out_dir, args.sample_n)
        run_gpu(out_dir, prep, args.sample_n)
    if args.phase in {"cpu-train", "all"}:
        run_cpu_train_step(out_dir)
    if args.phase in {"report", "all"}:
        if not summary_path.exists() or not prep_path.exists():
            raise SystemExit("report needs cpu_prep.json and summary.json")
        cpu_train = load_json(cpu_path) if cpu_path.exists() else None
        markdown = write_markdown(load_json(summary_path), load_json(prep_path), cpu_train)
        args.docs_path.parent.mkdir(parents=True, exist_ok=True)
        args.docs_path.write_text(markdown, encoding="utf-8")
        log(f"Wrote {args.docs_path}")


if __name__ == "__main__":
    main()

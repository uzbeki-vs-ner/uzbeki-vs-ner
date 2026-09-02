# BERT fine-tune on this laptop GPU

Numbers from `flock outputs/.gpu.lock uv run python scripts/bench_finetune.py --phase gpu`
(2026-09-02, RTX A1000 6 GB, torch 2.13+cu130, fp16 TokenClassification, 7 BIO tags).
Raw dump: `outputs/bench/summary.json` (gitignored). **Not a full training run.**

The bench agent died after the GPU phase; this note is assembled from that JSON plus
`outputs/bench/cpu_prep.json`. mDeBERTa was **not** proven impossible — the probe
crashed on a GradScaler bug (`Attempting to unscale FP16 gradients`).

## Hardware

| | |
|---|---|
| GPU | NVIDIA RTX A1000 Laptop, **5.67 GB** usable |
| CUDA | 13.0 (driver 580) |
| CPU | i7-13700H, 20 threads |
| RAM | 31 GB |

Display uses a few MiB. Treat **~5.5 GB** as the training budget.

## Verdict

| Backbone | Local alias | Params | Feasibility | Default recipe on this GPU |
|---|---|---:|---|---|
| **uztext RoBERTa 6L** | `exp1_uztext_roberta` | 83 M | **easy** | `max_length=512`, micro-batch **16–32**, fp16, no checkpointing |
| **XLM-RoBERTa-base** | `exp0_xlm_roberta_base` | 278 M | **tight** | `max_length=256`, micro-batch **8**; or `512` only with **bs=1** |
| **mDeBERTa-v3-base** | `exp2_mdeberta_v3_base` | 278 M | **unmeasured** (bench bug) | Expect XLM-R-like; retry with bf16 or correct AMP (weights fp32, autocast fp16) |

Do **not** trust the auto-written `recipe.verdict=easy` for XLM-R in `summary.json`:
it mixed seq-256 `max_batch=8` into a seq-512 recipe. Seq-512 XLM-R only survived **bs=1**.

## VRAM (AdamW + backward, fp16, subprocess-isolated probes)

Idle weights already eat a lot of the 6 GB:

| Model | Idle alloc | seq 256 max bs | peak @ that bs | seq 512 max bs | peak @ that bs |
|---|---:|---:|---:|---:|---:|
| uztext | 0.93 GB | **64** | 4.27 / 4.86 reserved | **32** (bs=48 OOM) | 4.27 / 4.86 |
| XLM-R | 3.10 GB | **8** (bs=16 OOM) | 4.19 / 4.87 | **1** (2+ OOM) | 4.16 / 4.41 |
| mDeBERTa | 1.58 GB | 16 OOM; 8→1 crashed scaler | — | 16–8 OOM; 4→1 scaler | — |

Gradient checkpointing on XLM-R seq 512: **all OOM** (16→1). Not a win on this card with the current probe (and it still used GradScaler).

Effective batch 16 at seq 512:

- uztext: **yes**, true batch 32 already ≥ 16.
- XLM-R: theoretically accum 16×bs1, but the accum probe **OOM’d** (fragmentation after the search). Prefer **seq 256 / bs 8** instead of heroic 512.
- mDeBERTa: not measured.

## Throughput (padded windows — pessimistic)

Only uztext completed a 40-step loop:

| | uztext |
|---|---|
| Setting | bs=8, seq≈510, fp16, pin_memory |
| steps/s | **4.36** |
| tokens/s (with padding) | **17.8k** |
| peak during loop | 1.88 alloc / 2.38 reserved GB |
| last loss (random head) | ~0.003 (meaningless) |

XLM-R 40-step loop at seq 512 / bs 1: **OOM** (same fragmentation). Single probe step at seq 256 / bs 8 was **0.67 s**.

## Windows on official train (13 000 docs, stride 128)

From `cpu_prep.json` (tokenizer.encode, no specials, then CLS/SEP budget):

| | uztext 512 | uztext 256 | XLM-R 512 | XLM-R 256 |
|---|---:|---:|---:|---:|
| docs overflowing | 3.7% | 9.8% | 6.2% | 14.6% |
| windows / epoch | 14 100 | 17 727 | 14 840 | 20 652 |
| windows / doc | 1.08 | 1.36 | 1.14 | 1.59 |

Throughput loops **pad to max_length**, so they overstate compute vs a length-grouped trainer.

## Wall-clock for 3 epochs (extrapolated)

**uztext, seq 512, bs 32** (VRAM-ok; speed scaled from 4.36 steps/s @ bs=8 — optimistic if bandwidth-bound):

- ~441 steps/epoch → **~0.5 min train / epoch** in the JSON (`15.7` steps/s scaled).
- Honest band: **~5–20 min for 3 epochs** including eval, logging, dataloader, sliding-window packing. This is the “iterate tonight” model.

**XLM-R, seq 256, bs 8** (the setting that actually fitted):

- ~2 582 steps/epoch × ~0.67 s/step (one-shot probe) → **~30 min/epoch train compute**.
- Band: **1.5–4 hours for 3 epochs** with eval. Tight but doable overnight / one afternoon.

**XLM-R seq 512 / bs 1**: ~14.8k steps/epoch; without a measured steps/s, assume several hours and a fragile VRAM margin. Use only if 256 truncates too many entities in error analysis.

**mDeBERTa:** no time estimate until AMP is fixed. Same 12×768 class as XLM-R → plan the XLM-R budget.

CPU 1-step timings were not recorded in `summary.json`.

## Recommended default (this machine)

1. **First training job:** uztext, `max_length=512`, `stride=128`, `batch_size=16` (leave headroom vs 32), `fp16`, AdamW `2e-5`, 3 epochs, Mix A (official train only).
2. **Quality competitor:** XLM-R, `max_length=256`, `batch_size=8`, same rest. Do not start at 512.
3. **mDeBERTa:** after trainer uses **bf16 or GradScaler-safe fp16** (model weights stay fp32). Then copy the XLM-R recipe.
4. **Soup:** only multiple seeds of the **same** backbone (uztext×N or XLM-R×N), not across these three.

## How to reproduce

```bash
flock outputs/.gpu.lock uv run python scripts/bench_finetune.py --phase cpu
flock outputs/.gpu.lock uv run python scripts/bench_finetune.py --phase gpu
```

`--phase probe` smokes a single (model, seq, batch). Keep `HF_HUB_OFFLINE=1`.

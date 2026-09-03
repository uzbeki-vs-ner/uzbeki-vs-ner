# Eval comparison dashboard

Visual layer on top of **MLflow**, not a replacement. MLflow still stores training runs, params, and artifacts. This stack compares **official exact-span** scores (ORG / NAME / GEO, same hash + exact `label/start/end`) across model runs.

The scorer is `uzbek_ner.metrics.exact_span.evaluate_prediction_files` — the same matching as the organizer `evaluate.py`. Do not reimplement it.

## Pieces

| Piece | Role |
|---|---|
| `outputs/eval/runs/{run_id}.json` | Registry (gitignored). Schema matches `scripts/train_ner.py`. |
| `scripts/register_eval_run.py` | Ingest metrics JSON, or score gold+predictions then register |
| `make eval-api` | CPU FastAPI on `0.0.0.0:8050` (HTML still at http://127.0.0.1:8050) |
| `make grafana-up` / `make eval-stack` | Prometheus + Grafana via `docker-compose.eval.yml` |

## Register a run

After a train/eval that wrote predictions (for example the uztext smoke):

```bash
make register-eval
```

That re-scores `outputs/official/uztext_smoke_dev_predictions.jsonl` against `data/official/dev.jsonl` and overwrites `outputs/eval/runs/uztext_smoke.json`.

From an existing metrics JSON (no re-score):

```bash
uv run python scripts/register_eval_run.py \
  --run-id uztext_smoke \
  --model rifkat/uztext-3Gb-BPE-Roberta \
  --checkpoint checkpoints/uztext_smoke \
  --split official_dev \
  --metrics outputs/official/uztext_smoke_dev_metrics.json \
  --gold data/official/dev.jsonl \
  --predictions outputs/official/uztext_smoke_dev_predictions.jsonl
```

`ner evaluate` / `dvc repro evaluate` also register a run when gold and predictions both exist (`configs/default.yaml` → `evaluate.*`). If files are missing, DVC still writes `metrics.json` with `status: waiting_for_predictions` and does **not** push a fake run to Grafana.

Smoke train already writes `outputs/eval/runs/uztext_smoke.json` itself.

Override the registry directory with `EVAL_RUNS_DIR`.

## Start the API

```bash
make eval-api
# GET http://127.0.0.1:8050/           HTML table (works without Grafana)
# GET http://127.0.0.1:8050/healthz
# GET http://127.0.0.1:8050/api/v1/runs
# GET http://127.0.0.1:8050/api/v1/runs/{run_id}
# GET http://127.0.0.1:8050/metrics    Prometheus text
```

Bind defaults to `0.0.0.0:8050` so Prometheus in Docker can scrape `host.docker.internal:8050`. Open the HTML table at [http://127.0.0.1:8050](http://127.0.0.1:8050). This process is CPU-only; do not take `outputs/.gpu.lock`.

## Start Grafana

Requires Docker. Prometheus and Grafana use **host networking** on this Linux box (they scrape `127.0.0.1:8050`; Docker-bridge → host timed out). Bind the eval API so Prometheus can reach it (`make eval-api` → `0.0.0.0:8050`).

```bash
make eval-api          # terminal 1
make eval-stack        # terminal 2 — Prometheus :9090, Grafana :3000
```

Open **http://127.0.0.1:3000** — login **admin / admin** (anonymous Viewer is also enabled). Dashboard: **NER → Uzbek NER — official exact-span comparison** (uid `uzbek-ner-eval`).

Stop: `make grafana-down`.

Compose must parse without a running stack: `docker compose -f docker-compose.eval.yml config`.

## Metrics in Prometheus

Gauges with labels `{run_id, model, split, label}`:

- `ner_eval_f1` / `ner_eval_precision` / `ner_eval_recall`
- `label` is `micro`, `macro`, `ORG`, `NAME`, or `GEO`

Plus `ner_eval_latest` (1 for newest `created_at`), `ner_eval_records`, `ner_eval_created_timestamp_seconds`.

Grafana OSS only — no extra plugins.

## How to read F1

This is **hackathon exact-span** F1, not CoNLL-English token F1. Same hash + exact `(label, start, end)`; a suffix off-by-one is a full miss. Organizers set **no F1 floor**.

| micro-F1 | Band | Meaning |
|---:|---|---|
| &lt; 0.20 | Broken | Random head / stub / frozen probe (~0.05–0.13) |
| 0.20–0.45 | Weak smoke | Training moves, not a system |
| 0.45–0.60 | Working baseline | **≈0.5 is not “as bad as zero”**, but not a product. uztext 2-epoch smoke ≈ **0.58** |
| 0.60–0.75 | Competitive | **≈0.7 is already good** on this metric |
| 0.75–0.85 | Strong hackathon | Ship-shaped; still watch ORG/NAME vs GEO and P vs R |
| ≥ 0.85 | Excellent / check the scorer | Rare here; 0.90 after two epochs is usually a bug |

Source of truth: `uzbek_ner.evaldash.bands` (`GET /api/v1/scale`). The HTML table and Grafana thresholds use the same cuts.

## Type vs boundary diagnostics

The organizer score stays **strict** (hash + label + start + end). A second, non-leaderboard breakdown lives on each run as `diagnostics`:

| Field | Meaning |
|---|---|
| `boundary_exact` | F1 on exact `(start, end)` **ignoring type** |
| `type_given_boundary.accuracy` | Of those aligned spans, fraction with the right ORG/NAME/GEO |
| `type_given_boundary.confusion` | 3×3 gold→pred counts on aligned spans |
| `buckets.exact_match` | Organizer TP |
| `buckets.type_mismatch` | Same offsets, wrong label |
| `buckets.partial_same_type` / `partial_diff_type` | Overlap but not exact bounds (suffix jitter) |
| `buckets.missed` / `spurious` | No overlap |

If boundary F1 ≫ official F1, the model finds mentions and **confuses types**. If both are low and `partial_same_type` is large, it is **agglutination / span edges**. Computed when gold+predictions are scored (`evaluate_and_register`); old run JSON without the field still loads.

Prometheus: `ner_eval_boundary_f1`, `ner_eval_type_accuracy`, `ner_eval_error_bucket{bucket=...}`.

## Tests

`tests/fixtures/eval/runs/` holds three fake runs with different micro-F1 so the API can rank models offline. Pytest does not start Docker or import torch for this stack.

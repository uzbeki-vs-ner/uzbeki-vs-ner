.PHONY: sync lint fmt typecheck test hooks mlflow-ui dvc-init dvc-repro dvc-exp pipeline \
	download-models download-external bench-gpu train-uztext-smoke train-uztext-mixb \
	train-uztext-mlp1 \
	check-api evaluate-official calibrate-threshold \
	evaluate-service-official service docker-build docker-run \
	eval-api grafana-up grafana-down eval-stack register-eval tune-decode-kfold

# Lint/test contract (CI calls `make lint` then `make test` — keep this order):
#   ruff check → ruff format --check → mypy → pytest
# Python 3.11, ruff paths src/tests/scripts, mypy via pyproject (`uv run mypy`).

OFFICIAL_DEV := $(CURDIR)/data/official/dev.jsonl
PREDICTIONS ?= $(CURDIR)/outputs/official/dev_predictions.jsonl
METRICS_OUT ?= $(CURDIR)/outputs/official/dev_metrics.json
SERVICE_URL ?= http://localhost:8000
SERVICE_HOST ?= 0.0.0.0
SERVICE_PORT ?= 8000
DOCKER_IMAGE ?= ner-uz-solution
# 0.0.0.0 so Prometheus in Docker can scrape host.docker.internal:8050.
# 127.0.0.1 is invisible from the compose network (host-gateway ≠ loopback).
EVAL_HOST ?= 0.0.0.0
EVAL_PORT ?= 8050
EVAL_COMPOSE := docker compose -f docker-compose.eval.yml

sync:
	uv sync --all-groups

lint:
	uv run ruff check src tests scripts
	uv run ruff format --check src tests scripts
	uv run mypy

fmt:
	uv run ruff format src tests scripts
	uv run ruff check --fix src tests scripts

typecheck:
	uv run mypy

test:
	uv run pytest

hooks:
	uv run pre-commit install

mlflow-init:
	uv run python scripts/register_mlflow_experiment.py

mlflow-ui:
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000

dvc-init:
	uv run dvc init -f 2>/dev/null || uv run dvc init
	uv run dvc config core.analytics false

dvc-repro:
	uv run dvc repro

dvc-exp:
	uv run dvc exp show

pipeline:
	uv run ner pipeline

download-models:
	uv run python scripts/download_models.py

download-external:
	uv run python scripts/download_external_datasets.py

# GPU lock: flock so other jobs (EMA, etc.) do not collide.
bench-gpu:
	flock outputs/.gpu.lock uv run python scripts/bench_finetune.py --phase gpu

# Smoke FT: uztext NER head, 2 epochs, official train/dev, exact-span score on dev.
train-uztext-smoke:
	flock outputs/.gpu.lock env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
		TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
		uv run python scripts/train_ner.py --epochs 2 --batch-size 16 --max-length 512 --stride 128

# Same recipe as smoke, but Linear(H,H)→GELU→Dropout→Linear(H,7) instead of Linear(H,7).
# Does not overwrite checkpoints/uztext_smoke.
train-uztext-mlp1:
	flock outputs/.gpu.lock env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
		TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
		uv run python scripts/train_ner.py \
			--head mlp \
			--output-dir checkpoints/uztext_mlp1 \
			--run-id uztext_mlp1 \
			--epochs 2 --batch-size 16 --max-length 512 --stride 128

# Mix B: continue from smoke, 1 epoch, official train + silver (capped to gold size).
# Score only official dev. Does not overwrite checkpoints/uztext_smoke.
train-uztext-mixb:
	flock outputs/.gpu.lock env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
		TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
		uv run python scripts/train_ner.py \
			--model checkpoints/uztext_smoke \
			--output-dir checkpoints/uztext_mixb_ep1 \
			--run-id uztext_mixb_ep1 \
			--extra-train data/processed/silver/all.jsonl \
			--extra-cap 13000 \
			--epochs 1 --batch-size 16 --max-length 512 --stride 128 \
			--learning-rate 1e-5

# Entity-confidence τ: GPU encode official dev if cache is cold, then CPU sweep.
# Separate from train. Does not change predict_records defaults.
calibrate-threshold:
	flock outputs/.gpu.lock env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
		TOKENIZERS_PARALLELISM=false \
		uv run ner calibrate

# Local uvicorn (CPU stub). One worker matches a future single-GPU model process.
service:
	uv run uvicorn uzbek_ner.service.app:app --host $(SERVICE_HOST) --port $(SERVICE_PORT) --workers 1

# Organizers: docker build -t ner-uz-solution . && docker run --rm -p 8000:8000 ner-uz-solution
# Stub image is CPU-only. A later GPU model will need: docker run --rm --gpus all -p 8000:8000 …
docker-build:
	docker build -t $(DOCKER_IMAGE) .

docker-run:
	docker run --rm -p 8000:8000 $(DOCKER_IMAGE)

check-api:
	uv run python scripts/run_official_tooling.py check-api --url $(SERVICE_URL)

evaluate-official:
	uv run python scripts/run_official_tooling.py evaluate \
	  --gold $(OFFICIAL_DEV) \
	  --predictions $(PREDICTIONS) \
	  --output $(METRICS_OUT)

evaluate-service-official:
	uv run python scripts/run_official_tooling.py evaluate-service \
	  --url $(SERVICE_URL) \
	  --predictions $(CURDIR)/outputs/official/service_dev_predictions.jsonl \
	  --output $(CURDIR)/outputs/official/service_dev_metrics.json

# Local eval comparison (CPU). Grafana scrapes this API; do not use the GPU lock.
eval-api:
	uv run uvicorn uzbek_ner.evaldash.app:app --host $(EVAL_HOST) --port $(EVAL_PORT)

grafana-up:
	$(EVAL_COMPOSE) up -d

grafana-down:
	$(EVAL_COMPOSE) down

eval-stack: grafana-up
	@echo "Grafana     http://127.0.0.1:3000  (admin/admin)"
	@echo "Prometheus  http://127.0.0.1:9090"
	@echo "Start the CPU metrics API in another terminal: make eval-api"
	@echo "Docs: docs/EVAL_DASHBOARD.md"

register-eval:
	uv run python scripts/register_eval_run.py \
	  --run-id uztext_smoke \
	  --model rifkat/uztext-3Gb-BPE-Roberta \
	  --checkpoint checkpoints/uztext_smoke \
	  --split official_dev \
	  --gold $(OFFICIAL_DEV) \
	  --predictions $(CURDIR)/outputs/official/uztext_smoke_dev_predictions.jsonl \
	  --metrics $(CURDIR)/outputs/official/uztext_smoke_dev_metrics.json

# CPU: K-fold decode tuning on official dev (analysis fold never scored).
tune-decode-kfold:
	uv run python scripts/tune_decode_kfold.py --k 5 --seed 42 \
	  --predictions $(CURDIR)/outputs/official/uztext_smoke_dev_predictions.jsonl \
	  --output $(CURDIR)/outputs/eval/decode_kfold_smoke_k5.json

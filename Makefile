.PHONY: sync lint fmt typecheck test hooks mlflow-ui dvc-init dvc-repro dvc-exp pipeline \
	download-models download-external check-api evaluate-official evaluate-service-official

# Lint/test contract (CI calls `make lint` then `make test` — keep this order):
#   ruff check → ruff format --check → mypy → pytest
# Python 3.11, ruff paths src/tests/scripts, mypy via pyproject (`uv run mypy`).

OFFICIAL_DEV := $(CURDIR)/data/official/dev.jsonl
PREDICTIONS ?= $(CURDIR)/outputs/official/dev_predictions.jsonl
METRICS_OUT ?= $(CURDIR)/outputs/official/dev_metrics.json
SERVICE_URL ?= http://localhost:8000

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

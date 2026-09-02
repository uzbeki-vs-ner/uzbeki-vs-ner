.PHONY: sync lint fmt test hooks mlflow-ui dvc-init dvc-repro dvc-exp pipeline

sync:
	uv sync --all-groups

lint:
	uv run ruff check src tests scripts
	uv run ruff format --check src tests scripts

fmt:
	uv run ruff format src tests scripts

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

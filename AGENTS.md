# Agent notes

## GitHub is a code platform

Do not commit datasets, zip archives, jsonl dumps, pretrained weights, checkpoints, MLflow stores, or virtualenvs. Track README / CATALOG / `.gitkeep` / DVC pipeline files (`dvc.yaml`, `dvc.lock`) only.

Official organizer data lives under `data/official/` (ignored). External sets under `data/external/` (ignored). Weights under `models/pretrained/` (ignored).

## Commit messages

Every commit needs a **subject**, a **blank line**, and a **body** with a few sentences or bullets explaining *why*. Conventional Commits is fine (`ci: …`, `chore: …`) if the body is still detailed. Do not commit `wip` / `fix` / `update`.

Local hook: `.githooks/commit-msg` (enable with `git config core.hooksPath .githooks`). Do not `--no-verify`.

## Lint / typecheck / test (single contract)

Python **3.11**. Ruff, MyPy, and pytest settings live in `pyproject.toml`.

**Command source of truth: `Makefile`.** Do not invent a second invocation.

- `make lint` — `uv run ruff check src tests scripts`, then `uv run ruff format --check src tests scripts`, then `uv run mypy` (mypy reads `[tool.mypy]`, currently `files = ["src"]`)
- `make test` — `uv run pytest`
- `make fmt` / `make typecheck` — local helpers only

CI (`.github/workflows/ci.yml`): setup-uv with Python 3.11, `uv sync --frozen --all-groups`, then **`make lint`** and **`make test`**. It must not download models/datasets or start MLflow.

Pre-commit uses the same ruff paths (`src|tests|scripts`) and `uv run mypy`. It does not run pytest.

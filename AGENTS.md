# Agent notes

## GitHub is a code platform

Do not commit datasets, zip archives, jsonl dumps, pretrained weights, checkpoints, MLflow stores, or virtualenvs. Track README / CATALOG / `.gitkeep` / DVC pipeline files (`dvc.yaml`, `dvc.lock`) only.

Official organizer data lives under `data/official/` (ignored). External sets under `data/external/` (ignored). Weights under `models/pretrained/` (ignored).

## Commit messages

Every commit needs a **subject**, a **blank line**, and a **body** with a few sentences or bullets explaining *why*. Conventional Commits is fine (`ci: …`, `chore: …`) if the body is still detailed. Do not commit `wip` / `fix` / `update`.

Local hook: `.githooks/commit-msg` (enable with `git config core.hooksPath .githooks`). Do not `--no-verify`.

## CI

`.github/workflows/ci.yml` runs `uv sync --frozen --all-groups`, then ruff, mypy, and pytest. It must not download models/datasets or start MLflow.

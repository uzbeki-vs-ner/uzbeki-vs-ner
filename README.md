# Uzbek NER — ITMO × Brand Analytics

Решение для выделения именованных сущностей (ORG, NAME, GEO) в узбекоязычных текстах.
Постановка задачи — в [CASE.md](CASE.md).

## Стек

- **Python 3.11+**, [uv](https://docs.astral.sh/uv/) — зависимости и lockfile
- **PyTorch + Transformers** — обучение и инференс
- **FastAPI** — HTTP-сервис (будет добавлен по мере реализации)
- **MLflow** — эксперименты, метрики, артефакты (`sqlite:///mlflow.db`)
- **DVC** — версионирование данных и воспроизводимый пайплайн (`dvc.yaml`)
- **Hydra + Typer** — конфиги и CLI
- **Ruff + MyPy + pre-commit** — форматирование, линт и проверка типов

## Быстрый старт

```bash
uv sync --all-groups
uv run pre-commit install          # опционально
make mlflow-init                   # зарегистрировать эксперименты
make dvc-init                      # инициализировать DVC (если ещё не)
uv run pytest
```

Переменные окружения — [.env.example](.env.example) → скопируй в `.env`.

## Данные (DVC)

1. Положи `train.json` и `val.json` в `data/raw/` (см. [data/raw/README.md](data/raw/README.md)).
2. Зафиксируй:

```bash
uv run dvc add data/raw/train.json data/raw/val.json
git add data/raw/*.dvc .gitignore
```

3. При необходимости настрой remote: `uv run dvc remote add -d storage /path/to/storage`.

## Пайплайн

```bash
make pipeline          # prepare → train → evaluate (CLI)
make dvc-repro         # то же через DVC (рекомендуется для сдачи)
make dvc-exp           # таблица экспериментов DVC
```

Гиперпараметры — `params.yaml` (DVC) и `configs/default.yaml` (Hydra).

## MLflow

```bash
make mlflow-ui         # http://127.0.0.1:5000
```

Эксперименты: `uzbek_ner` (основной), `uzbek_ner_smoke` (отладочные прогоны).

## Git

Репозиторий: `git@github.com:uzbeki-vs-ner/uzbeki-vs-ner.git`.

GitHub — **код**, не дамп: не коммить `data/official` / `data/external` датасеты, zip, jsonl, `models/pretrained`, чекпоинты, `mlflow.db` / `mlruns`. README, CATALOG, `.gitkeep`, `dvc.yaml` / `dvc.lock` — можно.

```bash
./scripts/git_init_github.sh   # user.email, origin, core.hooksPath=.githooks
git config core.hooksPath .githooks
chmod +x .githooks/commit-msg .githooks/pre-push
```

Хуки:

- **commit-msg** — тема + пустая строка + тело (почему). Однострочники вроде `wip` / `fix` отклоняются. Подробнее: [`.githooks/README.md`](.githooks/README.md).
- **pre-push** — блок корпоративных remote.

CI: [`.github/workflows/ci.yml`](.github/workflows/ci.yml) — Python 3.11, `uv sync --frozen --all-groups`, затем **`make lint`** и **`make test`**. Без скачивания моделей/датасетов и без MLflow UI. Команды линта/тестов не дублировать: источник истины — `Makefile` (см. ниже).

## Структура

```
├── CASE.md
├── AGENTS.md             # commit/CI/data rules for agents
├── configs/              # Hydra
├── params.yaml           # DVC params
├── dvc.yaml              # DVC pipeline (committed; cache is not)
├── src/uzbek_ner/        # код
├── tests/
├── scripts/
├── data/official/        # организаторы (jsonl/zip/bundle gitignored)
├── data/external/        # публичные датасеты (blobs gitignored; CATALOG.md да)
├── data/processed/       # артефакты prepare
├── checkpoints/          # веса (gitignored)
└── models/pretrained/    # HF dumps (gitignored)
```

## Линт и типы

Один контракт для локальной разработки, pre-commit и GitHub Actions. Конфиг инструментов — `pyproject.toml`; **команды — `Makefile`** (`make lint`, `make test`). Порядок: ruff check → ruff format --check → mypy → pytest.

Ruff смотрит `src tests scripts`. MyPy — `uv run mypy` (в pyproject сейчас `files = ["src"]`); сторонние ML-библиотеки (torch, transformers, datasets, mlflow, hydra, omegaconf, gliner, seqeval и т.п.) не требуют стабов — для них включён `ignore_missing_imports`.

```bash
make fmt          # ruff format + ruff check --fix
make lint         # ruff check + ruff format --check + mypy
make typecheck    # только mypy
make test         # pytest
```

Pre-commit гоняет `ruff-format` и `ruff` по `src|tests|scripts` и `uv run mypy` по `src/`. Pytest — только `make test` / CI.

## Команды

```bash
make sync fmt lint test
uv run ner prepare
uv run ner train
uv run ner evaluate
```

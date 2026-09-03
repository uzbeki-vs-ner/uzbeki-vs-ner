# Uzbek NER — ITMO × Brand Analytics

Решение для выделения именованных сущностей (ORG, NAME, GEO) в узбекоязычных текстах.
Постановка задачи — в [CASE.md](CASE.md).

## Стек

- **Python 3.11+**, [uv](https://docs.astral.sh/uv/) — зависимости и lockfile
- **PyTorch + Transformers** — обучение и инференс
- **FastAPI** — HTTP-сервис инференса (`GET /healthz`, `POST /api/v1/predict`) и CPU eval-API (`127.0.0.1:8050`)
- **MLflow** — эксперименты, метрики, артефакты (`sqlite:///mlflow.db`)
- **Grafana + Prometheus** — сравнение моделей по official exact-span (поверх MLflow, не вместо)
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
make calibrate-threshold   # отдельный этап: кэш logits на official dev + сетка τ
make dvc-exp           # таблица экспериментов DVC
```

Гиперпараметры — `params.yaml` (DVC) и `configs/default.yaml` (Hydra).

Обучение и порог уверенности — **разные этапы**. `prepare → train → evaluate` учит модель и считает exact-span по готовому JSONL. Поиск τ (`ner calibrate` / `make calibrate-threshold`) прогоняет чекпоинт по official dev, кэширует mean-logits и подбирает порог на analysis-fold. Это быстрее эпохи, но всё равно отдельная работа: без кэша нужен GPU (`flock outputs/.gpu.lock`), с кэшем — только CPU. Выбранный τ пишется в `outputs/eval/threshold.json` и `threshold_metrics.json`; в дефолтный decode он сам не включается.

## MLflow

```bash
make mlflow-ui         # http://127.0.0.1:5000
```

Эксперименты: `uzbek_ner` (основной), `uzbek_ner_smoke` (отладочные прогоны).

## Сравнение моделей (Grafana)

MLflow остаётся логом экспериментов. Для визуального сравнения official exact-span (ORG / NAME / GEO) — локальный реестр `outputs/eval/runs/*.json` + Grafana. Подробности: [docs/EVAL_DASHBOARD.md](docs/EVAL_DASHBOARD.md).

```bash
make register-eval     # gold + predictions → outputs/eval/runs/{run_id}.json
make eval-api          # http://127.0.0.1:8050  (HTML-таблица и /metrics)
make eval-stack        # Prometheus :9090, Grafana :3000 (admin/admin)
```

Дашборд: **NER → Uzbek NER — official exact-span comparison**. Scorer тот же, что у организаторов (`uzbek_ner.metrics.exact_span`).

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
├── Dockerfile            # stub inference image (no torch)
├── docker-compose.eval.yml  # Grafana + Prometheus (scrape host:8050)
├── eval/                 # provisioning Grafana / Prometheus
├── src/uzbek_ner/        # код (включая service/ и evaldash/)
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

## HTTP-сервис (контракт организаторов)

Слушает `0.0.0.0:8000`. Оценённый API — **один JSON-объект после всего батча** (не SSE/WebSocket). Пока бэкенд — детерминированный gazetteer-заглушка: без весов HuggingFace и без GPU. Настоящая модель позже подменит `StubNerBackend`.

Локально:

```bash
make service                 # uvicorn, 1 worker
make check-api               # official scripts/check_service.py → http://localhost:8000
```

Docker (без монтирований, без обязательных env, без интернета в runtime):

```bash
docker build -t ner-uz-solution .
docker run --rm -p 8000:8000 ner-uz-solution
# или: make docker-build && make docker-run
```

Образ stub — **CPU-only**. Когда появится GPU-модель, запуск будет:

```bash
docker run --rm --gpus all -p 8000:8000 ner-uz-solution
```

## Команды

```bash
make sync fmt lint test
uv run ner prepare
uv run ner train
uv run ner evaluate
uv run ner calibrate     # порог уверенности; не входит в `ner pipeline`
make calibrate-threshold # то же, под GPU lock
make service
make eval-api eval-stack register-eval
```

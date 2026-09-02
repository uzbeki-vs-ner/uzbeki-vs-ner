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
- **Ruff + pre-commit** — форматирование и линт

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

## Git (личный GitHub: bugkira)

Глобальный git на машине — `dsereda@ptsecurity.com`. В репозитории локально: `marvolo04@mail.ru` + SSH `git@github.com:bugkira/...`.

```bash
./scripts/git_init_github.sh
ssh -T git@github.com   # Hi bugkira!
```

Перед первым push создай пустой репозиторий `bugkira/ITMO_hack` на GitHub.

## Структура

```
├── CASE.md
├── configs/             # Hydra
├── params.yaml          # DVC params
├── dvc.yaml             # DVC pipeline
├── src/uzbek_ner/       # код
├── tests/
├── scripts/
├── data/raw/            # сырые данные (DVC)
├── data/processed/      # артефакты prepare
├── checkpoints/         # веса
└── models/              # экспорт для инференса
```

## Команды

```bash
make sync fmt lint test
uv run ner prepare
uv run ner train
uv run ner evaluate
```

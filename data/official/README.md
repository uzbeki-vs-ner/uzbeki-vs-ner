# Official — данные организаторов хакатона

**Приоритет #1.** Все эксперименты и метрики для сдачи считаются на этих файлах.

## Файлы

| Путь | Описание |
|---|---|
| `train.jsonl` | 13 000 текстов, ORG / NAME / GEO, exact-span offsets |
| `dev.jsonl` | 1 500 текстов (валидация) |
| `dataset_manifest.json` | схема, SHA-256, статистика по сплитам |

Формат записи:

```json
{"hash":"…","text":"…","entities":[{"label":"ORG","start":0,"end":8}]}
```

## Baseline и скрипты оргов

```
bundle/ner_uz_hackathon_participant/
├── baseline/          train.py, predict.py
├── scripts/         evaluate.py, check_service.py, evaluate_service.py
├── API.md
├── LABELING_GUIDE.md
└── README.md
```

Запуск baseline (из корня bundle):

```bash
cd data/official/bundle/ner_uz_hackathon_participant
python -m baseline.train --train data/train.jsonl --dev data/dev.jsonl --output-dir ../../../checkpoints/organizer_baseline
python scripts/evaluate.py --gold data/dev.jsonl --pred predictions.jsonl
```

## Official tooling из корня репозитория

Орговские скрипты (`evaluate.py`, `check_service.py`, `evaluate_service.py`) — **источник истины** для метрик и API.

```bash
# exact-span метрики по JSONL предсказаний
make evaluate-official PREDICTIONS=outputs/official/dev_predictions.jsonl

# проверка HTTP-контракта (сервис должен быть запущен)
make check-api SERVICE_URL=http://localhost:8000

# прогон dev через HTTP + метрики
make evaluate-service-official SERVICE_URL=http://localhost:8000
```

Наш порт span-логики: `src/uzbek_ner/spans.py` (`align_labels`, `decode_bio_tokens`).  
Наш scorer (совпадает с орговским): `src/uzbek_ner/metrics/exact_span.py`.

## Provenance

Источник: `archive/ner_uz_hackathon_participant-20260902T111458Z-1-001.zip` (Google Drive организаторов, 2026-09-02).

`train.jsonl`, `dev.jsonl`, `dataset_manifest.json` — симлинки на `bundle/.../data/`.

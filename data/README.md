# Данные проекта

## Приоритет источников

```
1. data/official/     ← ОРГАНИЗАТОРЫ (главный источник, метрики соревнования)
2. data/external/     ← дополнительные публичные датасеты (только augmentation)
3. data/processed/    ← артефакты пайплайна
```

**Правило:** обучение и отчётные метрики — только на `official/dev.jsonl`. External не смешивать с official val/dev.

## Official (организаторы)

| Файл | Записей | Назначение |
|---|---:|---|
| `official/train.jsonl` | 13 000 | обучение |
| `official/dev.jsonl` | 1 500 | валидация (в CASE.md — val) |
| `official/dataset_manifest.json` | — | схема, SHA-256, статистика |

Полный комплект (baseline, scripts, API): `official/bundle/ner_uz_hackathon_participant/`  
Архив: `official/archive/ner_uz_hackathon_participant-20260902T111458Z-1-001.zip`

## External (дополнение)

См. [`external/CATALOG.md`](external/CATALOG.md) и `external/manifest.json`.

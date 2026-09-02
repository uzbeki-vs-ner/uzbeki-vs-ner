# Внешние датасеты (Uzbek NER)

Скачивание:

```bash
uv run python scripts/download_external_datasets.py
uv run python scripts/download_external_datasets.py --only uzner_100k uznlp_uzbek_ner_gold
```

Манифест после загрузки: `data/external/manifest.json`

## Приоритет для хакатона (ORG / NAME / GEO)

| Key | Источник | ~Объём | Лейблы | Маппинг → hackathon |
|---|---|---:|---|---|
| **uzner_100k** | [Zenodo 18903080](https://doi.org/10.5281/zenodo.18903080) | 114k sent | 18 типов, BIOES | PER→NAME, ORG→ORG, LOC/GPE→GEO |
| **risqaliyevds_uzbek_ner** | [HF risqaliyevds/uzbek_ner](https://huggingface.co/datasets/risqaliyevds/uzbek_ner) | 19.6k docs | mention lists | PERSON→NAME, ORG→ORG, LOC/GPE→GEO |
| **ner_prepared_uzbek** | [HF ShakhzoDavronov/ner-prepared-uzbek](https://huggingface.co/datasets/ShakhzoDavronov/ner-prepared-uzbek) | 19.6k sent | token BIO | то же (готовый token-classification) |
| **uznlp_uzbek_ner_gold** | [HF uznlp-uz/uzbek_NER](https://huggingface.co/datasets/uznlp-uz/uzbek_NER) | 4.2k sent | BIO TSV | PER→NAME, ORG→ORG, LOC→GEO |
| **wikiann_uz** | [HF unimelb-nlp/wikiann](https://huggingface.co/datasets/unimelb-nlp/wikiann) | 3×1k sent | PER/ORG/LOC | weak supervision, мало данных |
| **uzlegalner_v3** | [Zenodo 18816402](https://doi.org/10.5281/zenodo.18816402) | legal | PER/ORG/LOC + legal | domain augment для ORG/NAME/GEO |
| **uzbek_legal_ner_full** | [Zenodo 19682709](https://doi.org/10.5281/zenodo.19682709) | legal | 12 типов | core gold + synthetic augment |

## Вспомогательные (не целевые ORG/NAME/GEO)

| Key | Зачем |
|---|---|
| **rubai_ner_150k_personal** | Synthetic PII, усиление NAME + смешанный uz/ru |
| **uz_medner** | Medical NER — другой домен, только если нужен transfer |

## Не включено

| Источник | Почему |
|---|---|
| `uznlp-uz/uz_edbench` | Medical triage classification, не token-NER |
| Common Voice / OSCAR | Нет NER-разметки |

## Official vs external

| | `data/official/` | `data/external/` |
|---|---|---|
| Приоритет | **#1 — организаторы** | #2 — augmentation |
| Метрики сдачи | ✅ train + dev | ❌ не для отчёта |
| Файлы | `train.jsonl`, `dev.jsonl` | HF, Zenodo (см. выше) |

## Структура на диске

```
data/external/
├── CATALOG.md
├── manifest.json
├── hf/
│   ├── uznlp_uzbek_ner_gold/
│   ├── risqaliyevds_uzbek_ner/
│   └── ...
└── zenodo/
    ├── uzner_100k/extracted/
    └── ...
```

## Важно

1. **Не мешать** external data с организаторским val при отчёте метрик соревнования.
2. Перед обучением — **конвертер** в единый JSON-формат (`text_id`, `text`, `entities[{start,end,type}]`).
3. Для `risqaliyevds/uzbek_ner` нужен aligner mention→char spans (строковый поиск + валидация).

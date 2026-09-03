# Текущий подход и замеры

Дата среза: **2026-09-03**. Ветка кода: `main` (этот документ живёт на `notes/current-approach`).
Метрика соревнования: exact-span micro-F1 по `(hash, label, start, end)`, **без частичных баллов**. Суффиксы падежа в gold входят в спан.

Читать это **до** смены бэкбона, головы, Mix-рецепта или дефолтного decode.

---

## Для агента: не делать без явного запроса

- **Не менять бэкбон.** Продуктовый выбор — `rifkat/uztext-3Gb-BPE-Roberta` (локально `models/pretrained/exp1_uztext_roberta`). XLM-R / mDeBERTa не подставлять «на всякий случай».
- **Не замораживать веса энкодера** на обучении. «Замороженный бэкбон» = *выбор модели зафиксирован*, а не `requires_grad=False`. Линейный зонд на замороженном энкодере давал exact F1 0.05–0.13.
- **Не включать τ в `predict_records` по умолчанию.** Порог измерен, в дефолтный decode не входит.
- **Не воскрешать** `FAILED/add_word/` (дописка соседнего слова). Held-out ΔF1 +0.003 поверх порога, без порога ~0.
- **Не склеивать** соседние same-type спаны (ловушка: 284 помогают / 383 ломают уже точные).
- **Не повторять** Mix B 1:1, MLP-голову «чтобы стало лучше», `min_p` / geom-mean / Platt как апгрейд лидерборда — всё уже измерено, см. ниже.
- Не коммитить `data/`, `checkpoints/`, `outputs/`, веса, `mlruns/`. Кэши логитов: `outputs/cache/*_official_dev/`.

GPU на этой машине делить через `flock outputs/.gpu.lock`.

---

## Рабочий рецепт (то, что считаем текущим)

| Слой | Что сейчас |
|---|---|
| Энкодер | uztext RoBERTa 6L, hidden 768, ~83 M. Дообучается целиком. |
| Голова | `RobertaForTokenClassification`: Dropout + **`Linear(768, 7)`**. Без CRF, без hidden MLP. |
| Обучение | official `train.jsonl` (13 k), 2 эпохи, lr 2e-5, bs 16, max_length 510 (RoBERTa clamp от 512), stride 128, seed 42. Чекпоинт: `checkpoints/uztext_smoke`. |
| Инференс окон | mean logits по перекрывающимся окнам, затем softmax. T не двигает argmax. |
| Decode (дефолт) | argmax BIO → `snap_entities` (края слова + суффикс внутри спана + NMS longer-wins). Gold не используется. |
| Порог уверенности | **не в дефолте.** Скор спана = mean p предсказанного класса по токенам BIO **до** snap. Лучший измеренный τ=0.7. |
| Калибровка температуры | T\*=1. Температура не нужна. |

Команды:

```bash
make train-uztext-smoke      # не затирать без нужды
make calibrate-threshold     # GPU-кэш, если холодный; τ в файл, не в decode
```

---

## Главные цифры (official-dev, 1500 доков, 7698 gold)

Сплит порога всегда: seed 42, 5-fold, fold 0 = analysis (300), rest = held-out (1200). Порог с analysis, цифра — held-out, если не сказано иное.

### Линейный smoke (`uztext_smoke`)

| Decode | Где | P | R | F1 |
|---|---|---:|---:|---:|
| raw BIO, без snap | полный dev | 0.519 | 0.652 | **0.578** |
| + word snap (дефолт) | полный dev | 0.619 | 0.729 | **0.669** |
| snap, без порога | held-out | 0.618 | 0.728 | **0.668** |
| snap + τ=0.7 по `mean_p` | held-out | 0.767 | 0.648 | **0.702** |
| snap + τ=0.67 (пик analysis) | held-out | 0.750 | 0.664 | **0.704** |

Snap: exact 5022→5611, partial same-type 1485→867, spurious 2733→2139, missed ~770. Reading `boundary_jitter` → `healthy`. Пропуски snap почти не лечит.

По классам после snap (полный dev): ORG ~0.610, NAME ~0.618, GEO ~0.780.

### Mix B (continue smoke, 1 эпоха, official+13k silver 1:1)

Чекпоинт `checkpoints/uztext_mixb_ep1`. Snap F1 **0.666** на полном dev — чуть хуже smoke. Рецепт **снят**. Порог τ=0.7 на held-out: **0.668 → 0.702** (тот же плюс, что у линейного). T\*=1, token ECE 0.011.

### MLP-голова (2 эпохи, тот же train, не бэкбон)

`Linear(H,H)→GELU→Dropout→Linear(H,7)`, чекпоинт `checkpoints/uztext_mlp1`.

| Decode | Linear smoke | MLP |
|---|---:|---:|
| raw | 0.578 | 0.581 |
| snap | **0.669** | 0.667 |
| snap+τ=0.7 held-out | 0.702 | 0.701 |

MLP **не лучше** линейной. Дальше смотреть линейный smoke.

---

## Уверенность и калибровка

Скор спана: mean predicted-class token p по токенам BIO-спана до snap.
Метка для binary-метрик: snapped `(label,start,end)` ∈ gold.

**Не путать две F1.** Binary F1 среди уже предсказанных спанов (~0.81 при τ=0.7) — не организаторская. В неё не входят gold, которых модель не предложила. Лидерборд — ~0.70 с порогом.

### Токенный ECE врёт из-за `O`

170 128 merged content-токенов. Gold `O` = **85.3%** (не 98%). На `O`: acc 0.980, mean max-p 0.971, ECE 0.009. 141 529 токенов в бине [0.93, 1.0] с gap 0.003 → all-token ECE **0.008**.

| Срез | n | ECE | Brier | NLL | acc | mean p | gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| Token · все | 170 128 | 0.008 | 0.085 | 0.169 | 0.943 | 0.951 | +0.008 |
| Token · gold O | 145 037 | 0.009 | 0.031 | 0.062 | 0.980 | 0.971 | −0.009 |
| Token · gold ≠ O | 25 091 | **0.107** | 0.400 | 0.791 | 0.726 | 0.833 | **+0.107** |
| **Спан · exact vs conf** | **9 672** | **0.200** | **0.211** | **0.610** | **0.596** | **0.796** | **+0.200** |

Модель думает, что спан точен на 80%, exact rate 60%. Классика NER: лишнее/недостающее слово на краю → 0 по метрике, скор остаётся высоким.

### Бины спанов (width 0.1, smoke, 9672 pred)

| bin | n | TP | FP | Precision |
|---|---:|---:|---:|---:|
| [0.0, 0.1) | 0 | 0 | 0 | — |
| [0.1, 0.2) | 1 | 0 | 1 | 0.00 |
| [0.2, 0.3) | 17 | 5 | 12 | 0.29 |
| [0.3, 0.4) | 210 | 33 | 177 | 0.16 |
| [0.4, 0.5) | 623 | 120 | 503 | 0.19 |
| [0.5, 0.6) | 1 102 | 254 | 848 | 0.23 |
| [0.6, 0.7) | 1 122 | 354 | 768 | 0.32 |
| [0.7, 0.8) | 1 112 | 492 | 620 | 0.44 |
| [0.8, 0.9) | 1 301 | 756 | 545 | 0.58 |
| [0.9, 1.0] | 4 184 | 3 749 | 435 | **0.90** |

Ниже 0.7 precision ≤ 0.32. Softmax max-p не бывает < 1/7 ≈ 0.14; спанов с conf < 0.10 нет. τ=0.3 ничего не режет.

ROC по `mean_p` vs exact: AUC **0.848**, AP 0.901. Token argmax-correct AUC 0.939; детект `1−p(O)` AUC 0.983.

Примеры FP/TP/misses: `outputs/error_analysis.txt` (gitignored). Пулы: 1600 FP с conf≥0.7; 608 exact в [0.5, 0.7) которых τ=0.7 убьёт; 714 gold без пересечения с pred.

---

## Что пробовали как фильтр и не взлетело

Порог и LR только с analysis, F1 — held-out organizer. Decode: фильтр → snap+NMS на документ.

| Decode | τ с analysis | Held-out F1 |
|---|---:|---:|
| snap, без порога | — | 0.668 |
| mean_p, τ=0.7 фикс. | 0.70 | **0.702** |
| mean_p, сетка 0.01 | 0.67 | 0.704 |
| geom_mean_p | 0.66 | 0.703 |
| min_p | 0.50 | 0.678 |
| min_p при τ=0.7 | 0.70 | 0.675 |
| LR `[mean_p, min_p, n_tok]` | 0.29 | 0.697 |

`min_p` хуже mean: слабый крайний токен бывает и у exact после snap.
LR **калибрует** (held-out span ECE 0.201 → 0.068, gap +0.200 → ~0), но **не поднимает** лидерборд: честная вероятность не чинит границы.

---

## Убитые decode-трюки

| Идея | Результат | Где лежит |
|---|---|---|
| Word snap | +0.09 F1, дефолт | `uzbek_ner.decode.snap` |
| Температура softmax | T\*=1, argmax плоский | `scripts/calibrate_temperature.py` |
| Mix B 1:1 silver | snap F1 0.666 < smoke | не рецепт |
| MLP 1 hidden | ≈ linear | `modeling/heads.py`, чекпоинт не дефолт |
| Дописка 1 слова | +0.003 / ~0 | `FAILED/add_word/` |
| Склейка соседей same-type | 284 help / 383 destroy | не трогать |
| min_p / geom / Platt как фильтр | ≤ mean τ=0.7 | этот документ |

---

## Куда бить дальше (ещё не измерено как рецепт)

Починить **границы**, а не порог. Пока частичный FP остаётся FP, модель будет «переуверенной» и F1 упрётся в ~0.70.

Идеи (не внедрены, не измерены end-to-end):

1. Отрезать служебные хвосты у GEO/ORG: `shahriga`, `shahri`, `viloyatida`, `markazi`, `apparati`, …
2. NAME: если спан = `o‘g‘li` / `qizi`, расширить влево на 2–3 capitalized слова.
3. Label smoothing α=0.1 — только если будет новое обучение, не вместо границ.

Ожидание «0.70 → 0.74+» зависит от того, сколько high-conf FP — это обрезанные ФИО и длинные ORG (см. группу 1 в `outputs/error_analysis.txt`: `Ortiqxoʻjayev` ⊂ `Jahongir Ortiqxoʻjayev`, `Mámleketlik` ⊂ длинного министерства).

---

## Где лежат артефакты (не в git)

| Что | Путь |
|---|---|
| Линейный чекпоинт | `checkpoints/uztext_smoke/` |
| MLP | `checkpoints/uztext_mlp1/` |
| Mix B | `checkpoints/uztext_mixb_ep1/` |
| Кэш logits smoke | `outputs/cache/uztext_smoke_official_dev/` |
| Кэш MLP / Mix B | `outputs/cache/uztext_mlp1_official_dev/`, `mixb_official_dev/` |
| Калибровка JSON | `outputs/eval/uztext_smoke_calibration.json` |
| ROC JSON | `outputs/eval/uztext_smoke_roc.json` |
| Примеры ошибок | `outputs/error_analysis.txt` |
| Гистограмма TP/FP | `confidence_distribution.png` (корень репо, не коммитить) |

Протокол кэша: `uzbek_ner.modeling.eval_cache.fill_cache` под `flock outputs/.gpu.lock`.
Протокол τ: `uzbek_ner.pipeline.calibrate` / `uzbek_ner.decode.threshold`.

Вот готовый, исчерпывающий Markdown-файл. Сохраните его как `UZBEK_NER_HACKATHON_GUIDE.md` и скормите своему Cursor Agent в качестве контекста проекта.

***

```markdown
# 🏆 Uzbek NER Engineering & Hackathon Master Guide

Руководство и архитектурный план для автономного агента разработки (Cursor Agent). Документ содержит правила задачи, разбор грабель, код для точной валидации Exact-Span F1, шаблоны обучения моделей-кандидатов, сервис инференса и Dockerfile.

---

## 📌 1. Спецификация задачи и ограничения

* **Язык:** Узбекский (латиница, кириллица, смешанный русско-узбекский текст, новостной и разговорный стили).
* **Классы сущностей (3 класса):**
  * `ORG` — Организации, бренды, госорганы, банки, платформы (*Spendrups*, *Vazirlar Mahkamasi*, *Kun.uz*, *Markaziy banki*).
  * `NAME` — Имена, фамилии, персоны, однозначные псевдонимы (*Shavkat Mirziyoyev*, *Islom Karimov*).
  * `GEO` — Страны, регионы, города, районы, именованные локации (*Toshkent*, *O'zbekiston*, *Farg'ona viloyati*).
* **Датасет:** 
  * `train.json` — 13 000 размеченных текстов.
  * `val.json` — 1 500 размеченных текстов.
  * `test` — закрытая выборка.
* **Целевая метрика:** **Exact-Span Micro-F1** (совпадение `text_id`, `start_char`, `end_char`, `label`). Частичные совпадения границ или неверный класс дают 0.
* **Требования к сдаче:** Воспроизводимый пайплайн, Docker-контейнер с FastAPI инференс-сервисом, таблица экспериментов, анализ ошибок, презентация.

---

## ⚠️ 2. Критические технические ловушки (Прочитать перед кодингом!)

1. **Сдвиг символьных координат (Offset Shift Bug):**
   * Если нормализовать текст (заменять апострофы `o'`, `o‘`, `oʻ` или удалять спецсимволы) прямо в строке, **символьные индексы `(start, end)` сместятся**, что приведет к `0.0 F1` на тесте.
   * *Правило:* Токенизировать исходную строку как есть с `return_offsets_mapping=True`. Либо вести строгую карту соответствия `char_map[new_idx] = orig_idx`.
2. **Проклятие мультиязычности (Curse of Multilinguality / Token Fertility):**
   * Мультиязычные модели (`XLM-R`, `mDeBERTa`) тратят на тюркские языки мало токенов в словаре — одно узбекское слово дробится на 4–7 subwords.
   * *Правило:* Обязательно сравнить мультиязычные модели с монолингвальными (`rifkat/uztext-3Gb-BPE-Roberta`) и микро-LLM с богатым словарем (`Qwen2.5-0.5B`).
3. **Длинные статьи СМИ (Truncation):**
   * BERT обрезает тексты длиннее 512 токенов. Сущности в конце длинных статей будут потеряны.
   * *Правило:* Использовать скользящее окно со страйдом (`stride=128`, `return_overflowing_tokens=True`).
4. **Доминирование класса `O` (Class Imbalance):**
   * Более 85–90% токенов — класс `O`. 
   * *Правило:* Использовать `BIOES` вместо `BIO`, применять Class Weights или Focal Loss при необходимости.

---

## 🛠 3. Архитектурный скоуп для экспериментов (Бенчмарк)

Агент должен реализовать единый пайплайн для сравнения следующих 4 подходов:

| ID | Архитектура / Модель | HuggingFace Checkpoint | Почему стоит тестировать |
|---|---|---|---|
| **EXP-1** | Uzbek Monolingual RoBERTa | `rifkat/uztext-3Gb-BPE-Roberta` | Оптимальный BPE для узбекского языка, низкая фрагментация слов. |
| **EXP-2** | Disentangled Multilingual | `microsoft/mdeberta-v3-base` | SOTA энкодер для смешанных (узб+рус) контекстов. |
| **EXP-3** | Decoder-as-Encoder (SLM) | `Qwen/Qwen2.5-0.5B` (TokenClassifier) | Претрейн на 18T токенов, словарь 151k, знание современных сущностей. |
| **EXP-4** | Direct Span Prediction | `urchade/gliner_multi-v2.1` | Прямое предсказание спанов без проблем склейки BIO-тегов. |

---

## 💻 4. Базовый код: Модули пайплайна

### Модуль 1: Метрика Exact-Span Micro-F1 (`metric.py`)
Официальный расчет метрики совпадения спанов для валидации:

```python
from typing import List, Dict, Tuple, Set

def compute_exact_span_f1(
    preds: List[List[Dict]], 
    golds: List[List[Dict]]
) -> Dict[str, float]:
    """
    preds/golds format per text:
    [{"start": 0, "end": 8, "type": "ORG"}, ...]
    """
    total_gold = 0
    total_pred = 0
    total_correct = 0
    
    class_stats = {
        "ORG": {"gold": 0, "pred": 0, "correct": 0},
        "NAME": {"gold": 0, "pred": 0, "correct": 0},
        "GEO": {"gold": 0, "pred": 0, "correct": 0},
    }

    for pred_list, gold_list in zip(preds, golds):
        pred_set: Set[Tuple[int, int, str]] = {(p["start"], p["end"], p["type"]) for p in pred_list}
        gold_set: Set[Tuple[int, int, str]] = {(g["start"], g["end"], g["type"]) for g in gold_list}

        total_pred += len(pred_set)
        total_gold += len(gold_set)
        total_correct += len(pred_set & gold_set)

        for _, _, label in gold_set:
            if label in class_stats:
                class_stats[label]["gold"] += 1
        for _, _, label in pred_set:
            if label in class_stats:
                class_stats[label]["pred"] += 1
        for _, _, label in (pred_set & gold_set):
            if label in class_stats:
                class_stats[label]["correct"] += 1

    precision = total_correct / total_pred if total_pred > 0 else 0.0
    recall = total_correct / total_gold if total_gold > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    metrics = {
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": f1
    }

    for label, stat in class_stats.items():
        p = stat["correct"] / stat["pred"] if stat["pred"] > 0 else 0.0
        r = stat["correct"] / stat["gold"] if stat["gold"] > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        metrics[f"{label}_f1"] = f

    return metrics
```

---

### Модуль 2: Токенизация и выравнивание оффсетов (`dataset.py`)
Преобразование спанов символов в subword BIO-разметку с сохранением границ:

```python
import torch
from transformers import PreTrainedTokenizerFast

LABEL_LIST = ["O", "B-ORG", "I-ORG", "B-NAME", "I-NAME", "B-GEO", "I-GEO"]
LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL = {i: l for i, l in enumerate(LABEL_LIST)}

def tokenize_and_align_labels(examples, tokenizer, max_length=512, stride=128):
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )

    sample_map = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")
    labels = []

    for i, offsets in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        entities = examples["entities"][sample_idx] # [{"start": int, "end": int, "type": str}]
        
        # Создаем маску символов для текста
        text_len = len(examples["text"][sample_idx])
        char_labels = ["O"] * text_len
        
        for ent in entities:
            s, e, t = ent["start"], ent["end"], ent["type"]
            char_labels[s] = f"B-{t}"
            for c in range(s + 1, e):
                char_labels[c] = f"I-{t}"

        token_labels = []
        for start, end in offsets:
            if start == end: # Special token ([CLS], [SEP], [PAD])
                token_labels.append(-100)
            else:
                # Берем метку начала токена
                label_str = char_labels[start] if start < text_len else "O"
                token_labels.append(LABEL2ID.get(label_str, LABEL2ID["O"]))

        labels.append(token_labels)

    tokenized["labels"] = labels
    tokenized["offset_mapping"] = offset_mapping
    return tokenized
```

---

### Модуль 3: Извлечение спанов из предсказаний модели (`postprocess.py`)

```python
from typing import List, Dict

def extract_entities_from_offsets(
    text: str, 
    predictions: List[int], 
    offset_mapping: List[Tuple[int, int]]
) -> List[Dict]:
    """Превращает сырые id классов и offset_mapping обратно в exact char spans."""
    entities = []
    current_entity = None

    for pred_id, (start_char, end_char) in zip(predictions, offset_mapping):
        if start_char == end_char: # Special tokens
            continue
            
        label = ID2LABEL.get(pred_id, "O")
        
        if label.startswith("B-"):
            if current_entity:
                entities.append(current_entity)
            ent_type = label.split("-")[1]
            current_entity = {
                "start": start_char,
                "end": end_char,
                "type": ent_type
            }
        elif label.startswith("I-"):
            ent_type = label.split("-")[1]
            if current_entity and current_entity["type"] == ent_type:
                current_entity["end"] = end_char
            else:
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    "start": start_char,
                    "end": end_char,
                    "type": ent_type
                }
        else: # "O"
            if current_entity:
                entities.append(current_entity)
                current_entity = None

    if current_entity:
        entities.append(current_entity)

    # Дедупликация и зачистка пробелов
    clean_entities = []
    for ent in entities:
        s, e, t = ent["start"], ent["end"], ent["type"]
        # Корректировка пробелов по краям
        ent_text = text[s:e]
        leading_spaces = len(ent_text) - len(ent_text.lstrip())
        trailing_spaces = len(ent_text) - len(ent_text.rstrip())
        
        clean_s = s + leading_spaces
        clean_e = e - trailing_spaces
        if clean_s < clean_e:
            clean_entities.append({"start": clean_s, "end": clean_e, "type": t, "text": text[clean_s:clean_e]})

    return clean_entities
```

---

### Модуль 4: Сервис Инференса FastAPI (`app.py`)

```python
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from transformers import AutoTokenizer, AutoModelForTokenClassification
from postprocess import extract_entities_from_offsets

app = FastAPI(title="Uzbek NER Inference API", version="1.0.0")

MODEL_PATH = "./best_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

class PredictRequest(BaseModel):
    id: str
    text: str

class Entity(BaseModel):
    start: int
    end: int
    type: str
    text: str

class PredictResponse(BaseModel):
    id: str
    entities: List[Entity]

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    inputs = tokenizer(
        request.text,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        max_length=512
    )
    
    offset_mapping = inputs.pop("offset_mapping")[0].tolist()
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=2)[0].tolist()

    entities_raw = extract_entities_from_offsets(request.text, predictions, offset_mapping)
    
    return PredictResponse(
        id=request.id,
        entities=[Entity(**ent) for ent in entities_raw]
    )
```

---

### Модуль 5: Production Dockerfile (`Dockerfile`)

```dockerfile
FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./best_model ./best_model
COPY *.py ./

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

`requirements.txt`:
```
torch>=2.0.0
transformers>=4.40.0
fastapi>=0.110.0
uvicorn>=0.28.0
pydantic>=2.0.0
seqeval>=1.2.2
accelerate>=0.28.0
```

---

## 🔬 5. План проведения экспериментов (Экспериментальная таблица)

Агент должен обучить кандидатов и заполнить следующую таблицу на валидации (1 500 текстов):

```markdown
| Эксперимент | Модель / Подход | Params | Micro-F1 | F1 ORG | F1 NAME | F1 GEO | Latency (CPU) |
|---|---|---|---|---|---|---|---|
| EXP-0 | Baseline (XLM-RoBERTa-base) | 270M | 0.00 | 0.00 | 0.00 | 0.00 | 30 ms |
| EXP-1 | rifkat/uztext-3Gb-BPE-Roberta | 110M | 0.00 | 0.00 | 0.00 | 0.00 | 12 ms |
| EXP-2 | microsoft/mdeberta-v3-base | 86M | 0.00 | 0.00 | 0.00 | 0.00 | 25 ms |
| EXP-3 | Qwen/Qwen2.5-0.5B (TokenClass) | 490M | 0.00 | 0.00 | 0.00 | 0.00 | 45 ms |
| EXP-4 | GLiNER-multi-v2.1 | 300M | 0.00 | 0.00 | 0.00 | 0.00 | 60 ms |
| **FINAL**| **Ensemble (Top-2 Models Weighted)** | — | **0.00** | **0.00** | **0.00** | **0.00** | — |
```

---

## 🎯 6. Анализ ошибок для финальной презентации (Error Analysis)

Агент должен автоматически сохранить топ-20 ошибок на валидации и сгруппировать их по категориям:
1. **Суффиксальные разночтения (Agglutination Suffixes):** Разница в захвате падежа (*Toshkent* vs *Toshkentda*).
2. **Переплетение ORG и GEO:** Названия компаний с гео-топонимами (*O'zbekiston pochtasi*, *Toshkent city*).
3. **Кодовые переключения:** Русско-узбекский сленг и транслитерированные заимствования.
4. **Редкие аббревиатуры:** Нестандартные сокращения ведомств и министерств (*O'zR IIV*, *DVXX*).

---

## 🚀 7. Порядок запуска Cursor Agent

1. `python dataset.py` $\rightarrow$ Проверить выравнивание оффсетов на 10 примерах с ассертами (`assert text[s:e] == label`).
2. Обучить `EXP-1`, `EXP-2`, `EXP-3` с `AdamW(lr=2e-5)`, `Linear Warmup (10%)`, `LinearDecay`, `Batch Size 16/32`, `Epochs 5`.
3. Запустить `evaluate.py` с расчетом `compute_exact_span_f1` на `val.json`.
4. Собрать ансамбль (слияние спанов с порогом уверенности $>0.5$).
5. Сохранить лучшую модель в директорию `./best_model`.
6. Собрать и протестировать Docker:
   ```bash
   docker build -t uzbek-ner:latest .
   docker run -d -p 8000:8000 uzbek-ner:latest
   curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
        -d '{"id": "test_1", "text": "Shavkat Mirziyoyev Toshkent shahrida yangi bank ochdi."}'
   ```
```
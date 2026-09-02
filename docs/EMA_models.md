# Local BERT checkpoints — Exploratory Model Analysis (EMA)

Numbers from `uv run python scripts/ema_models.py` (2026-09-02). Compact JSON: `outputs/ema/summary.json` (gitignored). This is **not** a training run.

The three local dumps are **MaskedLM**, not NER. Public Uzbek NER heads exist, but they do not speak the hackathon ontology `{ORG, NAME, GEO}` with official suffix-inclusive spans. This note asks only: what transfers, what must be reinitialized, and whether anyone is worth using as a distillation teacher.

Tokenizer **document-level** fertility (200 official texts) is already in [`docs/EDA_official.md`](EDA_official.md). Below adds entity-level splits, K-pop vs other NAME, and a chop-head parameter audit.

## Executive summary

**Chop the MLM head, attach a random 7-tag BIO classifier, fine-tune the encoder on official `train.jsonl`. Do not distill public NER logits.**

| Decision | What |
|---|---|
| **Use as encoder FT** | All three local checkpoints: `exp0` XLM-R-base, `exp1` uztext 6L, `exp2` mDeBERTa-v3-base |
| **Chop + retune** | Yes. That **is** the recipe. Encoder+embeddings transfer; `lm_head` / pooler / ELECTRA generator are dropped; a new `classifier` of **5 383** params (`768×7+7`) is randomly initialized (**0.002–0.007%** of the NER model) |
| **Distill teacher logits** | **No.** `risqaliyevds/xlm-roberta-large-ner` is 18-type news NER (24L, ~559 M) and **strips case endings**. `jamshidahmadov/roberta-ner-uz` is gated, 1-epoch XLM-R-base on the same news set, card lists B-\* only |
| **Zero-shot spans** | Impossible from MLM. A randomly initialized TokenClassification head on official text emits hundreds of junk spans (including on empty docs) |
| **Frozen linear probe** | 64+64 official docs: token acc ≈ majority-`O` (~87–90%), exact-span F1 **0.05–0.13**. The Uzbek prior is in the encoder but is **not** a usable NER head. Full FT is required |
| **They ate Uzbek?** | Yes, as **language models**. uztext was pretrained on ~3 GB Uzbek news (Latin+Cyrillic). XLM-R and mDeBERTa saw Uzbek inside CC100. That prior is why encoder FT is the right bet — not because their NER ontologies match |

## Verdict table

| Model | Use as encoder FT | Distill teacher | Ignore | Why |
|---|---|---|---|---|
| `exp0_xlm_roberta_base` (XLM-R-base MLM) | **Yes — Mix A first** | No | — | Organizer stack. 12L / 277.5 M NER params. Tied LM head. Best multilingual prior for K-pop / mixed / platforms |
| `exp1_uztext_roberta` (Uzbek RoBERTa MLM 6L) | **Yes — Mix A, fertility/VRAM** | No | — | 6L / **82.9 M**. Fuses `Toshkentda` / `Rossiyada` into one piece (BIO-friendly). Weaker on K-pop NAME pieces than XLM-R / mDeBERTa |
| `exp2_mdeberta_v3_base` (mDeBERTa-v3 MLM) | **Yes — Mix A, mixed script** | No | — | 12L / 278.2 M. Drop generator bin (~906 MB) and `word_embeddings._weight` duplicate (~193 M). Disentangled attention; tokenizer regex warning remains |
| `jamshidahmadov/roberta-ner-uz` | No (gated; encoder is XLM-R-base we already have) | **No** | **Yes** | HF gated. Card: risqaliyevds news NER, 1 epoch, lr `1e-6`. Lists B-LOC/PERSON/ORG/PRODUCT/DATE/TIME/LANGUAGE/GPE — no I-\*. Token acc 0.979 is O-class |
| `risqaliyevds/xlm-roberta-large-ner` | No (24L / 559 M; 6 GB cannot full-FT) | **No** | **Yes as teacher** | 37 BIO tags, 18 types. Only PERSON/ORG/LOC/GPE/FACILITY map. Card example tags `Rossiyada` as GPE=`Rossiya` (suffix **outside**). Same span poison as UzNER |
| GLiNER `urchade/gliner_multi-v2.1` | Separate line later | No | Not this BERT track | Already in deps. Optional span model; not analyzed here |

## 1. The three dumps are MaskedLM

`config.json` / architectures, local files only (`HF_HUB_OFFLINE=1`):

| Alias | Architecture | L / H / heads | Vocab (config) | On-disk MLM tensors | NER model after chop |
|---|---|---|---:|---:|---:|
| `exp0_xlm_roberta_base` | `XLMRobertaForMaskedLM` | 12 / 768 / 12 | 250 002 | 278.9 M | **277.5 M** |
| `exp1_uztext_roberta` | `RobertaForMaskedLM` | **6** / 768 / 12 | 52 000 | 123.5 M (untied decoder) | **82.9 M** |
| `exp2_mdeberta_v3_base` | `deberta-v2` (no `architectures` field) | 12 / 768 / 12 | 251 000 | 473.2 M (incl. duplicate embed) | **278.2 M** |

uztext’s public “83 M” is the **encoder**. The extra ~40.6 M on disk is `lm_head.decoder.weight` (untied, `52 000×768`) plus the usual dense/LN/bias — all dropped for NER.

mDeBERTa-v3 is ELECTRA-style. Ignore `pytorch_model.generator.bin` (~906 MB) entirely. The discriminator dump also carries `deberta.embeddings.word_embeddings._weight` (~193 M) which TokenClassification does **not** load. Loaded NER encoder is 278.2 M, matching the card’s “86 M backbone + 190 M embeddings”.

XLM-R’s dump still contains flax / ONNX / TF / `.bin` **and** `model.safetensors` (~6 GB total). Only safetensors is needed.

### Chop-head accounting (what transfers vs reinit)

`AutoModelForTokenClassification.from_pretrained` on each MaskedLM directory, 7 labels = official BIO (`O, B-ORG, I-ORG, B-NAME, I-NAME, B-GEO, I-GEO`).

| Checkpoint | Transfer (embeddings+encoder) | Dropped | Reinit (`classifier`) | Reinit % of NER model |
|---|---:|---|---:|---:|
| XLM-R-base | 277 453 056 | `lm_head.*` (0.84 M; decoder **tied** so not extra), `roberta.pooler` (0.59 M) | **5 383** | 0.0019% |
| uztext | 82 858 752 | full `lm_head` including **untied** decoder (40.6 M) | **5 383** | 0.0065% |
| mDeBERTa-v3 | 278 218 752 | `lm_predictions.*`, `mask_predictions.*`, `word_embeddings._weight`, generator file | **5 383** | 0.0019% |

Load report in every case: **MISSING** `classifier.weight` / `classifier.bias` (newly initialized); **UNEXPECTED** MLM / pooler / mask-prediction keys (safe to ignore).

So “отрезать голову и натюнить” is not a metaphor: **99.993%+ of the NER net is the pretrained encoder**. The head is a 7-way linear. Fine-tune the encoder; do not freeze it (see probe below).

## 2. Distillation: label maps and why we skip it

Hackathon types: `ORG`, `NAME`, `GEO`. A teacher is useful only if its BIO types map cleanly **and** its span policy matches (suffixes inside the mention). Soft hidden-state distillation (PKD) is theoretically possible without a type map, but it still needs a teacher forward on every train step — expensive on 6 GB — and buys little versus just fine-tuning uztext and XLM-R.

### `risqaliyevds/xlm-roberta-large-ner` (config + card only; **no large weights pulled**)

Hub `id2label` is 37 tags / 18 entity types. Architecture `XLMRobertaForTokenClassification`, 24 layers, hidden 1024, **558.9 M** params, ~2.2 GB `model.safetensors`. Trained on news (`risqaliyevds/uzbek_ner` family).

| Teacher type | Map | Distill BIO? |
|---|---|---|
| PERSON | NAME | Yes, with a map |
| ORG | ORG | Yes |
| LOC, GPE, FACILITY | GEO | Yes, after collapsing three types |
| DATE, TIME, MONEY, PERCENT, QUANTITY, PRODUCT, EVENT, WORK_OF_ART, LANGUAGE, CARDINAL, ORDINAL, NORP, LAW | — | **No — 13/18 types are noise** |

Mappable **types**: 5/18 = 27.8%. Distilling the full 37-way softmax would push PRODUCT/NORP/DATE mass onto the student. Even a masked loss on `{PERSON,ORG,LOC,GPE,FACILITY}` is the wrong span policy:

Hub card example, text `Shavkat Mirziyoyev Rossiyada rasmiy safarda bo'ldi.`:

| | Span | Type |
|---|---|---|
| Teacher (card) | `Shavkat` 0–7, `Mirziyoyev` 8–18, **`Rossiya` 19–26** | PERSON / PERSON / GPE |
| Official policy | `Shavkat Mirziyoyev` 0–18, **`Rossiyada` 19–28** | NAME / GEO |

`da` is left **outside** GEO. Exact-span micro-F1 scores that as a miss. This is the same agglutination bug measured on UzNER / UzLegalNER in the EDA. Teaching a student those logits would train it to drop suffixes.

VRAM: teacher fp16 ≈ 1.1 GB weights plus activations, on top of a 278 M student, optimizer, and batch. Not a hackathon-time setup on RTX A1000 6 GB.

### `jamshidahmadov/roberta-ner-uz` (gated — weights not downloaded)

HF `gated: auto` (same snapshot as `jmshd/roberta-ner-uz`). Config/tokenizer 401 without a token. README is public:

- Base: `FacebookAI/xlm-roberta-base` — **the encoder we already have as `exp0` MLM**
- Data: `risqaliyevds/uzbek_ner` (~19k), 1 epoch, lr `1e-6`, batch 4
- Card types (B-\* only listed): LOC, PERSON, ORG, PRODUCT, DATE, TIME, LANGUAGE, GPE
- Reported val accuracy 0.979 / P=R=F=0.97 — typical **token** scores dominated by `O`

Map: PERSON→NAME, ORG→ORG, LOC/GPE→GEO; drop PRODUCT/DATE/TIME/LANGUAGE. The B-only card listing matches the broken `ner_prepared_uzbek` conversion already rejected in EDA. Even ungated, this is a weak news NER head on silver we would at most use as Mix B **data**, not as a teacher over official gold.

### PKD-style hidden-state distill

Possible in principle (student mimics teacher encoder layers; label spaces never meet). Not worth the queue: two BERT forwards per step, no ontology gain, and the useful Uzbek prior is already **inside** the three local checkpoints. Fine-tune them.

**Distill recommendation: no.** If Mix B helps, use mapped **data** (risqaliyevds mentions → spans + suffix rewrite), not teacher logits.

## 3. Zero-shot: MLM cannot emit spans

Without a trained TokenClassification (or span) head, MaskedLM predicts token identities, not BIO. Converting the checkpoint and running a **random** head on eight official documents (CUDA, max length 256):

| | Random-head spans / doc (XLM-R) | Gold |
|---|---:|---|
| K-pop fanfic (`Tae` / `Jungkook` / `Jk`) | 252 | 8 NAME mentions |
| Cyrillic WHO/Lebanon news (`Ливанда`) | 249 | ORG+GEO with suffixes |
| Empty doc `бу холатга йиг'лаш керак` | 10 NAME fragments (`бу`, `хол`, `ат`, …) | 0 |

The empty document is the tell: a random 7-way classifier tags almost every subword. **Applicability of the checkpoints is as encoders after FT, not as off-the-shelf NER.**

`jamshidahmadov` could not be probed (gated). The risqaliyevds **card** already shows the span mismatch we would have measured live: K-pop fanfic and `Toshkentda`-style GEO are outside its news-NER comfort zone, and suffixes are excluded.

## 4. Tokenizer fertility — new numbers only

EDA (200 docs, seed 42): uztext 2.31 tok/word vs XLM-R 2.60 vs mDeBERTa 2.55; Cyrillic is where uztext wins (2.69 vs 3.54 / 3.09). Unchanged.

**New:** 1 500-doc official-train sample, **7 150 gold entity surfaces** (not document tok/word).

| | XLM-R | uztext | mDeBERTa |
|---|---:|---:|---:|
| Mean subwords / GEO | 3.41 | **2.84** | 3.52 |
| Mean subwords / ORG | 4.15 | **3.80** | 4.31 |
| Mean subwords / NAME | 3.82 | 3.84 | **3.66** |
| Single-piece entity % | 12.2 | **15.2** | 7.1 |
| Suffix-bearing entities, mean pieces | 3.33 | **2.93** | 3.24 |
| Lemma entities, mean pieces | 2.42 | 2.49 | 2.40 |
| Last piece **is** the suffix (%) | 95.5 | **64.5** | 96.2 |
| K-pop NAME mentions (n=178), mean pieces | 2.81 | 3.11 | **2.64** |
| Other NAME (n=2 046), mean pieces | 3.91 | 3.90 | 3.75 |

K-pop vs other NAME is the plot twist: **uztext does not win on fandom names**. `Jungkookning` → uztext `Ju / ngko / ok / ning`, XLM-R `Jung / ko / ok / ning`, mDeBERTa `Jung / kook / ning` (best). EDA’s “uztext always wins fertility” slogan is about Uzbek stems and Cyrillic, not Latin K-pop.

**Agglutination splits (the BIO boundary):**

| Surface | XLM-R | uztext | mDeBERTa |
|---|---|---|---|
| `Toshkent` | `Toshkent` | `Toshkent` | `▁` + `Toshkent` |
| `Toshkentda` | `Toshkent` + **`da`** | **`Toshkentda`** (one piece) | `Toshkent` + **`da`** |
| `Rossiyada` | `Rossiya` + `da` | **`Rossiyada`** | `Rossi` + `yada` |
| `Telegramda` | `Telegram` + `da` | `Telegram` + `da` | `Telegram` + `da` |
| `Jungkookning` | `Jung/ko/ok` + `ning` | `Ju/ngko/ok` + `ning` | `Jung/kook` + `ning` |

uztext fuses locative GEO (`Toshkentda`, `Rossiyada`) so BIO does not have to learn “`da` continues the span”. XLM-R/mDeBERTa almost always peel `da`/`ning` into their own piece (95–96% of suffix entities) — recoverable if the head is trained on official gold, fatal if the teacher leaves `da` as `O`.

mDeBERTa still emits a transformers regex warning (EDA); treat its pieces as ±a few percent. Tokenizer `vocab_size` 250 101 vs config 251 000.

## 5. Frozen linear probe (CUDA, then CPU)

Lock: `outputs/.gpu.lock` via `flock`. Short fp16 forwards on RTX A1000 6 GB; lock released before sklearn.

64 official-train docs to fit a balanced logistic regression on frozen last-layer states; 64 disjoint docs to eval. Max length 256. Not a leaderboard number.

| Encoder | Majority-O token acc | Frozen-linear token acc | Exact-span F1 | tp / fp / fn |
|---|---:|---:|---:|---|
| XLM-R | 0.876 | 0.864 | **0.103** | 58 / 734 / 272 |
| uztext | 0.875 | 0.865 | **0.127** | 64 / 616 / 266 |
| mDeBERTa | 0.882 | 0.896 | **0.051** | 24 / 587 / 306 |

Token accuracy stays on the `O` ceiling. Span F1 is low with massive false positives (sklearn `lbfgs` also hit `max_iter=200`). Reading: **there is a weak entity signal in the frozen encoder, far from a system.** Full fine-tuning of encoder + head is the experiment, not a linear probe and not logit distillation.

## Applicability: they ate Uzbek — as LMs

| Checkpoint | What it ate | What that buys | What it does not buy |
|---|---|---|---|
| XLM-R-base | CC100, 100 languages including Uzbek | Mixed script, Russian code-switch, Latin brands (`Telegram`, `Jungkook`) | Official BIO, suffix policy, social/fandom NAME mass as a NER task |
| uztext 6L | ~2 M Uzbek news articles, ~3 GB, Latin+Cyrillic | Cheap VRAM, fused GEO locatives, better Cyrillic fertility | Depth for long ORG names and K-pop; it is still 6 layers |
| mDeBERTa-v3 | CC100, 2.5 T, disentangled attention | Mixed-script NLU prior (XNLI > XLM-R in the card) | A NER head; generator is unused |

Ontology mismatch of **public NER models** does not make the **local MLM encoders** useless. The useful thing they ate is Uzbek (and multilingual) distributional structure. The useless thing is any attached news-NER classifier.

## Recommended experiment order (unchanged, now with EMA evidence)

1. Mix A + `exp0` XLM-R — chop MLM, 7-tag head, sliding window.
2. Mix A + `exp1` uztext — same recipe; expect GEO suffix wins, possible NAME/K-pop loss.
3. Mix A + `exp2` mDeBERTa — same recipe; ignore generator.
4. Sliding window on the winner.
5. Mix B (mapped risqaliyevds **data**, suffix rewrite) on the winner only — **not** teacher-KD.
6. GLiNER / LoRA-Qwen only if BERT Mix A has a number.

## Risks (model-side)

**Random head ≠ NER.** Never submit an uninitialized classifier.

**Teacher span policy.** Any logit distill from 18-type news NER teaches `Rossiya` not `Rossiyada`.

**uztext capacity.** Fertility on GEO is real; 6 layers may lose long ORG and mixed Russian. Measure, don’t assume.

**mDeBERTa tokenizer.** Regex warning + dummy `▁` pieces. Offsets still work with the slow/fast pair we load; verify `return_offsets_mapping` on a handful of gold spans before a long run.

**VRAM.** Three local base/6L models fit. XLM-R-large teacher does not, as a training partner.

**Gated hub.** Do not block Mix A on `jamshidahmadov` access.

## How to reproduce

```bash
# CPU configs, chop-head load, fertility, Hub configs (large weights not downloaded)
HF_HUB_OFFLINE=1 uv run python scripts/ema_models.py --skip-gpu

# Full: waits up to 45 min on outputs/.gpu.lock, then short CUDA forwards
uv run python scripts/ema_models.py
```

Artifacts (gitignored): `outputs/ema/summary.json`.

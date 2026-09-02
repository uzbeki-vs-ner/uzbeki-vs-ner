# Official Uzbek NER — EDA and ontology note

Numbers from `uv run python scripts/eda_official.py` (2026-09-02). Compact JSON: `outputs/eda/summary.json` (gitignored). This is **not** a training run.

BERT-family NER here is mostly **ontology alignment**: the model must reproduce the organizer’s three-class, exact-span, suffix-inclusive policy — not “generic NER”. Extra data that uses CoNLL-style types, WikiANN LOC, 18-type UzNER, or legal/PII/medical inventories is useful only after an explicit map, and is often harmful if the **span policy** differs.

## Executive summary

**Train on official `train.jsonl` (13 000). Report only on official `dev.jsonl` (1 500). Do not mix external rows into the scored dev set.**

| Decision | What |
|---|---|
| **Must train on** | Official train, all three labels, both scripts, including the 18% empty documents |
| **Worth trying as mapped silver** | `risqaliyevds_uzbek_ner` (PERSON/ORG/GPE/LOC only, after mention→span align + suffix rewrite); optionally the **real** subset of UzNER-100k `PER/ORG/GPE/LOC` after the same span rewrite |
| **Ignore / reject** | `uz_medner`; Rubai PII (NAME-only synthetic, wrong task); legal extra types (`DATE/MONEY/DOCNO/POSITION/TIN/CADASTRE/LAW`); UzNER `PRODUCT/POSITION/NORP/EVENT/DATE/TIME/numeric`; `ner_prepared_uzbek` (B-only BIO, noisy duplicate of risqaliyevds) |
| **Legal NER** | Domain-shifted and **suffix-excluding**. Do not add to the main mix. `BANK/COURT→ORG` is the only extra type that matches the official ORG definition, and even that is low-overlap |
| **Models** | Fine-tune the three local BERT checkpoints. Qwen full FT on RTX A1000 6 GB is a poor default; LoRA/GLiNER later if there is spare time |
| **Tokenizer takeaway** | `uztext` fragments Uzbek (especially Cyrillic) less than XLM-R / mDeBERTa, but is a 6-layer 83 M model. Capacity vs fertility is the real trade-off, not “uztext always wins” |

The single highest-risk mismatch with public Uzbek NER is **agglutination**: official gold **includes** attached case endings (`Toshkentda`, `Qodirovning`); UzNER and UzLegalNER typically put the suffix **outside** the span. Exact-span micro-F1 scores that as a full miss.

## Official dataset profile

Recomputed stats match `data/official/dataset_manifest.json`.

### Volume and classes

| Split | Docs | With entities | Empty | Entities | ORG | NAME | GEO |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 13 000 | 10 614 (81.65%) | 2 386 (18.35%) | 66 083 | 23 420 (35.4%) | 21 218 (32.1%) | 21 445 (32.5%) |
| dev | 1 500 | 1 225 (81.67%) | 275 (18.33%) | 7 698 | 2 659 (34.5%) | 2 319 (30.1%) | 2 720 (35.3%) |

Classes are nearly balanced. Empty documents are **intentional** (social/ad noise with no entity) and must stay in training so the model learns to predict nothing. Mean ~5 entities/doc on train.

### Label co-occurrence (train documents)

| Signature | n | % of docs |
|---|---:|---:|
| ORG only | 2 498 | 19.2 |
| EMPTY | 2 386 | 18.4 |
| GEO+ORG | 1 900 | 14.6 |
| GEO+NAME+ORG | 1 703 | 13.1 |
| NAME only | 1 702 | 13.1 |
| GEO only | 1 396 | 10.7 |
| NAME+ORG | 966 | 7.4 |
| GEO+NAME | 449 | 3.5 |

Pairwise document co-occurrence: GEO+ORG 3 603, NAME+ORG 2 669, GEO+NAME 2 152. Multi-type docs are common; a flat three-class head with no nesting is the right inductive bias (matches the labeling guide).

Dev signatures follow the same shape (ORG-only 288, EMPTY 275, GEO+ORG 219, …).

### Script (Latin / Cyrillic / mixed)

| Split | Latin docs | Mixed | Cyrillic | Other |
|---|---:|---:|---:|---:|
| train | 7 840 (60.3%) | 2 957 (22.7%) | 2 202 (16.9%) | 1 |
| dev | 892 (59.5%) | 337 (22.5%) | 271 (18.1%) | 0 |

Entities are more Latin than documents: train entities 50 064 Latin / 15 552 Cyrillic / 352 mixed / 115 other. Mixed **documents** (code-switching, Russian+Uzbek, Latin names in Cyrillic news) are ~23% — a reason not to drop multilingual encoders even if `uztext` tokenizes pure Uzbek better.

### Text length (truncation risk)

Train characters: median 151, mean 497, p90 1 089, p95 1 864, p99 4 854, **max 22 668**.  
Train whitespace words: median 21, mean 64, p90 136, p95 230, p99 610, max 3 088.  
Docs with >300 words: 436 train / 45 dev; >400 words: 279 / 25.

Most posts fit in 512 subwords; a fat tail of news articles does not. Use sliding windows (`stride≈128`) for all BERT runs. On a 200-doc sample, 19 texts exceeded 512 XLM-R tokens (11 for uztext, 16 for mDeBERTa).

### Entity spans

| | chars (train) | words (train) |
|---|---|---|
| all | mean 11.5, p50 9, p99 48, max 200 | mean 1.53, p50 1, p99 5 |
| GEO | mean 10.6, p50 10, max 68 | — |
| ORG | mean 14.0, p50 9, max 200 | longest class (official names) |
| NAME | mean 9.8, p50 8, max 74 | — |

**Integrity:** 0 overlapping pairs, 0 nested pairs, 0 invalid offsets, 0 duplicate spans, 0 empty spans, 0 leading/trailing whitespace. The official files are clean against the stated flat exact-span schema.

Edge punctuation inside the span: 82 train / 6 dev (quotes/brackets leftover). Small; not a data-bug class.

Unique surfaces (train): GEO 7 335, ORG 9 660, NAME 10 336 — high type diversity, especially NAME.

### Agglutination (the metric killer)

Official rule: attached endings **are inside** the span (`Toshkentda`, `Qodirovning`, `KFCda`, `Mobiuz'dan`). Separate particles are not (`KFC da` → `KFC`).

| Split | Entities with a detected attached suffix | % |
|---|---:|---:|
| train | 7 957 | 12.0 |
| dev | 1 004 | 13.0 |

GEO carries most locative/genitive endings (`da/да`, `ga/га`, `ning/нинг`, `dagi/даги`, `dan/дан`). NAME is dominated by `ning/ni` (Uzbek genitive/accusative on person names). ORG suffixes are rarer.

Stem vs inflected pairs in train (same class, same stem): **1 747** pairs. Typical:

- GEO: `Oʻzbekiston` (640) vs `Oʻzbekistonda` (271) / `-ga` (120) / `-dagi` (93) / `-ning` (86)
- GEO: `Toshkent` vs `Toshkentda` / `Toshkentdagi` (also on dev)
- NAME: `Jungkook` (679) vs `Jungkookning` (83) / `Jungkookni` (46)

A model that emits the lemma without the case ending scores **0** on that mention. Public datasets that strip suffixes will teach exactly that error.

### Top surfaces (train) — domain, not Wikipedia

**GEO:** `Oʻzbekiston` 640, `Эрон`/`Eron` ~277–280, `Oʻzbekistonda` 271, `АҚШ`/`AQSh`, `Ўзбекистон`, `Toshkent`. Country/city news + locative duplicates.

**ORG:** `Telegram` 515, `Instagram` 426, `Oqtepa Lavash`/`Oqtepa lavash` ~304+278, `Pepsi` 278, `Facebook` 270, `Murad Buildings` 253, `YouTube` 206. Social platforms, local QSR/real-estate brands, CPG — Brand Analytics social+media, not academic newswire.

**NAME:** `Tae` 828, `Jungkook` 679, `Jk` 597, `Jimin` 301, `Murod Nazarov` 258, `Jin` 242, `Taehyung` 222, `Yoongi` 182. K-pop fandom plus local public figures. Wiki/legal NER will not cover this NAME mass.

Dev repeats the same head types (Telegram, Murad Buildings, Oqtepa, Jungkook/Jk/Jimin). Train/dev look i.i.d. at the surface level.

### ORG / GEO / NAME ambiguity (same string, different label)

Train: **169** surfaces with more than one gold label. Dev: 17. This is labeled, not noise — the guide says class follows **contextual role**.

| Surface | Typical split | Reading |
|---|---|---|
| `Oʻzbekiston` | GEO 640 / ORG 7 | country vs institution-as-name fragment |
| `Eron`, `Ўзбекистон`, `Исроил`, `Toshkent` | almost all GEO, rare ORG | same |
| `Murad Buildings` | ORG 253 / GEO 3 | brand vs place/complex |
| `Andijon` | GEO 39 / ORG 20 | city vs football club (and similar: `Paxtakor`, `Bunyodkor`, `Soʻgʻdiyona`) |
| `Navoiy` (dev) | GEO / NAME | city vs person |
| `Tehron` (dev) | GEO / NAME | toponym vs personal name |
| `Magic City` | GEO / ORG | mall/place vs org |

Context classifiers matter more than a gazetteer. Copying a LOC gazetteer from WikiANN/UzNER will systematically over-tag sports clubs and developer brands as GEO.

## Ontology vs public datasets

Hackathon inventory is **only** `{ORG, NAME, GEO}` with official boundary rules. Everything else is a mapping problem.

### Keep / map / reject

| Source | Size seen | Map to hackathon | Action | Why |
|---|---|---|---|---|
| **official train/dev** | 13k + 1.5k | native | **KEEP — only gold** | Target ontology and span policy |
| **risqaliyevds_uzbek_ner** | 19.6k docs, 174k mentions | PERSON/PER→NAME, ORG→ORG, GPE/LOC/FAC→GEO | **MAP, then align spans** | Highest surface overlap with official (see below). Mention lists, not offsets. Drop DATE/EVENT/MONEY/PRODUCT/NORP/… (~16% of mentions) |
| **ner_prepared_uzbek** | 19.6k sent | same as risqaliyevds | **REJECT as a trainer** | Same source, ClassLabel is **B-\* only** (no I-\*). Treat as a broken conversion |
| **uzner_100k** train | 100k sent, 159k ents (30k synthetic) | PER→NAME, ORG→ORG, GPE/LOC→GEO; FAC→GEO **with care** | **MAP only core types, real subset, after suffix rewrite** | 18-type ontology. Keep-rate if mapped naively: **44.5%**. Rest is DATE/POSITION/PRODUCT/NORP/numeric/LAW/DOC — reject. Suffix policy is the opposite of official |
| **uznlp_uzbek_ner_gold** | 4.2k sent / 59.6k tokens | PER→NAME, ORG→ORG, LOC→GEO | **optional small MAP** | Token BIO. Keep-rate on tagged tokens **63.9%**; drop WORK/TEMPORAL/NUMERIC/MONEY/MISC |
| **wikiann_uz** | 3×1k | PER→NAME, ORG→ORG, LOC→GEO | **low priority** | Types map 1:1 but weak Wikipedia silver, tiny, no official GEO facilities/brands |
| **uzlegalner_v3** | 4.6k sents | PER→NAME, ORG→ORG, LOC→GEO | **REJECT for main mix** | Keep-rate 43.6%. **0** entities with attached suffix; examples like `Toshkent`\|`Toshkentdagi` — suffix left outside. Surface overlap with official ≈ 0 |
| **uzbek_legal_ner_full** | 1.2k core rows; 14k goldready | PER→NAME, ORG+BANK+COURT→ORG, LOC→GEO | **REJECT** | Silver/unverified, many missing offsets (890/1179 core; 5343/14036 goldready). Extra types (TIN, CADASTRE, DOCNO, LAW, POSITION) are not the task |
| **rubai_ner_150k_personal** | 25k of 142.7k scanned | NAME only | **REJECT for v1** | Synthetic PII. Type bag is TEXT/PHONE/ADDRESS/DATE; NAME is **2.4%** of type mentions. Wrong span task |
| **uz_medner** | 20k tokens / 945 docs | — | **REJECT** | DISEASE/BODY/SYMPTOM/… A handful of `ORG`/`LOCATION`/`DOCTOR` is not worth the domain shift |

### Cheap surface overlap with official unique strings (casefold)

Intersection over official unique surfaces after mapping. This is **not** mention-level recall; it only asks “does this extra corpus even talk about the same names?”.

| External (mapped) | ORG recall of official | NAME | GEO | Jaccard (ORG/NAME/GEO) |
|---|---:|---:|---:|---|
| risqaliyevds | **12.6%** | **10.5%** | **18.6%** | 0.062 / 0.051 / 0.066 |
| uzner_100k train | 4.0% | 2.5% | 6.8% | 0.026 / 0.017 / 0.039 |
| uzlegalner_v3 | 0.05% | 0.05% | 0.39% | ~0 |

Risqaliyevds is the only extra source that shares a non-trivial slice of official ORG/NAME/GEO strings (Telegram-class platforms, Uzbek places). UzNER-100k is large but **lexically far** from this social/media mix (K-pop names, Oqtepa, Murad Buildings). Legal NER does not overlap.

### What “18-type UzNER” actually does to this task

UzNER-100k train label mass (159 442 entities):

| Bucket | Types | n | % |
|---|---|---:|---:|
| Mappable if rewritten | PER, ORG, GPE, LOC, FAC | 70 977 | 44.5 |
| Must drop | DATE 18.6k, POSITION 9.2k, PRODUCT 8.0k, CARDINAL 7.6k, plus TIME/NORP/LAW/MONEY/EVENT/ORDINAL/DOC/PERCENT/QUANTITY at 5k each | 88 465 | 55.5 |

Fine-grained distinctions that **conflict** with the official guide:

| UzNER type | Official policy |
|---|---|
| GPE vs LOC vs FAC | Official **GEO** is one class: countries **and** streets, metro, airports, stadiums, markets, housing complexes. Mapping GPE+LOC+FAC→GEO is the least-bad collapse. Mosques/churches are **ORG** in the official guide, **FAC** in UzNER — residual error |
| POSITION | Titles are **not** NAME (`prezident`, `professor` stay out) |
| PRODUCT | Official tags the **brand** only (`Samsung Galaxy S24` → `Samsung`). PRODUCT spans are too wide |
| NORP | `toshkentlik` / `российский` are explicitly unlabeled |
| DATE/TIME/MONEY/… | Out of scope |

UzNER annotation §2.4: case suffixes are **outside** the entity when the tokenizer splits them (`Xiva` + `da`). Official: `Xivada` is one GEO. Measured: 6 725 UzNER train entities still contain an attached suffix (inconsistent tokenization), vs 12% of official entities **by design**.

30 000 / 100 000 UzNER train sentences are synthetic. If used at all, restrict to `meta.is_synthetic == false` / real partition.

### Skipped files (same content or too redundant)

- `uzner_{dev,test,hard_eval}_bioes.jsonl` and all `*.conll` (duplicate of jsonl)
- `uzner_gold_candidate_bioes.*`
- UzLegalNER CoNLL splits (same sentences as jsonl)
- Legal full `*.json` / `*.csv` / extended-augmented duplicates
- Rubai: 25 000 / 142 704 rows scanned for type histograms (enough to reject)

## Model / tokenizer implications

Local checkpoints, 200 random official train texts (seed 42). Fertility = subword tokens per whitespace word (special tokens excluded).

| Checkpoint | Params / depth | Vocab | tok/word mean | median | tok/char | tok / entity-word | Latin | Mixed | Cyrillic | sample >512 tok |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `exp0_xlm_roberta_base` | 278 M / 12L | 250 002 | 2.60 | 2.36 | 0.339 | 2.47 | 2.24 | 2.64 | **3.54** | 19 / 200 |
| `exp1_uztext_roberta` | **83 M / 6L** | 52 000 | **2.31** | **1.88** | **0.307** | **2.18** | 2.24 | **2.16** | **2.69** | **11 / 200** |
| `exp2_mdeberta_v3_base` | 278 M / 12L | 250 101 | 2.55 | 2.45 | 0.333 | 2.42 | 2.37 | 2.50 | 3.09 | 16 / 200 |

**Semantics of these numbers**

- On **Latin** Uzbek the three tokenizers are close (~2.2–2.4). The “uztext always wins fertility” slogan is overstated for Latin-only posts.
- On **Cyrillic**, XLM-R is clearly worse (3.54 vs uztext 2.69 vs mDeBERTa 3.09). ~17% of docs and ~24% of entities are Cyrillic; ignoring that tax costs window budget and boundary precision.
- uztext’s median 1.88 tok/word means many Uzbek stems are single pieces — better for `Toshkentda`-style BIO. It is still a **shallow** RoBERTa (6 layers). Expect it to win speed/VRAM and maybe GEO/ORG stems, and to lose on long ORG names, mixed Russian, and rare Latin NAME (K-pop) unless capacity is enough.
- XLM-R is the organizer baseline and the multilingual prior for `Telegram` / `Jungkook` / mixed posts. Fertility is the tax you pay for that prior.
- mDeBERTa sits between them on fertility, with disentangled attention that often helps NER. Tokenizer load from the local dir emitted a transformers regex warning; treat mDeBERTa fertility as ±a few percent, not gospel. Still in the XLM-R ballpark, not uztext.

**Hardware (RTX A1000 6 GB, 31 GB RAM)**

- uztext 83 M: comfortable full FT, larger batch.
- XLM-R / mDeBERTa 278 M: full FT with small batch, grad accumulation, fp16; 512-length windows.
- Qwen2.5-0.5B full FT: tight on 6 GB once optimizer states are included; only consider LoRA/QLoRA, not as the first line.
- Sliding window is mandatory for the length tail; fertility differences change how many official docs overflow 512.

## Recommended data mix for experiments

Never evaluate extra data on a merged “dev”. Official `dev.jsonl` is the only number that matters for the leaderboard analogue.

### Mix A — official only (default, run first)

```
train: data/official/train.jsonl
dev:   data/official/dev.jsonl
```

This is the ontology. All architecture / LR / window / BIO vs BIOES comparisons should be settled here.

### Mix B — mapped news silver (optional, after Mix A has a baseline)

1. Take **risqaliyevds** mentions with labels in `{PERSON, PER, ORG, GPE, LOC, FAC}`.
2. Align to character offsets on `text` (unique match; drop ambiguous/unfound).
3. **Rewrite spans to official suffix policy**: if `text[end:end+k]` is an attached ending (`da/dan/ga/ni/ning/dagi/…` and Cyrillic mirrors), extend `end`.
4. Drop DATE/EVENT/MONEY/PRODUCT/NORP/….
5. Concatenate with official train; **do not** replace official rows.
6. Cap or downsample so silver cannot drown the 13k gold (start with ≤ official size).

Same pipeline can be applied to UzNER **real** `PER/ORG/GPE/LOC` (optionally FAC→GEO) if Mix B helps. Do **not** add UzNER PRODUCT/POSITION/NORP/DATE. Prefer non-synthetic rows.

### Mix C — do not run as NER train

- WikiANN-uz: too small/weak unless used only as a cheap smoke test of the mapper.
- Legal v3 / legal full: domain + inverted suffixes + extra types.
- Rubai PII: synthetic NAME/PII, not GEO/ORG news.
- MedNER: different language game.
- `ner_prepared_uzbek`: duplicate with broken BIO.

### Experiment order (BERT line, no LLM FT required)

1. Mix A + `exp0` XLM-R (repro organizer stack).
2. Mix A + `exp1` uztext (fertility / VRAM).
3. Mix A + `exp2` mDeBERTa (capacity + mixed script).
4. Sliding window on the winner.
5. Mix B on the winner only; measure official-dev ΔF1. If GEO/NAME suffix errors fall and ORG sports-club errors rise, Mix B is a net loss — drop it.
6. Only then consider GLiNER or LoRA-Qwen as a separate line.

## Risks

**Label shift.** Official ORG includes platforms, clubs, brands, ministries, mosques. WikiANN/UzNER ORG is mostly institutions. Official GEO includes facilities that others call FAC/LOC. Official NAME is people (including fandom nicknames), never titles. Unmapped extra types (`POSITION`, `PRODUCT`, `NORP`) will move mass onto the wrong BIO tags.

**Span policy / suffixes.** ~12–13% of official entities are inflected in-span. UzNER §2.4 and UzLegalNER examples (`Toshkentda` gold as `Toshkent`) teach the opposite boundary. Exact-span F1 has no partial credit. **Do not train span models on extra data until suffixes are rewritten to the official rule.**

**Domain shift.** Official head types are Telegram/Instagram, Oqtepa Lavash, Murad Buildings, K-pop names, Iran/US/Uzbekistan news. Legal contracts, medical notes, synthetic PII, and Wikipedia lists do not share that prior. Surface-overlap numbers above quantify it.

**Ambiguity / no gazetteer shortcut.** `Andijon`, `Paxtakor`, `Bunyodkor`, `Murad Buildings`, `Oʻzbekiston` change class with context. Extra GEO-heavy data will over-predict GEO on clubs and developers.

**Script mix.** 23% mixed documents. A Latin-only uztext win on fertility can still lose on Cyrillic+Russian code-switch vs mDeBERTa/XLM-R.

**Length.** p99 train is thousands of characters. Unwindowed 512-token BERT silently drops tail entities.

**Empty docs.** 18% have no entities. Over-augmenting dense NER corpora (UzNER train is 100% entity-bearing) will raise false positives on ads/noise.

**Leakage.** External corpora must never enter the official-dev metric, even after mapping.

## How to reproduce

```bash
uv run python scripts/eda_official.py
# faster iteration:
uv run python scripts/eda_official.py --skip-tokenizers --uzner-limit 20000 --hf-max 5000
```

Artifacts (gitignored): `outputs/eda/summary.json`, `outputs/eda/official_overview.png`.

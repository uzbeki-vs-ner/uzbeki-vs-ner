# Silver NER (mapped external)

Gitignored JSONL in the **official hackathon schema**:

```json
{"hash":"sv…","text":"…","entities":[{"label":"ORG"|"NAME"|"GEO","start":0,"end":8}]}
```

Offsets are zero-based with exclusive `end`. Attached Uzbek case endings are included in the span when the rewriter is confident (`Toshkentda`), matching `LABELING_GUIDE.md`.

## Regenerate

From the repo root (CPU only; the script clears `CUDA_VISIBLE_DEVICES`):

```bash
uv run python scripts/extract_silver.py
```

Requires already-downloaded external data (`data/external/hf/…`, `data/external/zenodo/uzner_100k/extracted/`). See `data/external/CATALOG.md`.

## What is in here

| File | Source |
|---|---|
| `risqaliyevds.jsonl` | HF `risqaliyevds/uzbek_ner` after mention→span align |
| `uzner_100k_real.jsonl` | UzNER-100k **real** train subset (`PER/ORG/GPE/LOC`) |
| `uznlp_gold.jsonl` | HF `uznlp-uz/uzbek_NER` token BIO |
| `all.jsonl` | concatenation in that order, official-text and cross-source dups removed |
| `stats.json` | counts (also gitignored) |

`docs/SILVER_EXTRACT.md` has keep/drop counts, the type map, suffix-rewrite precision, and caveats.

**Not for leaderboard metrics.** Train mixes may concatenate silver with official `train.jsonl`; report only on official `dev.jsonl`.

# FAILED: one-word continuation (`add_word`)

Tried 2026-09-02 on Mix B cached official-dev logits
(`outputs/cache/mixb_official_dev`). Analysis fold = 300 docs (seed 42,
5-fold, fold 0). Numbers below are **held-out** exact-span micro-F1 (1200 docs).

## Idea

Word snap cannot cross spaces. After snap, if the next or previous
whitespace-separated word is free and still has mass on `B-L`/`I-L` of the
span's label, grow `[start, end)` through **one** extra word, then snap + NMS
again. Do not glue two already-predicted entities. Do not jump punctuation
or newlines.

This was meant to recover leftovers like `Devid Ebi` → `Devid`.

## Result

Leftover I-mass on the dropped extra word is **not zero** (analysis: 204
adjacent dropped words, mass p50 = 0.36, p75 = 0.79). The decoder just cannot
turn that into exact F1:

| Decode | Held-out F1 | Δ |
|---|---:|---:|
| snap | 0.668 | — |
| snap + continue (analysis pick: all / left / τ=0.3) | 0.667 | −0.0006 |
| snap + conf-gate 0.7 | 0.702 | — |
| snap + conf-gate 0.7 + continue (analysis pick: all / both / τ=0.5) | 0.705 | **+0.0029** |
| oracle NAME-only both τ=0.5 (not a valid pick) | 0.708 | +0.0056 |

Without the confidence gate the rule barely fires (7–10 span changes at τ=0.5).
With the gate it fires more because spurious neighbors are already gone, and
still only **+12 TP / −23 FP** on held-out. That is noise, not a decode policy.

Raw grid: `outputs/eval/mixb_continue.json` (gitignored).

## Do not revive

- Do not import this from `predict_records`.
- Do not merge adjacent same-type entities either (separate trap: 284 help /
  383 destroy already-exact spans on snap preds).
- Multi-word misses after snap are a **training / head** problem, not a
  one-word grow rule.

Fossil files in this folder are the implementation and CPU tests as they
stood when we stopped. They are not on `PYTHONPATH` and `make test` does not
collect them.

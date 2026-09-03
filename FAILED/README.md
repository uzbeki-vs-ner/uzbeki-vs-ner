# Failed experiments

Code and notes for decode / training tricks we **measured and dropped**.
Nothing here is imported by `uzbek_ner` or collected by `make test`.

| Folder | What we tried | Held-out exact F1 | Why it is here |
|---|---|---:|---|
| [`add_word/`](add_word/) | Grow a snapped span by one neighboring word from leftover I-mass | +0.003 vs snap+conf-gate; ~0 without the gate | Too small. Do not put it back on the default path. |

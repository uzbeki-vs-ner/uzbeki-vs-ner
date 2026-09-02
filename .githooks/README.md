# Local Git hooks

This repo uses `core.hooksPath=.githooks` (set by `scripts/git_init_github.sh`).
Enable it in a fresh clone:

```bash
git config core.hooksPath .githooks
chmod +x .githooks/commit-msg .githooks/pre-push
```

`make hooks` still installs [pre-commit](https://pre-commit.com/) (ruff, large-file check). That is separate from these Git hooks.

## `commit-msg`

Rejects one-line / placeholder commits (`wip`, `fix`, `update`, …).

Required shape:

1. Subject line (Conventional Commits is fine, e.g. `ci: add GitHub Actions workflow`)
2. A blank line
3. A **body** with at least a few sentences or bullet points explaining *why*, not only *what*

Merge / revert / rebase / cherry-pick machinery is not blocked.

## `pre-push`

Blocks pushes whose remote URL looks corporate/internal. Personal GitHub only
(`git@github.com:uzbeki-vs-ner/uzbeki-vs-ner.git`).

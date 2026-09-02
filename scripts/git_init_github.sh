#!/bin/bash
# Local git init for ITMO_hack → GitHub (bugkira). Does not push.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GITHUB_REMOTE="${GITHUB_REMOTE:-git@github.com:uzbeki-vs-ner/uzbeki-vs-ner.git}"
GIT_USER_NAME="${GIT_USER_NAME:-Daniil Sereda}"
# Global git uses dsereda@ptsecurity.com — personal email for public GitHub:
GIT_USER_EMAIL="${GIT_USER_EMAIL:-marvolo04@mail.ru}"

if [ -d .git ]; then
  echo "Already a git repo: $ROOT/.git"
else
  git config --local init.defaultBranch main 2>/dev/null || true
  if git init -b main >/dev/null 2>&1; then
    echo "Initialized empty git repo in $ROOT (branch main)"
  else
    git init
    git branch -m main
    echo "Initialized empty git repo in $ROOT (branch renamed to main)"
  fi
fi

git config --local user.name "$GIT_USER_NAME"
git config --local user.email "$GIT_USER_EMAIL"
git config --local core.hooksPath .githooks
git config --local commit.gpgsign false

if git remote get-url origin >/dev/null 2>&1; then
  echo "remote origin already set: $(git remote get-url origin)"
else
  git remote add origin "$GITHUB_REMOTE"
fi

git config --local remote.origin.pushurl "$GITHUB_REMOTE"
chmod +x .githooks/pre-push .githooks/commit-msg 2>/dev/null || true

echo ""
echo "Local git configured:"
echo "  branch     = $(git symbolic-ref --short HEAD 2>/dev/null || echo '(no commits yet)')"
echo "  user.name  = $(git config --local user.name)"
echo "  user.email = $(git config --local user.email)"
echo "  hooksPath  = $(git config --local core.hooksPath)"
echo "  origin     = $(git remote get-url origin 2>/dev/null || echo '(not set)')"
echo "  pushurl    = $(git config --local remote.origin.pushurl 2>/dev/null || echo '(not set)')"
echo ""
echo "Create the repo on GitHub if needed, then:"
echo "  git add … && git commit … && git push -u origin main"

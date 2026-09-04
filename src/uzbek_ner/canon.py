# ruff: noqa: RUF001
"""Product v0 entity canon: glued case suffix + apostrophe fold. Offsets stay put.

This does **not** change inference or the scored API. Brand Analytics aggregates
mentions by ``(label, key)``; the quote in the feed is still ``text[start:end]``.

v0 on purpose does not: latin↔cyrillic, NAME clustering, or stripping separate
admin words such as ``viloyati``.
"""

from __future__ import annotations

from uzbek_ner.decode.snap import APOSTROPHES, SUFFIXES

_APOSTROPHE_PREFIXES: tuple[str, ...] = ("", *APOSTROPHES)
_FOLD_APOSTROPHE = "'"

# Last-token lemmas whose letters coincide with a case suffix (Kanada/da, …).
# Locative of the same word still strips: Kanadada → Kanada.
_LEMMA_LAST_TOKENS = frozenset(
    {
        "amanda",
        "florida",
        "grenada",
        "honda",
        "kanada",
        "nevada",
        "rwanda",
        "uganda",
        "аманда",
        "канада",
        "невада",
        "руанда",
        "уганда",
        "флорида",
        "хонда",
        "гренада",
    }
)


def fold_apostrophes(text: str) -> str:
    out = text
    for mark in APOSTROPHES:
        if mark == _FOLD_APOSTROPHE:
            continue
        out = out.replace(mark, _FOLD_APOSTROPHE)
    return out


def _split_last_token(surface: str) -> tuple[str, str]:
    index = len(surface)
    while index > 0 and not surface[index - 1].isspace():
        index -= 1
    return surface[:index], surface[index:]


def _trailing_suffix(token: str) -> str | None:
    """Longest glued case ending at the end of ``token`` (original casing)."""

    if not token or token.casefold() in _LEMMA_LAST_TOKENS:
        return None
    folded = token.casefold()
    best: str | None = None
    for suffix in SUFFIXES:
        for prefix in _APOSTROPHE_PREFIXES:
            candidate = prefix + suffix
            length = len(candidate)
            if length >= len(token) or not folded.endswith(candidate.casefold()):
                continue
            stem = token[:-length]
            if not stem:
                continue
            last = stem[-1]
            if not (last.isalnum() or last in APOSTROPHES or last in "-‑"):
                continue
            matched = token[-length:]
            if best is None or length > len(best):
                best = matched
    return best


def strip_attached_suffix(surface: str) -> str:
    """Drop a snap.py case ending glued onto the last token. Leave ``KFC da``."""

    prefix, token = _split_last_token(surface)
    matched = _trailing_suffix(token)
    if matched is None:
        return surface
    return prefix + token[: -len(matched)]


def canon_surface(surface: str) -> str:
    """Display lemma: suffix off, apostrophes folded. Stem casing kept."""

    return fold_apostrophes(strip_attached_suffix(surface))


def mention_key(label: str, surface: str) -> tuple[str, str]:
    """Aggregation key. Same string, different label → different keys (Andijon)."""

    return (label, canon_surface(surface).casefold())

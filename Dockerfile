# syntax=docker/dockerfile:1
# Stub-only inference image: FastAPI + uvicorn, no torch / HuggingFace weights.
# Build: docker build -t ner-uz-solution .
# Run:   docker run --rm -p 8000:8000 ner-uz-solution
# A later GPU model image may need: docker run --rm --gpus all -p 8000:8000 ner-uz-solution
#
# ARG BACKEND=stub is reserved for a future torch stage (uv sync of the full lockfile).

FROM python:3.11-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.9.24 /uv /uvx /bin/

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0 \
    PATH="/app/.venv/bin:$PATH"

ARG BACKEND=stub

COPY pyproject.toml uv.lock README.md ./

# --only-group service skips the project ML stack (torch, transformers, …).
RUN uv sync --frozen --only-group service --no-install-project \
    && echo "BACKEND=${BACKEND}" >/app/.backend

COPY src/uzbek_ner ./src/uzbek_ner

RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# One worker: stub is fine; a GPU model must stay at a single process anyway.
CMD ["uvicorn", "uzbek_ner.service.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

"""FastAPI inference app matching the organizer HTTP contract.

The scored API is a **non-streaming JSON batch**: POST /api/v1/predict accepts a
JSON array and returns one JSON object after the whole batch is processed.
Do not add SSE or WebSocket on the scored routes. Optional streaming would
belong on unused /internal paths only (not implemented).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status

from uzbek_ner.service.backend import NerBackend, StubNerBackend
from uzbek_ner.service.schemas import HealthResponse, PredictBatch, PredictResponse, PredictResult


def create_backend() -> NerBackend:
    """Build the process-wide backend. Swap StubNerBackend for Torch later."""

    return StubNerBackend()


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Load the backend before serving. Until then uvicorn refuses connections.

    A slower (Torch) load can set ``ready`` only after weights are in memory
    and return 503 from /healthz in the meantime.
    """

    application.state.backend = create_backend()
    application.state.ready = True
    yield
    application.state.ready = False
    application.state.backend = None


def create_app() -> FastAPI:
    """Application factory (tests use this so lifespan runs per client)."""

    application = FastAPI(
        title="Uzbek NER",
        version="0.1.0",
        lifespan=lifespan,
    )
    application.add_api_route("/healthz", healthz, methods=["GET"], response_model=HealthResponse)
    application.add_api_route(
        "/api/v1/predict",
        predict,
        methods=["POST"],
        response_model=PredictResponse,
    )
    return application


def _require_backend(request: Request) -> NerBackend:
    if not getattr(request.app.state, "ready", False):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="backend not ready",
        )
    backend = getattr(request.app.state, "backend", None)
    if not isinstance(backend, NerBackend):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="backend not ready",
        )
    return backend


async def healthz(request: Request) -> HealthResponse:
    """Cheap liveness. Must not run inference."""

    _require_backend(request)
    return HealthResponse(status="ok")


async def predict(request: Request, items: PredictBatch) -> PredictResponse:
    """Non-streaming batch NER. Empty or duplicate-hash bodies are 4xx."""

    backend = _require_backend(request)
    results = await backend.predict_batch(items)
    if len(results) != len(items):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="backend returned a different batch size",
        )
    aligned: list[PredictResult] = []
    for item, result in zip(items, results, strict=True):
        if result.hash != item.hash:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="backend changed document hash or order",
            )
        aligned.append(result)
    return PredictResponse(data=aligned)


app = create_app()

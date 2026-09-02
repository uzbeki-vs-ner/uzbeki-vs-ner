"""HTTP inference service (organizer contract: /healthz and /api/v1/predict)."""

from uzbek_ner.service.app import app, create_app

__all__ = ["app", "create_app"]

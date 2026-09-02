"""Offline HTTP contract tests for the inference service (no Docker, no models)."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from uzbek_ner.labels import ENTITY_LABELS
from uzbek_ner.service.app import create_app

LATIN_TEXT = "Ali Toshkent shahrida ishlaydi."
CYRILLIC_TEXT = "Алишер Навоий Тошкентда туғилган."


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(create_app()) as test_client:
        yield test_client


def _assert_valid_entities(text: str, entities: list[object]) -> None:
    assert isinstance(entities, list)
    seen: set[tuple[object, object, object]] = set()
    for entity in entities:
        assert isinstance(entity, dict)
        assert entity["label"] in ENTITY_LABELS
        start = entity["start"]
        end = entity["end"]
        assert isinstance(start, int) and isinstance(end, int)
        assert 0 <= start < end <= len(text)
        key = (entity["label"], start, end)
        assert key not in seen
        seen.add(key)


def test_healthz_ok(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert response.json() == {"status": "ok"}


def test_predict_latin(client: TestClient) -> None:
    payload = [{"hash": "contract-latin", "text": LATIN_TEXT}]
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert list(body.keys()) == ["data"]
    assert len(body["data"]) == 1
    result = body["data"][0]
    assert result["hash"] == "contract-latin"
    _assert_valid_entities(LATIN_TEXT, result["entities"])
    labels = {(e["label"], LATIN_TEXT[e["start"] : e["end"]]) for e in result["entities"]}
    assert ("NAME", "Ali") in labels
    assert ("GEO", "Toshkent") in labels


def test_predict_cyrillic(client: TestClient) -> None:
    payload = [{"hash": "contract-cyrillic", "text": CYRILLIC_TEXT}]
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    result = response.json()["data"][0]
    assert result["hash"] == "contract-cyrillic"
    _assert_valid_entities(CYRILLIC_TEXT, result["entities"])
    spans = {(e["label"], CYRILLIC_TEXT[e["start"] : e["end"]]) for e in result["entities"]}
    assert ("NAME", "Алишер Навоий") in spans
    assert ("GEO", "Тошкентда") in spans


def test_predict_batch_preserves_order(client: TestClient) -> None:
    payload = [
        {"hash": "contract-latin", "text": LATIN_TEXT},
        {"hash": "contract-cyrillic", "text": CYRILLIC_TEXT},
    ]
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    results = response.json()["data"]
    assert [item["hash"] for item in results] == ["contract-latin", "contract-cyrillic"]


def test_predict_empty_entities(client: TestClient) -> None:
    payload = [{"hash": "none-001", "text": "salom dunyo"}]
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    result = response.json()["data"][0]
    assert result["hash"] == "none-001"
    assert result["entities"] == []


def test_predict_rejects_empty_array(client: TestClient) -> None:
    response = client.post("/api/v1/predict", json=[])
    assert 400 <= response.status_code < 500


def test_predict_rejects_duplicate_hash(client: TestClient) -> None:
    payload = [
        {"hash": "dup", "text": "Ali"},
        {"hash": "dup", "text": "Toshkent"},
    ]
    response = client.post("/api/v1/predict", json=payload)
    assert 400 <= response.status_code < 500


def test_predict_rejects_missing_text(client: TestClient) -> None:
    response = client.post("/api/v1/predict", json=[{"hash": "only-hash"}])
    assert 400 <= response.status_code < 500


def test_predict_rejects_missing_hash(client: TestClient) -> None:
    response = client.post("/api/v1/predict", json=[{"text": LATIN_TEXT}])
    assert 400 <= response.status_code < 500

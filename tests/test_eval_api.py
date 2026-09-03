"""Offline HTTP tests for the eval comparison API (no Docker, no torch)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from uzbek_ner.evaldash.app import create_app

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_RUNS = REPO_ROOT / "tests" / "fixtures" / "eval" / "runs"


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(create_app(runs_dir=FIXTURE_RUNS)) as test_client:
        yield test_client


def test_healthz(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_list_runs_sorted_by_micro_f1_desc(client: TestClient) -> None:
    response = client.get("/api/v1/runs")
    assert response.status_code == 200
    run_ids = [item["run_id"] for item in response.json()["runs"]]
    assert run_ids == ["uztext_full", "uztext_smoke", "gazetteer_stub"]
    f1s = [item["metrics"]["micro"]["f1"] for item in response.json()["runs"]]
    assert f1s == sorted(f1s, reverse=True)


def test_list_runs_sort_ascending(client: TestClient) -> None:
    response = client.get("/api/v1/runs", params={"sort": "micro.f1", "order": "asc"})
    assert response.status_code == 200
    run_ids = [item["run_id"] for item in response.json()["runs"]]
    assert run_ids == ["gazetteer_stub", "uztext_smoke", "uztext_full"]


def test_list_runs_rejects_unknown_sort(client: TestClient) -> None:
    response = client.get("/api/v1/runs", params={"sort": "nope"})
    assert response.status_code == 400


def test_get_run_detail(client: TestClient) -> None:
    response = client.get("/api/v1/runs/uztext_smoke")
    assert response.status_code == 200
    body = response.json()
    assert body["run_id"] == "uztext_smoke"
    assert body["split"] == "official_dev"
    assert body["metrics"]["by_label"]["GEO"]["f1"] > 0


def test_get_run_missing(client: TestClient) -> None:
    response = client.get("/api/v1/runs/does-not-exist")
    assert response.status_code == 404


def test_prometheus_metrics(client: TestClient) -> None:
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    text = response.text
    assert "ner_eval_f1{" in text
    assert 'label="NAME"' in text
    assert "ner_eval_precision{" in text
    assert "ner_eval_recall{" in text
    assert 'run_id="gazetteer_stub"' in text


def test_html_index_table(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    body = response.text
    assert "uztext_full" in body
    assert "uztext_smoke" in body
    assert "gazetteer_stub" in body
    assert "micro F1" in body
    assert "Рабочий бейзлайн" in body
    assert "Конкурентно" in body
    assert "границы F1" in body
    assert "тип @ спан" in body


def test_scale_endpoint_explains_half_and_point_seven(client: TestClient) -> None:
    response = client.get("/api/v1/scale")
    assert response.status_code == 200
    body = response.json()
    ids = [band["id"] for band in body["bands"]]
    assert ids == ["broken", "weak", "baseline", "competitive", "strong", "excellent"]
    assert "0.5" in body["note"]
    assert "0.7" in body["note"]
    assert body["anchors"]["uztext_2ep_smoke"].startswith("0.578")

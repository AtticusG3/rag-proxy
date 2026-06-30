"""PROXY_INTERNAL_TOKEN opt-in gate on proxy and /metrics."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from conftest import buffered_upstream_response, pooled_client_mock
from rag_proxy import sidecar_client as sc
from rag_proxy.app import app
from rag_proxy.config import settings


@pytest.fixture(autouse=True)
def _reset_sidecar_clients():
    import rag_proxy.sidecar_client as sc

    sc._embed_client = None
    sc._qdrant_client = None
    sc._sparse_client = None
    sc._reranker_client = None
    yield
    sc._embed_client = None
    sc._qdrant_client = None
    sc._sparse_client = None
    sc._reranker_client = None


def _pooled_upstream():
    mock_response = buffered_upstream_response()
    mock_client = pooled_client_mock(mock_response)
    return patch("rag_proxy.upstream_client.httpx.AsyncClient", return_value=mock_client)


def test_proxy_allows_requests_when_token_unset(monkeypatch):
    monkeypatch.setattr(settings, "proxy_internal_token", "")

    with _pooled_upstream():
        with TestClient(app) as client:
            resp = client.get("/v1/models")

    assert resp.status_code == 200


def test_proxy_rejects_missing_token_when_configured(monkeypatch):
    monkeypatch.setattr(settings, "proxy_internal_token", "secret-token")

    with _pooled_upstream():
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/v1/models")

    assert resp.status_code == 401
    assert resp.text == "unauthorized\n"


def test_proxy_accepts_matching_token_header(monkeypatch):
    monkeypatch.setattr(settings, "proxy_internal_token", "secret-token")

    with _pooled_upstream():
        with TestClient(app) as client:
            resp = client.get("/v1/models", headers={"X-Internal-Token": "secret-token"})

    assert resp.status_code == 200


def test_metrics_rejects_missing_token_when_configured(monkeypatch):
    monkeypatch.setattr(settings, "proxy_internal_token", "metrics-secret")
    monkeypatch.setattr(settings, "enable_metrics", True)

    with _pooled_upstream():
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/metrics")

    assert resp.status_code == 401


def test_metrics_accepts_matching_token_header(monkeypatch):
    monkeypatch.setattr(settings, "proxy_internal_token", "metrics-secret")
    monkeypatch.setattr(settings, "enable_metrics", True)

    with _pooled_upstream():
        with TestClient(app) as client:
            resp = client.get("/metrics", headers={"X-Internal-Token": "metrics-secret"})

    assert resp.status_code == 200
    assert "rag_proxy" in resp.text or resp.text.strip() != ""

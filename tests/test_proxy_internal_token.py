"""PROXY_INTERNAL_TOKEN opt-in gate on proxy and /metrics, and upstream header hygiene."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

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


def _recording_pooled_upstream() -> tuple[Any, list[dict[str, str]]]:
    """Fake upstream that records the headers the proxy hands to the outbound request."""
    mock_response = buffered_upstream_response()
    mock_client = pooled_client_mock(mock_response)
    received: list[dict[str, str]] = []

    def _build_request(**kwargs):
        received.append(dict(kwargs.get("headers") or {}))
        return "req"

    mock_client.build_request = MagicMock(side_effect=_build_request)
    patcher = patch(
        "rag_proxy.upstream_client.httpx.AsyncClient", return_value=mock_client
    )
    return patcher, received


def _sent_headers(received: list[dict[str, str]]) -> dict[str, str]:
    assert len(received) == 1, "proxy must build exactly one upstream request"
    return {k.lower(): v for k, v in received[0].items()}


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


def test_proxy_strips_internal_token_before_forwarding(monkeypatch):
    """The shared secret is proxy-only; the upstream must never see X-Internal-Token."""
    monkeypatch.setattr(settings, "proxy_internal_token", "secret-token")

    patcher, received = _recording_pooled_upstream()
    with patcher:
        with TestClient(app) as client:
            resp = client.get(
                "/v1/models", headers={"X-Internal-Token": "secret-token"}
            )

    assert resp.status_code == 200
    assert "x-internal-token" not in _sent_headers(received)


def test_proxy_strips_proxy_only_control_headers_before_forwarding(monkeypatch):
    """Per-request RAG controls are consumed by the proxy, not the upstream."""
    monkeypatch.setattr(settings, "proxy_internal_token", "")

    patcher, received = _recording_pooled_upstream()
    with patcher:
        with TestClient(app) as client:
            resp = client.get(
                "/v1/models",
                headers={
                    "X-RAG-Mode": "force",
                    "X-No-Cache": "true",
                    "X-Conversation-Id": "conv-42",
                    "X-Capture-Log": "true",
                },
            )

    assert resp.status_code == 200
    sent = _sent_headers(received)
    for name in ("x-rag-mode", "x-no-cache", "x-conversation-id", "x-capture-log"):
        assert name not in sent


def test_proxy_forwards_authorization_unchanged(monkeypatch):
    """Client auth for the upstream must survive proxying untouched."""
    monkeypatch.setattr(settings, "proxy_internal_token", "secret-token")

    patcher, received = _recording_pooled_upstream()
    with patcher:
        with TestClient(app) as client:
            resp = client.get(
                "/v1/models",
                headers={
                    "X-Internal-Token": "secret-token",
                    "Authorization": "Bearer upstream-key-77",
                },
            )

    assert resp.status_code == 200
    assert _sent_headers(received)["authorization"] == "Bearer upstream-key-77"


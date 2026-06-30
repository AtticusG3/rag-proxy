"""HTTP-layer chat proxy: fail-open and RAG injection through upstream forward."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from rag_proxy.app import app
from rag_proxy.config import settings
from rag_proxy.context import ChunkHit

from conftest import FakeAsyncClient, buffered_upstream_response, capture_upstream_body, pooled_client_mock


def test_chat_post_fail_open_forwards_original_body_on_augment_error():
    """RAG augmentation errors must not alter the body forwarded to llama-swap."""

    original = {"model": "chat-demo", "messages": [{"role": "user", "content": "hi"}]}
    mock_response = buffered_upstream_response(b'{"choices":[]}')
    mock_client = pooled_client_mock(mock_response)
    captured = capture_upstream_body(mock_client)

    with patch("rag_proxy.upstream_client.httpx.AsyncClient", return_value=mock_client):
        with patch(
            "rag_proxy.app.augment_chat_payload_with_context",
            AsyncMock(side_effect=RuntimeError("rag boom")),
        ):
            with TestClient(app) as client:
                resp = client.post("/v1/chat/completions", json=original)

    assert resp.status_code == 200
    assert len(captured) == 1
    assert json.loads(captured[0]) == original


def test_chat_post_injects_retrieved_context_into_upstream_body(monkeypatch):
    """Legacy POST chat path must embed retrieved chunks into the upstream JSON body."""

    monkeypatch.setattr(settings, "enable_cognitive_pipeline", False)
    chunk_text = "homelab docker compose stack deployment guide"

    async def fake_hybrid(_query, limit, score_threshold=None, no_cache=False, cache_hits=None):
        return [ChunkHit(id="doc-1", text=chunk_text, score=0.91, source="dense")]

    mock_response = buffered_upstream_response(b'{"choices":[]}')
    mock_client = pooled_client_mock(mock_response)
    captured = capture_upstream_body(mock_client)

    with patch("rag_proxy.pipeline_stages.hybrid_search", fake_hybrid):
        with patch("rag_proxy.upstream_client.httpx.AsyncClient", return_value=mock_client):
            with TestClient(app) as client:
                resp = client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "chat-demo",
                        "messages": [{"role": "user", "content": "how do I deploy?"}],
                    },
                )

    assert resp.status_code == 200
    upstream = json.loads(captured[0])
    assert upstream["messages"][0]["role"] == "system"
    assert chunk_text in upstream["messages"][0]["content"]
    assert upstream["messages"][1]["content"] == "how do I deploy?"

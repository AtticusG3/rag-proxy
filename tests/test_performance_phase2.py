"""Tests for token_estimate and /debug endpoint."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from conftest import buffered_upstream_response, pooled_client_mock, pooled_ctor_side_effect
from rag_proxy.app import app
from rag_proxy.clients.retrieval_core import prepare_embed_text
from rag_proxy.clients.retrieve_sync import RetrieveConfig, embed_query
from rag_proxy.config import settings
from rag_proxy.stages.tier2_context import apply_context_budget, resolve_inject_budget_chars
from rag_proxy.context import ChunkHit, RequestContext
from rag_proxy.registry.models import ModelCapabilities, ModelRegistry
from rag_proxy.token_estimate import count_tokens, truncate_to_tokens


def test_prepare_embed_text_uses_tail():
    text = "prefix-" + "x" * 100
    trimmed = prepare_embed_text(text, 20)
    assert trimmed == "x" * 20
    assert not trimmed.startswith("prefix")


def test_count_tokens_char_fallback_when_estimate_off(monkeypatch):
    monkeypatch.setattr(settings, "enable_tokenizer_estimate", False)
    assert count_tokens("abcd") == 1


def test_count_tokens_uses_tiktoken_when_enabled(monkeypatch):
    monkeypatch.setattr(settings, "enable_tokenizer_estimate", True)
    tokens = count_tokens("hello world")
    assert tokens >= 2


def test_truncate_to_tokens_limits_output(monkeypatch):
    monkeypatch.setattr(settings, "enable_tokenizer_estimate", True)
    text = "one two three four five six seven eight nine ten"
    short = truncate_to_tokens(text, 3)
    assert count_tokens(short) <= 3


def test_apply_context_budget_tokens_keeps_high_score_first(monkeypatch):
    monkeypatch.setattr(settings, "enable_tokenizer_estimate", True)
    hits = [
        ChunkHit(id="a", text="alpha beta gamma", score=0.9, source="dense"),
        ChunkHit(id="b", text="delta epsilon zeta eta", score=0.5, source="dense"),
    ]
    kept = apply_context_budget(hits, 5)
    assert len(kept) == 1
    assert kept[0].id == "a"


def test_resolve_inject_budget_tokens_mode(monkeypatch):
    monkeypatch.setattr(settings, "enable_tokenizer_estimate", True)
    monkeypatch.setattr(settings, "context_budget_ratio", 0.25)
    monkeypatch.setattr(settings, "default_completion_reserve", 1024)
    ctx = RequestContext(messages=[], requested_model="test-model")
    registry = ModelRegistry()
    registry._cache["test-model"] = ModelCapabilities(
        model_id="test-model",
        context_length=8192,
    )
    budget = resolve_inject_budget_chars(ctx, registry)
    assert budget == int(8192 * 0.25) - 1024


def test_debug_requires_token_when_configured(monkeypatch):
    monkeypatch.setattr(settings, "proxy_internal_token", "debug-secret")

    mock_response = buffered_upstream_response()
    mock_client = pooled_client_mock(mock_response)
    with patch(
        "rag_proxy.upstream_client.httpx.AsyncClient",
        side_effect=pooled_ctor_side_effect(mock_client),
    ):
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/debug")

    assert resp.status_code == 401


def test_debug_returns_snapshot(monkeypatch):
    monkeypatch.setattr(settings, "proxy_internal_token", "")
    monkeypatch.setattr(settings, "enable_embed_cache", True)

    mock_response = buffered_upstream_response()
    mock_client = pooled_client_mock(mock_response)
    with patch(
        "rag_proxy.upstream_client.httpx.AsyncClient",
        side_effect=pooled_ctor_side_effect(mock_client),
    ):
        with TestClient(app) as client:
            resp = client.get("/debug")

    assert resp.status_code == 200
    data = resp.json()
    assert "upstream" in data
    assert "embed_cache" in data
    assert data["enable_embed_cache"] is True


def test_embed_query_uses_tail_truncation(monkeypatch):
    captured: list[str] = []

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [{"embedding": [0.1]}]}

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def post(self, url, json=None, **_kwargs):
            captured.append(json["input"])
            return FakeResponse()

    config = RetrieveConfig(
        embed_url="http://embed.test",
        qdrant_url="http://qdrant.test",
        qdrant_collection="c",
        sparse_index_url="",
        reranker_url="",
        similarity_threshold=0.5,
        hybrid_dense_weight=0.7,
        embed_max_chars=10,
    )
    query = "keep-this-tail-endpiece"
    with patch("rag_proxy.clients.retrieve_sync.httpx.Client", return_value=FakeClient()):
        vector = embed_query(config, query)

    assert vector == [0.1]
    assert captured[0] == query[-10:]

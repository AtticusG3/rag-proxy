"""Tests for llama-swap JSON helpers."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from rag_proxy.clients.llama_swap import fetch_models, parse_json_object


def test_parse_json_object_extracts_brace_wrapped_payload():
    raw = 'Here is JSON: {"query":"kubernetes pods"} trailing text'
    data = parse_json_object(raw)
    assert data == {"query": "kubernetes pods"}


def test_parse_json_object_returns_none_for_invalid_payload():
    assert parse_json_object("not json") is None


def test_fetch_models_reuses_upstream_pool_with_short_timeout(monkeypatch):
    """Intent model listing must borrow the proxy pool instead of a client per call."""
    client = MagicMock()
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = {"data": [{"id": "loaded"}]}
    client.get = AsyncMock(return_value=response)

    async def borrow():
        return client

    monkeypatch.setattr(
        "rag_proxy.clients.llama_swap.ensure_upstream_client",
        borrow,
    )
    with patch("rag_proxy.clients.llama_swap.httpx.AsyncClient") as ctor:
        models = asyncio.run(fetch_models("http://llama.test"))

    ctor.assert_not_called()
    client.get.assert_awaited_once_with("http://llama.test/v1/models", timeout=10)
    assert models == [{"id": "loaded"}]


def test_fetch_models_opens_one_shot_client_when_pool_is_down(monkeypatch):
    """Scripts and tests have no lifespan pool, so the helper still opens its own client."""

    async def missing():
        raise RuntimeError("upstream client not started")

    monkeypatch.setattr(
        "rag_proxy.clients.llama_swap.ensure_upstream_client",
        missing,
    )
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = {"data": []}
    client = MagicMock()
    client.get = AsyncMock(return_value=response)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)

    with patch(
        "rag_proxy.clients.llama_swap.httpx.AsyncClient",
        return_value=client,
    ) as ctor:
        models = asyncio.run(fetch_models("http://llama.test"))

    ctor.assert_called_once()
    assert models == []

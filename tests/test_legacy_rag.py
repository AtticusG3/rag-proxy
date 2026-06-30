"""legacy_rag embed/Qdrant HTTP helpers (mocked; no live services)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from conftest import FakeAsyncClient
from rag_proxy.chunk_text import extract_chunk_text
from rag_proxy.clients.retrieval_async import embed_text, get_embedding, search_qdrant_dense
from rag_proxy.config import settings


def test_get_embedding_returns_none_on_http_500(monkeypatch):
    """Embed server failures must fail-open to None (no vector for retrieval)."""

    monkeypatch.setattr(settings, "embed_retries", 1)

    response = MagicMock()
    response.status_code = 500
    response.text = "server error"
    response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "500",
        request=MagicMock(),
        response=response,
    )
    post = AsyncMock(return_value=response)

    with patch(
        "rag_proxy.clients.retrieval_async.httpx.AsyncClient",
        return_value=FakeAsyncClient(post),
    ):
        vector = asyncio.run(get_embedding("hello"))

    assert vector is None
    assert post.await_count == 1


def test_get_embedding_returns_vector_on_success():
    response = MagicMock()
    response.status_code = 200
    response.text = ""
    response.json.return_value = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}

    post = AsyncMock(return_value=response)

    with patch(
        "rag_proxy.clients.retrieval_async.httpx.AsyncClient",
        return_value=FakeAsyncClient(post),
    ):
        vector = asyncio.run(get_embedding("hello"))

    assert vector == [0.1, 0.2, 0.3]
    post.assert_awaited_once()
    assert post.await_args.kwargs["json"]["model"] == "nomic-embed-text-v1.5"


def test_search_qdrant_passes_similarity_threshold(monkeypatch):
    """Qdrant search must honor SIMILARITY_THRESHOLD so low-score chunks are excluded server-side."""

    monkeypatch.setattr(settings, "similarity_threshold", 0.72)
    monkeypatch.setattr(settings, "qdrant_url", "http://qdrant.test")
    monkeypatch.setattr(settings, "qdrant_collection", "test_collection")
    captured: list[dict] = []

    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "result": [{"id": "1", "score": 0.9, "payload": {"text": "hit"}}],
    }
    response.raise_for_status = MagicMock()

    async def post(url, json=None, **_kwargs):
        captured.append({"url": url, "json": json})
        return response

    with patch(
        "rag_proxy.clients.retrieval_async.httpx.AsyncClient",
        return_value=FakeAsyncClient(AsyncMock(side_effect=post)),
    ):
        hits = asyncio.run(search_qdrant_dense([0.1, 0.2]))

    assert len(hits) == 1
    assert captured[0]["json"]["score_threshold"] == pytest.approx(0.72)
    assert captured[0]["url"].endswith("/collections/test_collection/points/search")


def test_search_qdrant_returns_empty_on_http_error():
    post = AsyncMock(side_effect=httpx.ConnectError("qdrant down"))

    with patch(
        "rag_proxy.clients.retrieval_async.httpx.AsyncClient",
        return_value=FakeAsyncClient(post),
    ):
        hits = asyncio.run(search_qdrant_dense([0.1]))

    assert hits == []


def test_embed_text_uses_cache_when_enabled(monkeypatch):
    monkeypatch.setattr(settings, "enable_embed_cache", True)
    calls = 0

    async def fake_get_embedding(text: str):
        nonlocal calls
        calls += 1
        return [0.1, 0.2]

    monkeypatch.setattr(
        "rag_proxy.clients.retrieval_async.get_embedding",
        fake_get_embedding,
    )
    monkeypatch.setattr("rag_proxy.clients.retrieval_async._embed_cache", {})

    async def run():
        first = await embed_text("cached query")
        second = await embed_text("cached query")
        return first, second

    first, second = asyncio.run(run())
    assert first == [0.1, 0.2]
    assert second == [0.1, 0.2]
    assert calls == 1


def test_extract_chunk_text_prefers_text_over_content():
    hit = {"payload": {"content": "from content", "text": "from text"}}
    assert extract_chunk_text(hit) == "from text"


def test_extract_chunk_text_falls_back_to_content():
    hit = {"payload": {"content": "only content field"}}
    assert extract_chunk_text(hit) == "only content field"


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"chunk": "from chunk"}, "from chunk"),
        ({"document": "from document"}, "from document"),
        ({"page_content": "from page_content"}, "from page_content"),
        ({"text": "", "content": "fallback content"}, "fallback content"),
    ],
)
def test_extract_chunk_text_payload_key_precedence(payload, expected):
    """Guard canonical Qdrant payload field order (PAYLOAD_TEXT_KEYS)."""
    assert extract_chunk_text({"payload": payload}) == expected


def test_extract_chunk_text_null_payload():
    assert extract_chunk_text({"payload": None}) == ""

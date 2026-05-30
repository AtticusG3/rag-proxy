"""Rerank sidecar fail-open behavior."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from conftest import FakeAsyncClient
from rag_proxy.config import settings
from rag_proxy.context import ChunkHit, RequestContext
from rag_proxy.stages.tier2_rerank import run_rerank


def test_run_rerank_fail_open_preserves_hits_on_sidecar_error(monkeypatch):
    """Rerank sidecar errors must leave hit order intact and record fallback in trace."""

    monkeypatch.setattr(settings, "enable_reranker", True)
    monkeypatch.setattr(settings, "reranker_url", "http://rerank.test")

    original_hits = [
        ChunkHit(id="a", text="first chunk", score=0.9),
        ChunkHit(id="b", text="second chunk", score=0.8),
    ]
    ctx = RequestContext(
        query_text="deploy homelab stack",
        retrieval_query="deploy homelab stack",
        hits=list(original_hits),
    )

    error_response = MagicMock()
    error_response.status_code = 502
    post = AsyncMock(
        side_effect=httpx.HTTPStatusError(
            "502",
            request=MagicMock(),
            response=error_response,
        )
    )

    with patch(
        "rag_proxy.stages.tier2_rerank.httpx.AsyncClient",
        return_value=FakeAsyncClient(post),
    ):
        asyncio.run(run_rerank(ctx))

    assert [h.id for h in ctx.hits] == ["a", "b"]
    assert "rerank:fallback" in ctx.stage_trace
    assert not any("rerank:ok" in s for s in ctx.stage_trace)

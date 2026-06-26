"""MemGraphRAG pipeline stage fail-open and hit merging."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from rag_proxy.config import settings
from rag_proxy.context import ChunkHit, RequestContext
from rag_proxy.stages.tier3_memgraphrag import run_memgraphrag


def test_run_memgraphrag_appends_hits_to_context(monkeypatch) -> None:
    """Stage must append memgraphrag hits without replacing prior retrieval hits."""
    monkeypatch.setattr(settings, "memgraphrag_db_path", "/tmp/memgraphrag.sqlite")

    memory = MagicMock()
    memory.facts = {0: object()}
    expected = [
        ChunkHit(id="mg-1", text="graph passage", score=0.7, source="memgraphrag"),
    ]

    async def fake_retrieve(_query: str):
        return list(expected)

    with patch("rag_proxy.stages.tier3_memgraphrag.load_memory", return_value=memory):
        with patch(
            "rag_proxy.stages.tier3_memgraphrag.MemGraphRetriever"
        ) as retriever_cls:
            retriever_cls.return_value.retrieve = AsyncMock(side_effect=fake_retrieve)
            ctx = RequestContext(
                query_text="who knows Bob?",
                hits=[ChunkHit(id="dense-1", text="dense hit", score=0.9)],
            )
            asyncio.run(run_memgraphrag(ctx))

    assert [h.id for h in ctx.hits] == ["dense-1", "mg-1"]
    assert ctx.hits[-1].source == "memgraphrag"
    assert "memgraphrag:1" in ctx.stage_trace


def test_run_memgraphrag_fail_open_on_retriever_error(monkeypatch) -> None:
    """Retriever exceptions must not break the request; errors are recorded."""
    monkeypatch.setattr(settings, "memgraphrag_db_path", "/tmp/memgraphrag.sqlite")

    memory = MagicMock()
    memory.facts = {0: object()}

    with patch("rag_proxy.stages.tier3_memgraphrag.load_memory", return_value=memory):
        with patch(
            "rag_proxy.stages.tier3_memgraphrag.MemGraphRetriever"
        ) as retriever_cls:
            retriever_cls.return_value.retrieve = AsyncMock(
                side_effect=RuntimeError("sqlite locked")
            )
            ctx = RequestContext(
                query_text="query",
                hits=[ChunkHit(id="keep", text="stay", score=0.5)],
            )
            asyncio.run(run_memgraphrag(ctx))

    assert [h.id for h in ctx.hits] == ["keep"]
    assert any("memgraphrag:sqlite locked" in err for err in ctx.errors)


def test_run_memgraphrag_skips_without_query(monkeypatch) -> None:
    """No query text means the stage is a no-op."""
    monkeypatch.setattr(settings, "memgraphrag_db_path", "/tmp/memgraphrag.sqlite")
    ctx = RequestContext(query_text=None, hits=[])

    with patch("rag_proxy.stages.tier3_memgraphrag.load_memory") as load_memory:
        asyncio.run(run_memgraphrag(ctx))

    load_memory.assert_not_called()
    assert ctx.hits == []


def test_run_memgraphrag_skips_without_db_path(monkeypatch) -> None:
    """Unset MEMGRAPHRAG_DB_PATH skips loading memory."""
    monkeypatch.setattr(settings, "memgraphrag_db_path", "")
    ctx = RequestContext(query_text="query", hits=[])

    with patch("rag_proxy.stages.tier3_memgraphrag.load_memory") as load_memory:
        asyncio.run(run_memgraphrag(ctx))

    load_memory.assert_not_called()


def test_run_memgraphrag_skips_empty_memory(monkeypatch) -> None:
    """Empty fact layer skips retrieval without error."""
    monkeypatch.setattr(settings, "memgraphrag_db_path", "/tmp/memgraphrag.sqlite")
    memory = MagicMock()
    memory.facts = {}
    ctx = RequestContext(query_text="query", hits=[])

    with patch("rag_proxy.stages.tier3_memgraphrag.load_memory", return_value=memory):
        with patch("rag_proxy.stages.tier3_memgraphrag.MemGraphRetriever") as retriever_cls:
            asyncio.run(run_memgraphrag(ctx))

    retriever_cls.assert_not_called()
    assert ctx.hits == []

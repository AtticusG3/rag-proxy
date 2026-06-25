"""Tests for MCP RAG hybrid retrieval."""

from __future__ import annotations

from unittest.mock import patch

import sys
from pathlib import Path

MCP_RAG = Path(__file__).resolve().parents[1] / "sidecars" / "mcp_rag"
sys.path.insert(0, str(MCP_RAG))

from retrieve import (
    RetrievedChunk,
    RetrieveSettings,
    format_chunks_for_agent,
    hybrid_retrieve,
    rrf_merge,
    search_knowledge_base,
)


def _settings(**overrides: object) -> RetrieveSettings:
    base = RetrieveSettings(
        embed_url="http://embed.test",
        qdrant_url="http://qdrant.test",
        qdrant_collection="test_collection",
        sparse_index_url="http://sparse.test",
        reranker_url="http://rerank.test",
        hybrid_dense_weight=0.7,
        similarity_threshold=0.5,
        enable_hybrid=True,
        enable_rerank=True,
        embed_max_chars=2000,
        user_agent="test",
    )
    return RetrieveSettings(**{**base.__dict__, **overrides})


def test_format_chunks_empty() -> None:
    assert "No matching" in format_chunks_for_agent([])


def test_rrf_merge_prefers_both_lists() -> None:
    fused = rrf_merge(
        [
            [("a", 0.9), ("b", 0.5)],
            [("b", 1.0), ("c", 0.4)],
        ],
        limit=3,
        list_weights=[0.7, 0.3],
    )
    assert fused[0] == "b"


def test_search_knowledge_base_pipeline() -> None:
    settings = _settings()
    chunk = RetrievedChunk(
        chunk_id="1",
        text="Python asyncio guide",
        score=0.9,
        source="/zim/python.zim",
        title="asyncio",
        retrieval="dense",
    )
    with patch("retrieve.hybrid_retrieve", return_value=[chunk]) as hybrid:
        with patch("retrieve.rerank_chunks", return_value=[chunk]) as rerank:
            result = search_knowledge_base("python asyncio", top_k=1, settings=settings)
    hybrid.assert_called_once()
    rerank.assert_called_once()
    assert len(result) == 1
    assert "asyncio" in format_chunks_for_agent(result)


def test_hybrid_dense_only_when_disabled() -> None:
    settings = _settings(enable_hybrid=False)
    chunk = RetrievedChunk(
        chunk_id="d1",
        text="only dense",
        score=0.8,
        source="s",
        title="t",
        retrieval="dense",
    )
    with patch("retrieve.embed_query", return_value=[0.1]):
        with patch("retrieve.dense_search", return_value=[{"id": "d1", "score": 0.8, "payload": {"text": "only dense", "title": "t"}}]):
            chunks = hybrid_retrieve(settings, "query", top_k=3)
    assert len(chunks) == 1
    assert chunks[0].retrieval == "dense"

"""Tests for MCP RAG hybrid retrieval."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parents[1]
MCP_RAG = REPO_ROOT / "sidecars" / "mcp_rag"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(MCP_RAG))

from rag_proxy.clients.retrieve_sync import rrf_merge

from retrieve import (
    RetrievedChunk,
    RetrieveSettings,
    format_chunks_for_agent,
    hybrid_retrieve,
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
    """Parity with canonical retrieve_sync RRF fusion."""
    fused = rrf_merge(
        [
            [("a", 0.9), ("b", 0.5)],
            [("b", 1.0), ("c", 0.4)],
        ],
        limit=3,
        list_weights=[0.7, 0.3],
    )
    assert fused[0][0] == "b"


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
    with patch(
        "rag_proxy.clients.retrieve_sync.embed_query", return_value=[0.1]
    ) as embed:
        with patch(
            "rag_proxy.clients.retrieve_sync.dense_search",
            return_value=[
                {
                    "id": "d1",
                    "score": 0.8,
                    "payload": {"text": "only dense", "title": "t"},
                }
            ],
        ) as dense:
            chunks = hybrid_retrieve(settings, "query", top_k=3)
    embed.assert_called_once()
    dense.assert_called_once()
    assert len(chunks) == 1
    assert chunks[0].retrieval == "dense"


def test_hybrid_dense_only_embed_dense_httpx_sequence() -> None:
    """Exercises embed then dense HTTP calls through retrieve_sync (not full mock)."""
    settings = _settings(enable_hybrid=False)

    embed_response = MagicMock()
    embed_response.raise_for_status = MagicMock()
    embed_response.json.return_value = {"data": [{"embedding": [0.1, 0.2]}]}

    qdrant_response = MagicMock()
    qdrant_response.raise_for_status = MagicMock()
    qdrant_response.json.return_value = {
        "result": [
            {
                "id": "d1",
                "score": 0.8,
                "payload": {"text": "only dense", "title": "t"},
            }
        ]
    }

    def fake_post(url: str, json: dict | None = None) -> MagicMock:
        if url.endswith("/v1/embeddings"):
            assert json is not None
            assert json["input"] == "query"
            return embed_response
        if "/points/search" in url:
            return qdrant_response
        raise AssertionError(f"unexpected POST {url}")

    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.post.side_effect = fake_post

    with patch(
        "rag_proxy.clients.retrieve_sync.httpx.Client", return_value=mock_client
    ):
        chunks = hybrid_retrieve(settings, "query", top_k=3)

    assert len(chunks) == 1
    assert chunks[0].chunk_id == "d1"
    assert chunks[0].text == "only dense"
    assert chunks[0].retrieval == "dense"
    assert mock_client.post.call_count == 2


def test_hybrid_mode_single_embed_dense_httpx() -> None:
    """Hybrid mode must not duplicate embed+dense HTTP before tagging."""
    settings = _settings(enable_hybrid=True)

    embed_response = MagicMock()
    embed_response.raise_for_status = MagicMock()
    embed_response.json.return_value = {"data": [{"embedding": [0.1, 0.2]}]}

    qdrant_response = MagicMock()
    qdrant_response.raise_for_status = MagicMock()
    qdrant_response.json.return_value = {
        "result": [
            {
                "id": "d1",
                "score": 0.9,
                "payload": {"text": "dense hit", "title": "Dense"},
            }
        ]
    }

    sparse_response = MagicMock()
    sparse_response.raise_for_status = MagicMock()
    sparse_response.json.return_value = {
        "results": [
            {
                "id": "s1",
                "score": 0.8,
                "payload": {"text": "sparse only", "title": "Sparse"},
            }
        ]
    }

    embed_calls = 0
    dense_calls = 0

    def fake_post(url: str, json: dict | None = None) -> MagicMock:
        nonlocal embed_calls, dense_calls
        if url.endswith("/v1/embeddings"):
            embed_calls += 1
            return embed_response
        if "/points/search" in url:
            dense_calls += 1
            return qdrant_response
        if url.endswith("/search"):
            return sparse_response
        raise AssertionError(f"unexpected POST {url}")

    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.post.side_effect = fake_post

    with patch(
        "rag_proxy.clients.retrieve_sync.httpx.Client", return_value=mock_client
    ):
        chunks = hybrid_retrieve(settings, "query", top_k=3)

    assert embed_calls == 1
    assert dense_calls == 1
    by_id = {c.chunk_id: c for c in chunks}
    assert by_id["d1"].retrieval == "dense"
    assert by_id["s1"].retrieval == "sparse"
    assert by_id["d1"].to_dict() == {
        "id": "d1",
        "text": "dense hit",
        "score": 0.9,
        "source": "",
        "title": "Dense",
        "retrieval": "dense",
    }

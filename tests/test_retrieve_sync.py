"""Unit tests for sync hybrid retrieval foundation."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from rag_proxy.clients.retrieve_sync import (
    RetrieveConfig,
    hybrid_retrieve,
    hybrid_retrieve_with_dense_ids,
    rerank_pairs,
    rrf_merge,
)


def _config(**overrides: object) -> RetrieveConfig:
    base = RetrieveConfig(
        embed_url="http://embed.test",
        qdrant_url="http://qdrant.test",
        qdrant_collection="test_collection",
        sparse_index_url="http://sparse.test",
        reranker_url="http://rerank.test",
        similarity_threshold=0.65,
        hybrid_dense_weight=0.7,
        embed_max_chars=2000,
        enable_hybrid=True,
        enable_rerank=True,
        rerank_top_k=3,
    )
    return RetrieveConfig(**{**base.__dict__, **overrides})


def test_rrf_merge_orders_shared_docs_higher() -> None:
    """RRF ranks docs appearing in both lists above single-list docs."""
    dense = [("a", 0.9), ("b", 0.8)]
    sparse = [("b", 0.7), ("c", 0.6)]
    merged = rrf_merge([dense, sparse], limit=3)
    ids = [doc_id for doc_id, _ in merged]
    assert ids[0] == "b"
    assert "b" in ids


def test_rrf_list_weights_scale_contribution() -> None:
    """RRF list_weights bias fusion toward the heavier list."""
    lists = [[("a", 1.0)], [("b", 1.0)]]
    assert rrf_merge(lists, limit=2, list_weights=[0.9, 0.1])[0][0] == "a"
    assert rrf_merge(lists, limit=2, list_weights=[0.1, 0.9])[0][0] == "b"


def test_rrf_list_weights_length_must_match_ranked_lists() -> None:
    """RRF rejects list_weights length mismatches."""
    with pytest.raises(ValueError, match="list_weights length"):
        rrf_merge([[("a", 1.0)]], list_weights=[0.5, 0.5])


def test_rrf_merge_matches_qdrant_canonical_scores() -> None:
    """RRF scores match the canonical formula (weighted 1/(k+rank+1))."""
    dense = [("x", 0.9), ("y", 0.5)]
    sparse = [("y", 0.8), ("z", 0.4)]
    merged = rrf_merge([dense, sparse], k=60, limit=3, list_weights=[0.7, 0.3])
    scores = dict(merged)
    expected_y = 0.7 * (1.0 / 62) + 0.3 * (1.0 / 61)
    expected_x = 0.7 * (1.0 / 61)
    expected_z = 0.3 * (1.0 / 62)
    assert scores["y"] == pytest.approx(expected_y)
    assert scores["x"] == pytest.approx(expected_x)
    assert scores["z"] == pytest.approx(expected_z)
    assert list(scores.keys())[0] == "y"


def test_hybrid_retrieve_dense_only_when_hybrid_disabled() -> None:
    """Hybrid retrieve falls back to dense hits when hybrid is disabled."""
    cfg = _config(enable_hybrid=False)
    dense_hit = {"id": "d1", "score": 0.9, "payload": {"text": "dense only"}}

    with patch(
        "rag_proxy.clients.retrieve_sync.embed_query", return_value=[0.1, 0.2]
    ) as embed:
        with patch(
            "rag_proxy.clients.retrieve_sync.dense_search", return_value=[dense_hit]
        ) as dense:
            hits = hybrid_retrieve(cfg, "query", limit=3)

    embed.assert_called_once()
    dense.assert_called_once()
    assert len(hits) == 1
    assert hits[0]["id"] == "d1"


def test_hybrid_retrieve_with_dense_ids_tags_dense_sources() -> None:
    """Hybrid fusion returns dense id set for downstream retrieval tagging."""
    cfg = _config(enable_hybrid=True)
    dense_hit = {"id": "d1", "score": 0.9, "payload": {"text": "dense"}}
    sparse_hit = {"id": "s1", "score": 0.8, "payload": {"text": "sparse"}}

    with patch(
        "rag_proxy.clients.retrieve_sync.embed_query", return_value=[0.1, 0.2]
    ):
        with patch(
            "rag_proxy.clients.retrieve_sync.dense_search", return_value=[dense_hit]
        ):
            with patch(
                "rag_proxy.clients.retrieve_sync.sparse_search",
                return_value=[sparse_hit],
            ):
                hits, dense_ids = hybrid_retrieve_with_dense_ids(cfg, "query", limit=3)

    assert dense_ids == {"d1"}
    assert {str(h["id"]) for h in hits} == {"d1", "s1"}


def test_rerank_pairs_returns_sidecar_order() -> None:
    """Rerank pairs returns indices from the sidecar response."""
    cfg = _config()
    pairs = [{"query": "q", "document": "a"}, {"query": "q", "document": "b"}]

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"indices": [1, 0]}

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *args) -> None:
            return None

        def post(self, url: str, json: dict) -> FakeResponse:
            assert url.endswith("/rerank")
            assert json["top_k"] == 3
            return FakeResponse()

    with patch("rag_proxy.clients.retrieve_sync.httpx.Client", FakeClient):
        indices = rerank_pairs(cfg, pairs)

    assert indices == [1, 0]

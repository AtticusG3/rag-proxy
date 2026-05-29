"""Unit tests for rewrite, RRF merge, and dedupe."""

import asyncio

import pytest

from rag_proxy.clients.qdrant import _apply_recency_boost, hybrid_search, rrf_merge
from rag_proxy.config import settings
from rag_proxy.context import ChunkHit, RequestContext
from rag_proxy.stages.tier2_context import apply_context_budget, dedupe_chunks
from rag_proxy.stages.tier2_rewrite import (
    _parse_rewrite_json,
    rewrite_query_deterministic,
    run_rewrite,
)


def test_rewrite_preserves_ip_and_path():
    q = "fix v1.2.3 on 192.168.1.36 /etc/nomad/config"
    out = rewrite_query_deterministic(q)
    assert "192.168.1.36" in out
    assert "/etc/nomad/config" in out


def test_rewrite_expands_k8s_glossary():
    q = "k8s pod scheduling"
    out = rewrite_query_deterministic(q)
    assert "kubernetes" in out


def test_rrf_merge_orders_shared_docs_higher():
    dense = [("a", 0.9), ("b", 0.8)]
    sparse = [("b", 0.7), ("c", 0.6)]
    merged = rrf_merge([dense, sparse], limit=3)
    ids = [doc_id for doc_id, _ in merged]
    assert "b" in ids
    assert ids[0] == "b"


def test_rrf_list_weights_scale_contribution():
    lists = [[("a", 1.0)], [("b", 1.0)]]
    assert rrf_merge(lists, limit=2, list_weights=[0.9, 0.1])[0][0] == "a"
    assert rrf_merge(lists, limit=2, list_weights=[0.1, 0.9])[0][0] == "b"


def test_rrf_list_weights_length_must_match_ranked_lists():
    with pytest.raises(ValueError, match="list_weights length"):
        rrf_merge([[("a", 1.0)]], list_weights=[0.5, 0.5])


def test_dedupe_drops_subset_chunk_when_semantic_enabled():
    hits = [
        ChunkHit(id="1", text="short", score=0.5),
        ChunkHit(id="2", text="short and much longer detail", score=0.9),
    ]
    out = dedupe_chunks(hits, True)
    texts = [h.text for h in out]
    assert "short and much longer detail" in texts
    assert "short" not in texts


def test_dedupe_hash_only_keeps_distinct_when_semantic_disabled():
    hits = [
        ChunkHit(id="1", text="short", score=0.5),
        ChunkHit(id="2", text="short and much longer detail", score=0.9),
    ]
    out = dedupe_chunks(hits, False)
    assert len(out) == 2


def test_budget_keeps_constraint_lines():
    hits = [
        ChunkHit(id="1", text="filler " * 200, score=0.5),
        ChunkHit(id="2", text="ERROR: disk full on /dev/sda", score=0.9),
    ]
    kept = apply_context_budget(hits, budget_chars=120)
    assert any("ERROR" in h.text for h in kept)


def test_recency_boost_noop_without_timestamp():
    score = _apply_recency_boost(0.5, {})
    assert score == 0.5


def test_parse_rewrite_json_rejects_non_string_query():
    assert _parse_rewrite_json('{"query": null}') is None
    assert _parse_rewrite_json('{"query": 42}') is None


def test_llm_rewrite_rejects_dropped_literal(monkeypatch):
    monkeypatch.setattr(settings, "enable_query_rewrite", True)
    monkeypatch.setattr(settings, "enable_query_rewrite_llm", True)
    monkeypatch.setattr(settings, "intent_model", "test-model")

    async def fake_llm(_model, _query, _timeout):
        return '{"query": "kubernetes pod scheduling"}'

    monkeypatch.setattr(
        "rag_proxy.stages.tier2_rewrite.rewrite_query_via_model",
        fake_llm,
    )
    ctx = RequestContext(query_text="k8s pod on 192.168.1.36")
    asyncio.run(run_rewrite(ctx))
    assert "192.168.1.36" in ctx.retrieval_query
    assert "rewrite:llm" not in ctx.stage_trace


def test_hybrid_rrf_includes_sparse_only_doc(monkeypatch):
    monkeypatch.setattr(settings, "enable_hybrid_retrieval", True)
    monkeypatch.setattr(settings, "sparse_index_url", "http://sparse.test")
    monkeypatch.setattr(settings, "hybrid_dense_weight", 0.95)

    dense = [
        ChunkHit(id=f"d{i}", text=f"dense {i}", score=0.9 - i * 0.01)
        for i in range(5)
    ]

    async def fake_dense(*_a, **_k):
        return dense

    async def fake_sparse(_query, _limit):
        return [{"id": "sparse-only", "score": 0.99, "payload": {"text": "sparse hit"}}]

    monkeypatch.setattr("rag_proxy.clients.qdrant._dense_chunks", fake_dense)
    monkeypatch.setattr("rag_proxy.clients.qdrant.sparse_search", fake_sparse)

    hits = asyncio.run(hybrid_search("test query", limit=5))
    ids = [h.id for h in hits]
    assert "sparse-only" in ids


def test_chunk_texts_property_matches_hits():
    ctx = RequestContext(
        hits=[
            ChunkHit(id="a", text="one", score=0.9),
            ChunkHit(id="b", text="", score=0.1),
            ChunkHit(id="c", text="two", score=0.8),
        ]
    )
    assert ctx.chunk_texts == ["one", "two"]

"""Unit tests for rewrite and dedupe."""

import asyncio

from rag_proxy.clients.qdrant import (
    _apply_recency_boost,
    hybrid_search,
    merge_fused_with_sparse_reserve,
)
from rag_proxy.config import settings
from rag_proxy.context import ChunkHit, RequestContext
from rag_proxy.stages.tier2_context import apply_context_budget, dedupe_chunks
from rag_proxy.stages.tier2_rewrite import (
    _rewrite_query_from_dict,
    rewrite_query_deterministic,
    run_rewrite,
)


def test_rewrite_preserves_ip_and_path():
    """Deterministic rewrite keeps IPs, versions, and paths."""
    q = "fix v1.2.3 on 192.168.1.36 /etc/nomad/config"
    out = rewrite_query_deterministic(q)
    assert "192.168.1.36" in out
    assert "/etc/nomad/config" in out


def test_rewrite_expands_k8s_glossary():
    """Deterministic rewrite expands k8s to kubernetes."""
    q = "k8s pod scheduling"
    out = rewrite_query_deterministic(q)
    assert "kubernetes" in out


def test_dedupe_drops_subset_chunk_when_semantic_enabled():
    """Semantic dedupe drops shorter chunks contained in longer ones."""
    hits = [
        ChunkHit(id="1", text="short", score=0.5),
        ChunkHit(id="2", text="short and much longer detail", score=0.9),
    ]
    out = dedupe_chunks(hits, True)
    texts = [h.text for h in out]
    assert "short and much longer detail" in texts
    assert "short" not in texts


def test_dedupe_hash_only_keeps_distinct_when_semantic_disabled():
    """Hash-only dedupe keeps distinct chunks when semantic is off."""
    hits = [
        ChunkHit(id="1", text="short", score=0.5),
        ChunkHit(id="2", text="short and much longer detail", score=0.9),
    ]
    out = dedupe_chunks(hits, False)
    assert len(out) == 2


def test_budget_keeps_constraint_lines():
    """Context budget keeps high-signal lines like errors."""
    hits = [
        ChunkHit(id="1", text="filler " * 200, score=0.5),
        ChunkHit(id="2", text="ERROR: disk full on /dev/sda", score=0.9),
    ]
    kept = apply_context_budget(hits, 120)
    assert any("ERROR" in h.text for h in kept)


def test_merge_fused_with_sparse_reserve_keeps_sparse_slot():
    """Sparse reserve keeps sparse-only docs in the merged id list."""
    fused = ["d0", "d1", "d2", "d3", "d4"]
    sparse_only = ["sparse-only"]
    merged = merge_fused_with_sparse_reserve(fused, sparse_only, limit=5)
    assert "sparse-only" in merged
    assert len(merged) == 5


def test_recency_boost_noop_without_timestamp():
    """Recency boost is a no-op when payload has no timestamp."""
    score = _apply_recency_boost(0.5, {})
    assert score == 0.5


def test_rewrite_query_from_dict_rejects_non_string_query():
    """Rewrite JSON parser rejects non-string query values."""
    assert _rewrite_query_from_dict({"query": None}) is None
    assert _rewrite_query_from_dict({"query": 42}) is None


def test_llm_rewrite_rejects_dropped_literal(monkeypatch):
    """LLM rewrite is rejected when it drops preserved literals."""

    monkeypatch.setattr(settings, "enable_query_rewrite", True)
    monkeypatch.setattr(settings, "enable_query_rewrite_llm", True)
    monkeypatch.setattr(settings, "intent_model", "test-model")

    async def fake_llm(_model, _query, _timeout, **_kwargs):
        """Stub LLM rewrite that drops the IP literal."""
        return {"query": "kubernetes pod scheduling"}

    monkeypatch.setattr(
        "rag_proxy.stages.tier2_rewrite.rewrite_query_via_model",
        fake_llm,
    )
    ctx = RequestContext(query_text="k8s pod on 192.168.1.36")
    asyncio.run(run_rewrite(ctx))
    assert "192.168.1.36" in ctx.retrieval_query
    assert "rewrite:llm" not in ctx.stage_trace


def test_hybrid_search_dense_failure_returns_sparse(monkeypatch):
    """Hybrid search returns sparse hits when dense search fails."""

    monkeypatch.setattr(settings, "enable_hybrid_retrieval", True)
    monkeypatch.setattr(settings, "sparse_index_url", "http://sparse.test")

    async def fake_dense(*_a, **_k):
        """Simulate dense search failure."""
        raise RuntimeError("dense down")

    async def fake_sparse(_query, _limit):
        """Return a single sparse hit."""
        return [{"id": "sparse-only", "score": 0.9, "payload": {"text": "sparse hit"}}]

    monkeypatch.setattr("rag_proxy.clients.qdrant._dense_chunks", fake_dense)
    monkeypatch.setattr("rag_proxy.clients.qdrant.sparse_search", fake_sparse)

    hits = asyncio.run(hybrid_search("test query", limit=3))
    assert len(hits) == 1
    assert hits[0].id == "sparse-only"


def test_hybrid_rrf_includes_sparse_only_doc(monkeypatch):
    """Hybrid RRF includes sparse-only documents in results."""

    monkeypatch.setattr(settings, "enable_hybrid_retrieval", True)
    monkeypatch.setattr(settings, "sparse_index_url", "http://sparse.test")
    monkeypatch.setattr(settings, "hybrid_dense_weight", 0.95)

    dense = [
        ChunkHit(id=f"d{i}", text=f"dense {i}", score=0.9 - i * 0.01)
        for i in range(5)
    ]

    async def fake_dense(*_a, **_k):
        """Return fixed dense chunk hits."""
        return dense

    async def fake_sparse(_query, _limit):
        """Return a sparse-only hit."""
        return [{"id": "sparse-only", "score": 0.99, "payload": {"text": "sparse hit"}}]

    monkeypatch.setattr("rag_proxy.clients.qdrant._dense_chunks", fake_dense)
    monkeypatch.setattr("rag_proxy.clients.qdrant.sparse_search", fake_sparse)

    hits = asyncio.run(hybrid_search("test query", limit=5))
    ids = [h.id for h in hits]
    assert "sparse-only" in ids


def test_chunk_texts_property_matches_hits():
    """chunk_texts returns non-empty hit texts in order."""
    ctx = RequestContext(
        hits=[
            ChunkHit(id="a", text="one", score=0.9),
            ChunkHit(id="b", text="", score=0.1),
            ChunkHit(id="c", text="two", score=0.8),
        ]
    )
    assert ctx.chunk_texts == ["one", "two"]

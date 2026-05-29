"""Unit tests for rewrite, RRF merge, and dedupe."""

from rag_proxy.clients.qdrant import rrf_merge
from rag_proxy.context import ChunkHit
from rag_proxy.stages.tier2_context import apply_context_budget, dedupe_chunks
from rag_proxy.stages.tier2_rewrite import rewrite_query_deterministic


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


def test_dedupe_drops_subset_chunk():
    hits = [
        ChunkHit(id="1", text="short", score=0.5),
        ChunkHit(id="2", text="short and much longer detail", score=0.9),
    ]
    out = dedupe_chunks(hits, False)
    texts = [h.text for h in out]
    assert "short and much longer detail" in texts


def test_budget_keeps_constraint_lines():
    hits = [
        ChunkHit(id="1", text="filler " * 200, score=0.5),
        ChunkHit(id="2", text="ERROR: disk full on /dev/sda", score=0.9),
    ]
    kept = apply_context_budget(hits, budget_chars=120)
    assert any("ERROR" in h.text for h in kept)

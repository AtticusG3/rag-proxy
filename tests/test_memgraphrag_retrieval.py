"""MemGraphRAG retrieval HTTP contracts and pipeline behavior."""

from __future__ import annotations

import asyncio
import threading
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import httpx

import numpy as np

from conftest import FakeAsyncClient
from rag_proxy.config import settings
from rag_proxy.memgraphrag.cache import build_memory_index
from rag_proxy.memgraphrag.memory import ThreeLayerMemory
from rag_proxy.memgraphrag.retrieval import MemGraphRetriever


def _minimal_memory() -> ThreeLayerMemory:
    """One schema, fact, and passage linked for PPR -> passage retrieval."""
    mem = ThreeLayerMemory()
    schema_idx = mem.add_schema("Person", "knows", "Person")
    passage_idx = mem.add_passage("chunk-1", "Alice knows Bob in the lab.", fact_indices=[])
    fact_idx = mem.add_fact("Alice", "knows", "Bob", schema_idx, passage_idx)
    mem.passages[passage_idx].fact_indices.append(fact_idx)
    mem.set_fact_embedding(fact_idx, [1.0, 0.0, 0.0])
    return mem


def _minimal_index():
    return build_memory_index(_minimal_memory())


def test_embed_query_reuses_retrieve_vector_else_shared_embed() -> None:
    """A private embed client skips truncation and the cache; reuse the retrieve vector."""
    retriever = MemGraphRetriever(
        index=_minimal_index(),
        embed_url="http://embed.test",
        reranker_url="",
    )

    async def fail_embed(_text: str, **_kwargs):
        raise AssertionError("provided query vector must skip a second embed")

    with patch(
        "rag_proxy.memgraphrag.retrieval.embed_text",
        side_effect=fail_embed,
    ), patch("rag_proxy.memgraphrag.retrieval.httpx.AsyncClient") as client:
        vector = asyncio.run(
            retriever._embed_query("who knows Bob?", query_embedding=[0.2, 0.4, 0.0])
        )

    assert vector == [0.2, 0.4, 0.0]
    client.assert_not_called()

    seen: list[str] = []

    async def fake_embed(text: str, **_kwargs):
        seen.append(text)
        return [1.0, 0.0, 0.0]

    with patch("rag_proxy.memgraphrag.retrieval.embed_text", side_effect=fake_embed):
        vector = asyncio.run(retriever._embed_query("who knows Bob?"))

    assert vector == [1.0, 0.0, 0.0]
    assert seen == ["who knows Bob?"]


def test_score_facts_uses_precomputed_embeddings_only() -> None:
    """score_facts embeds the query once and scores facts from stored vectors."""
    retriever = MemGraphRetriever(
        index=_minimal_index(),
        embed_url="http://embed.test",
        reranker_url="",
    )
    embed_calls: list[str] = []

    async def fake_embed(text: str, **_kwargs):
        embed_calls.append(text)
        return [1.0, 0.0, 0.0]

    with patch("rag_proxy.memgraphrag.retrieval.embed_text", side_effect=fake_embed):
        scored = asyncio.run(retriever.score_facts("who knows Bob?"))

    assert len(embed_calls) == 1
    assert embed_calls[0] == "who knows Bob?"
    assert scored == [(0, 1.0)]


def test_score_facts_skips_facts_without_embeddings() -> None:
    """Facts missing precomputed vectors are not scored."""
    mem = ThreeLayerMemory()
    schema_idx = mem.add_schema("Person", "knows", "Person")
    passage_idx = mem.add_passage("chunk-1", "Alice knows Bob.", fact_indices=[])
    mem.add_fact("Alice", "knows", "Bob", schema_idx, passage_idx)
    retriever = MemGraphRetriever(
        index=build_memory_index(mem),
        embed_url="http://embed.test",
        reranker_url="",
    )

    async def fake_embed(_text: str, **_kwargs):
        return [1.0, 0.0, 0.0]

    with patch("rag_proxy.memgraphrag.retrieval.embed_text", side_effect=fake_embed):
        scored = asyncio.run(retriever.score_facts("who knows Bob?"))

    assert scored == []


def test_memory_roundtrip_preserves_fact_embeddings(tmp_path) -> None:
    """SQLite save/load keeps fact embedding vectors for online scoring."""
    mem = _minimal_memory()
    db_path = tmp_path / "mem.sqlite"
    mem.save(db_path)
    loaded = ThreeLayerMemory(db_path=db_path)
    assert loaded.facts[0].embedding == [1.0, 0.0, 0.0]


def test_rerank_facts_uses_sidecar_pairs_contract() -> None:
    """Rerank must POST /rerank with pairs + top_k and honor indices order."""
    retriever = MemGraphRetriever(
        index=_minimal_index(),
        embed_url="http://embed.test",
        reranker_url="http://rerank.test",
    )
    captured: list[tuple[str, dict]] = []

    async def post(url: str, json: dict | None = None, **_kwargs):
        captured.append((url, json or {}))
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = {"indices": [0]}
        return response

    with patch(
        "rag_proxy.memgraphrag.retrieval.httpx.AsyncClient",
        return_value=FakeAsyncClient(AsyncMock(side_effect=post)),
    ):
        ranked = asyncio.run(
            retriever.rerank_facts(
                "who knows Bob?",
                [0],
                ["(Alice, knows, Bob)"],
            )
        )

    assert ranked == [(0, 1.0)]
    assert len(captured) == 1
    url, payload = captured[0]
    assert url == "http://rerank.test/rerank"
    assert payload["top_k"] == 1
    assert payload["pairs"] == [
        {"query": "who knows Bob?", "document": "(Alice, knows, Bob)"}
    ]


def test_retrieve_returns_chunk_hits_when_facts_score() -> None:
    """Full pipeline returns memgraphrag ChunkHits when facts and passages link."""
    retriever = MemGraphRetriever(
        index=_minimal_index(),
        embed_url="http://embed.test",
        reranker_url="http://rerank.test",
        top_k=3,
        ppr_threshold=0.0,
    )

    async def post(url: str, json: dict | None = None, **_kwargs):
        if url.endswith("/rerank"):
            response = MagicMock()
            response.raise_for_status = MagicMock()
            response.json.return_value = {"indices": [0]}
            return response
        raise AssertionError(f"unexpected url: {url}")

    async def fake_embed(_text: str, **_kwargs):
        return [1.0, 0.0, 0.0]

    with patch(
        "rag_proxy.memgraphrag.retrieval.embed_text",
        side_effect=fake_embed,
    ), patch(
        "rag_proxy.memgraphrag.retrieval.httpx.AsyncClient",
        return_value=FakeAsyncClient(AsyncMock(side_effect=post)),
    ):
        hits = asyncio.run(retriever.retrieve("who knows Bob?"))

    assert len(hits) == 1
    assert hits[0].id == "chunk-1"
    assert hits[0].source == "memgraphrag"
    assert "Alice" in hits[0].text


def test_retrieve_returns_empty_when_no_facts_scored() -> None:
    """Empty memory yields no hits instead of a dense fallback."""
    mem = ThreeLayerMemory()
    retriever = MemGraphRetriever(
        index=build_memory_index(mem),
        embed_url="http://embed.test",
        reranker_url="http://rerank.test",
    )

    async def fake_embed(_text: str, **_kwargs):
        return [1.0, 0.0, 0.0]

    with patch(
        "rag_proxy.memgraphrag.retrieval.embed_text",
        side_effect=fake_embed,
    ), patch("rag_proxy.memgraphrag.retrieval.httpx.AsyncClient") as client:
        hits = asyncio.run(retriever.retrieve("orphan query"))

    client.assert_not_called()

    assert hits == []


def test_rerank_facts_fail_open_on_sidecar_error() -> None:
    """A failed rerank must abstain, not hand back uniform scores.

    Uniform scores would seed PPR with every candidate fact weighted equally,
    which is worse than the dense ordering the caller already has.
    """
    retriever = MemGraphRetriever(
        index=_minimal_index(),
        embed_url="http://embed.test",
        reranker_url="http://rerank.test",
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
        "rag_proxy.memgraphrag.retrieval.httpx.AsyncClient",
        return_value=FakeAsyncClient(post),
    ):
        ranked = asyncio.run(
            retriever.rerank_facts("q", [0], ["(Alice, knows, Bob)"])
        )

    assert ranked is None


def test_reranker_skipped_when_globally_disabled(monkeypatch) -> None:
    """MemGraphRAG must not call a reranker sidecar that ENABLE_RERANKER turned off."""
    monkeypatch.setattr(settings, "enable_reranker", False)
    monkeypatch.setattr(settings, "reranker_url", "http://rerank.test")

    retriever = MemGraphRetriever(index=_minimal_index(), embed_url="http://embed.test")
    assert retriever.reranker_url == ""

    ranked = asyncio.run(retriever.rerank_facts("q", [0], ["(Alice, knows, Bob)"]))
    assert ranked is None


def test_retrieve_seeds_ppr_with_dense_scores_without_reranker(monkeypatch) -> None:
    """With no reranker, PPR seeds must keep the dense similarity ordering."""
    monkeypatch.setattr(settings, "enable_reranker", False)

    mem = ThreeLayerMemory()
    schema_idx = mem.add_schema("Person", "knows", "Person")
    near = mem.add_passage("chunk-near", "Alice knows Bob.", [])
    far = mem.add_passage("chunk-far", "Zed knows Yan.", [])
    near_fact = mem.add_fact("Alice", "knows", "Bob", schema_idx, near)
    far_fact = mem.add_fact("Zed", "knows", "Yan", schema_idx, far)
    mem.facts[near_fact].embedding = [1.0, 0.0, 0.0]
    mem.facts[far_fact].embedding = [0.2, 0.98, 0.0]
    mem.add_passage("chunk-near", "Alice knows Bob.", [near_fact])
    mem.add_passage("chunk-far", "Zed knows Yan.", [far_fact])

    retriever = MemGraphRetriever(
        index=build_memory_index(mem),
        embed_url="http://embed.test",
        top_k=2,
        ppr_threshold=0.0,
    )

    async def fake_embed(_text: str, **_kwargs):
        return [1.0, 0.0, 0.0]

    with patch(
        "rag_proxy.memgraphrag.retrieval.embed_text",
        side_effect=fake_embed,
    ), patch("rag_proxy.memgraphrag.retrieval.httpx.AsyncClient") as client:
        hits = asyncio.run(retriever.retrieve("who knows Bob?"))

    client.assert_not_called()

    assert [h.id for h in hits] == ["chunk-near", "chunk-far"]
    # Uniform seeds would tie these; the gap proves dense similarity survived.
    assert hits[0].score > hits[1].score


def _baseline_fact_scores(
    mem: ThreeLayerMemory, query_emb: np.ndarray
) -> list[tuple[int, float]]:
    """Loop-based cosine scores used as oracle for vectorized scoring."""
    query_norm = float(np.linalg.norm(query_emb))
    if query_norm == 0:
        return []
    scored: list[tuple[int, float]] = []
    for fi, fact in mem.facts.items():
        if not fact.embedding:
            continue
        fact_emb = np.array(fact.embedding, dtype=np.float32)
        fact_norm = float(np.linalg.norm(fact_emb))
        if fact_norm == 0:
            continue
        score = float(np.dot(query_emb, fact_emb) / (query_norm * fact_norm))
        scored.append((fi, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def test_score_facts_vectorized_matches_baseline() -> None:
    """Vectorized matmul scoring must match per-fact cosine loop."""
    mem = ThreeLayerMemory()
    schema_idx = mem.add_schema("Person", "knows", "Person")
    passage_idx = mem.add_passage("chunk-1", "Alice knows Bob.", fact_indices=[])
    fact_a = mem.add_fact("Alice", "knows", "Bob", schema_idx, passage_idx)
    fact_b = mem.add_fact("Bob", "works_at", "Lab", schema_idx, passage_idx)
    mem.set_fact_embedding(fact_a, [1.0, 0.0, 0.0])
    mem.set_fact_embedding(fact_b, [0.6, 0.8, 0.0])

    index = build_memory_index(mem)
    retriever = MemGraphRetriever(index=index, embed_url="http://embed.test", reranker_url="")
    query_emb = np.array([0.8, 0.6, 0.0], dtype=np.float32)

    vectorized = retriever._score_facts_vectorized(query_emb)
    baseline = _baseline_fact_scores(mem, query_emb)

    assert vectorized == baseline
    assert vectorized[0][0] == fact_b


def test_concurrent_retrieve_overlaps_during_cold_cache(tmp_path) -> None:
    """Two concurrent retrieve() calls should overlap embed work after index load."""
    mem = _minimal_memory()
    db_path = tmp_path / "mem.sqlite"
    mem.save(db_path)

    from rag_proxy.memgraphrag.cache import get_memory_index, invalidate_memory_index

    invalidate_memory_index()
    load_release = threading.Event()

    def slow_load(path):
        load_release.wait(timeout=5.0)
        return _minimal_index()

    active_embeds = 0
    max_active_embeds = 0
    embed_lock = threading.Lock()

    async def slow_embed(_text: str, **_kwargs):
        nonlocal active_embeds, max_active_embeds
        with embed_lock:
            active_embeds += 1
            max_active_embeds = max(max_active_embeds, active_embeds)
        await asyncio.sleep(0.05)
        with embed_lock:
            active_embeds -= 1
        return [1.0, 0.0, 0.0]

    async def run_pair() -> None:
        with patch("rag_proxy.memgraphrag.cache._load_index", side_effect=slow_load):
            load_release.set()
            index = await asyncio.to_thread(get_memory_index, db_path)
            retriever = MemGraphRetriever(
                index=index,
                embed_url="http://embed.test",
                reranker_url="",
            )
            with patch(
                "rag_proxy.memgraphrag.retrieval.embed_text",
                side_effect=slow_embed,
            ):
                await asyncio.gather(
                    retriever.retrieve("who knows Bob?"),
                    retriever.retrieve("who knows Alice?"),
                )

    started = time.monotonic()
    asyncio.run(run_pair())
    elapsed = time.monotonic() - started

    assert max_active_embeds >= 2
    assert elapsed < 0.18


def test_ppr_dangling_node_does_not_grow_mass() -> None:
    """Undamped dangling redistribution pushes an isolated seed above 1 and swamps edges."""
    retriever = MemGraphRetriever(
        index=_minimal_index(),
        ppr_iterations=20,
        ppr_damping=0.85,
        ppr_threshold=0.0,
    )
    scores = retriever._ppr({0: 1.0}, {0: []})
    assert scores[0] == pytest.approx(1.0)


def test_ppr_default_threshold_keeps_diluted_neighbors() -> None:
    """A 0.01 cutoff drops neighbors once seed mass spreads; top_k truncates instead."""
    retriever = MemGraphRetriever(index=_minimal_index())
    assert retriever.ppr_threshold == 0.0
    neighbors = list(range(1, 80))
    adj = {0: neighbors}
    for node in neighbors:
        adj[node] = []
    scores = retriever._ppr({0: 1.0}, adj)
    assert 1 in scores


def test_ppr_walks_seed_neighborhood_not_full_schema_clique() -> None:
    """Schema cliques make a full-graph walk dilute every seed under a fixed threshold."""
    retriever = MemGraphRetriever(index=_minimal_index())
    adj = {0: list(range(1, 101))}
    for node in range(1, 101):
        adj[node] = []
    subgraph = retriever._seed_subgraph({0: 1.0}, adj)
    assert len(subgraph) <= 33
    assert 0 in subgraph
    assert 100 not in subgraph

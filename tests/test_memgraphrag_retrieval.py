"""MemGraphRAG retrieval HTTP contracts and pipeline behavior."""

from __future__ import annotations

import asyncio
import threading
import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

import numpy as np

from conftest import FakeAsyncClient
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


def _embedding_response() -> MagicMock:
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = {"data": [{"embedding": [1.0, 0.0, 0.0]}]}
    return response


def test_embed_query_uses_openai_embeddings_contract() -> None:
    """Embed client must call /v1/embeddings with model + input like the rest of the stack."""
    retriever = MemGraphRetriever(
        index=_minimal_index(),
        embed_url="http://embed.test",
        reranker_url="",
    )
    captured: list[tuple[str, dict]] = []

    async def post(url: str, json: dict | None = None, **_kwargs):
        captured.append((url, json or {}))
        return _embedding_response()

    with patch(
        "rag_proxy.memgraphrag.retrieval.httpx.AsyncClient",
        return_value=FakeAsyncClient(AsyncMock(side_effect=post)),
    ):
        vector = asyncio.run(retriever._embed_query("who knows Bob?"))

    assert vector == [1.0, 0.0, 0.0]
    assert len(captured) == 1
    url, payload = captured[0]
    assert url == "http://embed.test/v1/embeddings"
    assert payload == {"model": "nomic-embed-text-v1.5", "input": "who knows Bob?"}


def test_score_facts_uses_precomputed_embeddings_only() -> None:
    """score_facts embeds the query once and scores facts from stored vectors."""
    retriever = MemGraphRetriever(
        index=_minimal_index(),
        embed_url="http://embed.test",
        reranker_url="",
    )
    embed_calls: list[str] = []

    async def post(url: str, json: dict | None = None, **_kwargs):
        embed_calls.append((json or {}).get("input", ""))
        return _embedding_response()

    with patch(
        "rag_proxy.memgraphrag.retrieval.httpx.AsyncClient",
        return_value=FakeAsyncClient(AsyncMock(side_effect=post)),
    ):
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

    with patch(
        "rag_proxy.memgraphrag.retrieval.httpx.AsyncClient",
        return_value=FakeAsyncClient(AsyncMock(return_value=_embedding_response())),
    ):
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
        if url.endswith("/v1/embeddings"):
            return _embedding_response()
        if url.endswith("/rerank"):
            response = MagicMock()
            response.raise_for_status = MagicMock()
            response.json.return_value = {"indices": [0]}
            return response
        raise AssertionError(f"unexpected url: {url}")

    with patch(
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

    async def post(url: str, json: dict | None = None, **_kwargs):
        if url.endswith("/v1/embeddings"):
            return _embedding_response()
        raise AssertionError("rerank must not run when no facts exist")

    with patch(
        "rag_proxy.memgraphrag.retrieval.httpx.AsyncClient",
        return_value=FakeAsyncClient(AsyncMock(side_effect=post)),
    ):
        hits = asyncio.run(retriever.retrieve("orphan query"))

    assert hits == []


def test_rerank_facts_fail_open_on_sidecar_error() -> None:
    """Reranker HTTP errors preserve fact order with neutral scores."""
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

    assert ranked == [(0, 1.0)]


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

    async def slow_post(url: str, json: dict | None = None, **_kwargs):
        nonlocal active_embeds, max_active_embeds
        with embed_lock:
            active_embeds += 1
            max_active_embeds = max(max_active_embeds, active_embeds)
        await asyncio.sleep(0.05)
        with embed_lock:
            active_embeds -= 1
        return _embedding_response()

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
                "rag_proxy.memgraphrag.retrieval.httpx.AsyncClient",
                return_value=FakeAsyncClient(AsyncMock(side_effect=slow_post)),
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

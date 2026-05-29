"""Qdrant dense + optional sparse / hybrid retrieval."""

from __future__ import annotations

import logging
from collections import defaultdict

import httpx

from rag_proxy.clients.embed import embed_text
from rag_proxy.config import settings
from rag_proxy.context import ChunkHit
from rag_proxy.legacy_rag import extract_chunk_text, search_qdrant_dense

log = logging.getLogger("rag-proxy")


def rrf_merge(
    ranked_lists: list[list[tuple[str, float]]],
    k: int = 60,
    limit: int = 20,
) -> list[tuple[str, float]]:
    """Reciprocal rank fusion over document ids."""
    scores: dict[str, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, (doc_id, _score) in enumerate(ranked):
            scores[doc_id] += 1.0 / (k + rank + 1)
    merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return merged[:limit]


async def sparse_search(query: str, limit: int) -> list[dict]:
    """Optional BM25 sidecar; fail-open to empty list."""
    if not settings.sparse_index_url:
        return []
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(
                f"{settings.sparse_index_url.rstrip('/')}/search",
                json={"query": query, "limit": limit, "collection": settings.qdrant_collection},
            )
            r.raise_for_status()
            return r.json().get("results", [])
    except Exception as e:
        log.warning(f"Sparse search failed: {e}")
        return []


def _hit_to_chunk(hit: dict, source: str) -> ChunkHit:
    text = extract_chunk_text(hit)
    hit_id = str(hit.get("id", hit.get("point_id", "")))
    payload = hit.get("payload", {})
    score = float(hit.get("score", 0.0))
    return ChunkHit(id=hit_id, text=text, score=score, source=source, metadata=payload)


async def hybrid_search(
    query: str,
    limit: int,
    score_threshold: float | None = None,
    no_cache: bool = False,
) -> list[ChunkHit]:
    vector = await embed_text(query, no_cache=no_cache)
    if vector is None:
        return []

    dense_hits = await search_qdrant_dense(
        vector, limit=limit, score_threshold=score_threshold
    )
    dense_chunks = [_hit_to_chunk(h, "dense") for h in dense_hits if extract_chunk_text(h)]

    if not settings.enable_hybrid_retrieval:
        return dense_chunks

    sparse_raw = await sparse_search(query, limit)
    if not sparse_raw:
        return dense_chunks

    dense_ranked = [(c.id, c.score) for c in dense_chunks]
    sparse_ranked = [
        (str(h.get("id", i)), float(h.get("score", 0.0)))
        for i, h in enumerate(sparse_raw)
    ]
    fused_ids = [doc_id for doc_id, _ in rrf_merge([dense_ranked, sparse_ranked], limit=limit)]

    by_id = {c.id: c for c in dense_chunks}
    for h in sparse_raw:
        cid = str(h.get("id", ""))
        if cid and cid not in by_id:
            by_id[cid] = _hit_to_chunk(h, "sparse")

    ordered: list[ChunkHit] = []
    for doc_id in fused_ids:
        if doc_id in by_id:
            ordered.append(by_id[doc_id])
    return ordered

"""Qdrant dense + optional sparse / hybrid retrieval."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from datetime import datetime

import httpx

from rag_proxy.clients.embed import embed_text
from rag_proxy.config import settings
from rag_proxy.context import ChunkHit
from rag_proxy.legacy_rag import extract_chunk_text, search_qdrant_dense

log = logging.getLogger("rag-proxy")

_RECENCY_KEYS = ("updated_at", "mtime", "timestamp")


def rrf_merge(
    ranked_lists: list[list[tuple[str, float]]],
    k: int = 60,
    limit: int = 20,
    list_weights: list[float] | None = None,
) -> list[tuple[str, float]]:
    """Reciprocal rank fusion over document ids.

    When list_weights is set, each ranked list is scaled by its weight (e.g. dense vs sparse).
    """
    scores: dict[str, float] = defaultdict(float)
    if list_weights is not None:
        if len(list_weights) != len(ranked_lists):
            raise ValueError(
                f"list_weights length ({len(list_weights)}) "
                f"must match ranked_lists length ({len(ranked_lists)})"
            )
        weights = list_weights
    else:
        weights = [1.0] * len(ranked_lists)
    for weight, ranked in zip(weights, ranked_lists):
        for rank, (doc_id, _score) in enumerate(ranked):
            scores[doc_id] += weight * (1.0 / (k + rank + 1))
    merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return merged[:limit]


def _parse_recency_epoch(payload: dict) -> float | None:
    for key in _RECENCY_KEYS:
        raw = payload.get(key)
        if raw is None:
            continue
        if isinstance(raw, (int, float)):
            ts = float(raw)
            if ts > 1e12:
                ts /= 1000.0
            return ts
        if isinstance(raw, str):
            try:
                return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
            except ValueError:
                continue
    return None


def _apply_recency_boost(score: float, payload: dict) -> float:
    if settings.recency_weight <= 0:
        return score
    epoch = _parse_recency_epoch(payload)
    if epoch is None:
        return score
    days = max(0.0, (time.time() - epoch) / 86400.0)
    decay = max(0.0, 1.0 - days / 365.0)
    return score + settings.recency_weight * decay


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
    score = _apply_recency_boost(float(hit.get("score", 0.0)), payload)
    return ChunkHit(id=hit_id, text=text, score=score, source=source, metadata=payload)


async def _dense_chunks(
    query: str,
    limit: int,
    score_threshold: float | None,
    no_cache: bool,
) -> list[ChunkHit]:
    vector = await embed_text(query, no_cache=no_cache)
    if vector is None:
        return []
    dense_hits = await search_qdrant_dense(
        vector, limit=limit, score_threshold=score_threshold
    )
    return [_hit_to_chunk(h, "dense") for h in dense_hits if extract_chunk_text(h)]


async def hybrid_search(
    query: str,
    limit: int,
    score_threshold: float | None = None,
    no_cache: bool = False,
) -> list[ChunkHit]:
    if not settings.enable_hybrid_retrieval or not settings.sparse_index_url:
        return await _dense_chunks(query, limit, score_threshold, no_cache)

    sparse_raw, dense_chunks = await asyncio.gather(
        sparse_search(query, limit),
        _dense_chunks(query, limit, score_threshold, no_cache),
    )
    if not sparse_raw:
        return dense_chunks

    dense_ranked = [(c.id, c.score) for c in dense_chunks]
    sparse_ranked = [
        (str(h.get("id", i)), float(h.get("score", 0.0)))
        for i, h in enumerate(sparse_raw)
    ]
    dense_w = settings.hybrid_dense_weight
    sparse_w = max(0.0, 1.0 - dense_w)
    fused_ids = [
        doc_id
        for doc_id, _ in rrf_merge(
            [dense_ranked, sparse_ranked],
            limit=limit,
            list_weights=[dense_w, sparse_w],
        )
    ]

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

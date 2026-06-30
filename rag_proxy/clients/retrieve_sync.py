"""Sync hybrid retrieval primitives (embed, dense, sparse, rerank, RRF)."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass

import httpx

from rag_proxy.chunk_text import extract_chunk_text
from rag_proxy.clients.retrieval_core import (
    dense_search_payload,
    embed_payload,
    parse_dense_hits,
    parse_embedding,
    parse_sparse_hits,
    sparse_search_payload,
)

log = logging.getLogger("rag-proxy")


@dataclass(frozen=True)
class RetrieveConfig:
    """URLs and thresholds for sync retrieval (no global settings dependency)."""

    embed_url: str
    qdrant_url: str
    qdrant_collection: str
    sparse_index_url: str
    reranker_url: str
    similarity_threshold: float
    hybrid_dense_weight: float
    embed_max_chars: int
    enable_hybrid: bool = True
    enable_rerank: bool = False
    rerank_top_k: int = 5
    rerank_timeout_sec: float = 2.5
    user_agent: str = "rag-proxy-retrieve-sync/1.0"


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
        if weight <= 0:
            continue
        for rank, (doc_id, _score) in enumerate(ranked):
            scores[doc_id] += weight * (1.0 / (k + rank + 1))
    merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return merged[:limit]


def _headers(config: RetrieveConfig) -> dict[str, str]:
    return {"User-Agent": config.user_agent}


def embed_query(config: RetrieveConfig, query: str) -> list[float] | None:
    """Embed query text via the standalone nomic-embed server."""
    trimmed = query.strip()[: config.embed_max_chars]
    if not trimmed:
        return None
    try:
        with httpx.Client(timeout=30.0, headers=_headers(config)) as client:
            r = client.post(
                f"{config.embed_url.rstrip('/')}/v1/embeddings",
                json=embed_payload(trimmed),
            )
            r.raise_for_status()
            return parse_embedding(r.json())
    except Exception as e:
        log.warning(f"Embedding failed: {e}")
        return None


def dense_search(
    config: RetrieveConfig,
    vector: list[float],
    *,
    limit: int,
    score_threshold: float | None = None,
) -> list[dict]:
    """Return top-k chunks from Qdrant above the similarity threshold."""
    threshold = (
        config.similarity_threshold if score_threshold is None else score_threshold
    )
    body = dense_search_payload(
        vector,
        limit,
        threshold,
        omit_zero_threshold=True,
    )
    try:
        with httpx.Client(timeout=10.0, headers=_headers(config)) as client:
            r = client.post(
                f"{config.qdrant_url.rstrip('/')}/collections/"
                f"{config.qdrant_collection}/points/search",
                json=body,
            )
            r.raise_for_status()
            return parse_dense_hits(r.json())
    except Exception as e:
        log.warning(f"Qdrant search failed: {e}")
        return []


def sparse_search(
    config: RetrieveConfig,
    query: str,
    *,
    limit: int,
) -> list[dict]:
    """Optional BM25 sidecar; fail-open to empty list."""
    if not config.sparse_index_url:
        return []
    body = sparse_search_payload(query, limit, config.qdrant_collection)
    try:
        with httpx.Client(timeout=10.0, headers=_headers(config)) as client:
            r = client.post(
                f"{config.sparse_index_url.rstrip('/')}/search",
                json=body,
            )
            r.raise_for_status()
            return parse_sparse_hits(r.json())
    except Exception as e:
        log.warning(f"Sparse search failed: {e}")
        return []


def rerank_pairs(
    config: RetrieveConfig,
    pairs: list[dict[str, str]],
    *,
    top_k: int | None = None,
) -> list[int]:
    """Rerank query-document pairs via sidecar; returns indices in ranked order."""
    if not config.enable_rerank or not config.reranker_url or not pairs:
        cap = top_k if top_k is not None else len(pairs)
        return list(range(min(cap, len(pairs))))
    effective_top_k = top_k if top_k is not None else config.rerank_top_k
    try:
        with httpx.Client(
            timeout=config.rerank_timeout_sec, headers=_headers(config)
        ) as client:
            r = client.post(
                f"{config.reranker_url.rstrip('/')}/rerank",
                json={"pairs": pairs, "top_k": effective_top_k},
            )
            r.raise_for_status()
            order = r.json().get("indices", [])
            if order:
                return [i for i in order if 0 <= i < len(pairs)]
    except Exception as e:
        log.warning(f"Rerank failed: {e}")
    return list(range(min(effective_top_k, len(pairs))))


def _hit_has_text(hit: dict) -> bool:
    return bool(extract_chunk_text(hit))


def hybrid_retrieve_with_dense_ids(
    config: RetrieveConfig,
    query: str,
    *,
    limit: int,
    score_threshold: float | None = None,
) -> tuple[list[dict], set[str]]:
    """Dense-only or RRF hybrid retrieval; returns hits and dense-source id set."""
    if not config.enable_hybrid or not config.sparse_index_url:
        vector = embed_query(config, query)
        if vector is None:
            return [], set()
        hits = dense_search(
            config, vector, limit=limit, score_threshold=score_threshold
        )
        hits = [h for h in hits if _hit_has_text(h)]
        dense_ids = {str(h.get("id", "")) for h in hits if h.get("id")}
        return hits, dense_ids

    vector = embed_query(config, query)
    dense_hits: list[dict] = []
    dense_ids: set[str] = set()
    if vector is not None:
        dense_hits = dense_search(
            config, vector, limit=limit, score_threshold=score_threshold
        )
        dense_hits = [h for h in dense_hits if _hit_has_text(h)]
        dense_ids = {str(h.get("id", "")) for h in dense_hits if h.get("id")}

    sparse_raw = sparse_search(config, query, limit=limit)
    if not sparse_raw:
        return dense_hits, dense_ids

    dense_ranked = [
        (str(h.get("id", "")), float(h.get("score", 0.0))) for h in dense_hits
    ]
    sparse_ranked = [
        (str(h.get("id", i)), float(h.get("score", 0.0)))
        for i, h in enumerate(sparse_raw)
    ]
    dense_w = config.hybrid_dense_weight
    sparse_w = max(0.0, 1.0 - dense_w)
    fused_ids = [
        doc_id
        for doc_id, _ in rrf_merge(
            [dense_ranked, sparse_ranked],
            limit=limit,
            list_weights=[dense_w, sparse_w],
        )
    ]

    by_id: dict[str, dict] = {str(h.get("id", "")): h for h in dense_hits}
    for h in sparse_raw:
        cid = str(h.get("id", ""))
        if cid and cid not in by_id and _hit_has_text(h):
            by_id[cid] = h

    ordered: list[dict] = []
    for doc_id in fused_ids:
        if doc_id in by_id:
            ordered.append(by_id[doc_id])
    return ordered, dense_ids


def hybrid_retrieve(
    config: RetrieveConfig,
    query: str,
    *,
    limit: int,
    score_threshold: float | None = None,
) -> list[dict]:
    """Dense-only or RRF hybrid dense+sparse retrieval; returns raw hit dicts."""
    hits, _ = hybrid_retrieve_with_dense_ids(
        config, query, limit=limit, score_threshold=score_threshold
    )
    return hits

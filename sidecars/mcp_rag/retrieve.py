"""Hybrid retrieval for MCP tools (embed + Qdrant + sparse + rerank)."""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import httpx

from chunk_text import extract_chunk_text

DEFAULT_EMBED_MODEL = "nomic-embed-text-v1.5"


@dataclass(frozen=True)
class RetrieveSettings:
    embed_url: str
    qdrant_url: str
    qdrant_collection: str
    sparse_index_url: str
    reranker_url: str
    hybrid_dense_weight: float
    similarity_threshold: float
    enable_hybrid: bool
    enable_rerank: bool
    embed_max_chars: int
    user_agent: str

    @classmethod
    def from_env(cls) -> RetrieveSettings:
        return cls(
            embed_url=os.getenv("EMBED_URL", "http://127.0.0.1:18089"),
            qdrant_url=os.getenv("QDRANT_URL", "http://127.0.0.1:6333"),
            qdrant_collection=os.getenv(
                "QDRANT_COLLECTION", "nomad_knowledge_base"
            ),
            sparse_index_url=os.getenv("SPARSE_INDEX_URL", "http://127.0.0.1:18096"),
            reranker_url=os.getenv("RERANKER_URL", "http://127.0.0.1:18095"),
            hybrid_dense_weight=float(os.getenv("HYBRID_DENSE_WEIGHT", "0.7")),
            similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.65")),
            enable_hybrid=os.getenv("ENABLE_HYBRID_RETRIEVAL", "true").lower()
            in ("1", "true", "yes", "on"),
            enable_rerank=os.getenv("ENABLE_RERANKER", "true").lower()
            in ("1", "true", "yes", "on"),
            embed_max_chars=int(os.getenv("EMBED_MAX_CHARS", "2000")),
            user_agent=os.getenv("MCP_RAG_USER_AGENT", "mcp-rag-context/1.0"),
        )


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    source: str
    title: str
    retrieval: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.chunk_id,
            "text": self.text,
            "score": round(self.score, 4),
            "source": self.source,
            "title": self.title,
            "retrieval": self.retrieval,
        }


def _headers(settings: RetrieveSettings) -> dict[str, str]:
    return {"User-Agent": settings.user_agent}


def embed_query(settings: RetrieveSettings, query: str) -> list[float]:
    trimmed = query.strip()[: settings.embed_max_chars]
    with httpx.Client(timeout=60.0, headers=_headers(settings)) as client:
        response = client.post(
            f"{settings.embed_url.rstrip('/')}/v1/embeddings",
            json={"model": DEFAULT_EMBED_MODEL, "input": [trimmed]},
        )
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]


def dense_search(
    settings: RetrieveSettings,
    vector: list[float],
    *,
    limit: int,
    score_threshold: float | None,
) -> list[dict[str, Any]]:
    body: dict[str, Any] = {
        "vector": vector,
        "limit": limit,
        "with_payload": True,
    }
    if score_threshold is not None and score_threshold > 0:
        body["score_threshold"] = score_threshold
    with httpx.Client(timeout=30.0, headers=_headers(settings)) as client:
        response = client.post(
            f"{settings.qdrant_url.rstrip('/')}/collections/"
            f"{settings.qdrant_collection}/points/search",
            json=body,
        )
        response.raise_for_status()
        return response.json().get("result", [])


def sparse_search(
    settings: RetrieveSettings,
    query: str,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    if not settings.sparse_index_url:
        return []
    try:
        with httpx.Client(timeout=15.0, headers=_headers(settings)) as client:
            response = client.post(
                f"{settings.sparse_index_url.rstrip('/')}/search",
                json={
                    "query": query,
                    "limit": limit,
                    "collection": settings.qdrant_collection,
                },
            )
            response.raise_for_status()
            return response.json().get("results", [])
    except httpx.HTTPError:
        return []


def rrf_merge(
    ranked_lists: list[list[tuple[str, float]]],
    *,
    limit: int,
    list_weights: list[float],
    k: int = 60,
) -> list[str]:
    scores: dict[str, float] = defaultdict(float)
    for weight, ranked in zip(list_weights, ranked_lists):
        if weight <= 0:
            continue
        for rank, (doc_id, _score) in enumerate(ranked):
            scores[doc_id] += weight * (1.0 / (k + rank + 1))
    merged = sorted(scores.items(), key=lambda row: row[1], reverse=True)
    return [doc_id for doc_id, _ in merged[:limit]]


def _hit_to_chunk(hit: dict[str, Any], retrieval: str) -> RetrievedChunk | None:
    text = extract_chunk_text(hit)
    if not text:
        return None
    payload = hit.get("payload") or {}
    return RetrievedChunk(
        chunk_id=str(hit.get("id", "")),
        text=text,
        score=float(hit.get("score", 0.0)),
        source=str(payload.get("source", "")),
        title=str(payload.get("title", "")),
        retrieval=retrieval,
    )


def hybrid_retrieve(
    settings: RetrieveSettings,
    query: str,
    *,
    top_k: int,
    score_threshold: float | None = None,
) -> list[RetrievedChunk]:
    threshold = (
        settings.similarity_threshold
        if score_threshold is None
        else score_threshold
    )
    candidate_k = max(top_k, top_k * 4)
    vector = embed_query(settings, query)
    dense_hits = dense_search(
        settings, vector, limit=candidate_k, score_threshold=threshold
    )
    dense_chunks = [
        c
        for c in (_hit_to_chunk(h, "dense") for h in dense_hits)
        if c is not None
    ]

    if not settings.enable_hybrid or not settings.sparse_index_url:
        return dense_chunks[:top_k]

    sparse_hits = sparse_search(settings, query, limit=candidate_k)
    if not sparse_hits:
        return dense_chunks[:top_k]

    dense_ranked = [(c.chunk_id, c.score) for c in dense_chunks]
    sparse_ranked = [
        (str(h.get("id", i)), float(h.get("score", 0.0)))
        for i, h in enumerate(sparse_hits)
    ]
    dense_w = settings.hybrid_dense_weight
    sparse_w = max(0.0, 1.0 - dense_w)
    fused_ids = rrf_merge(
        [dense_ranked, sparse_ranked],
        limit=candidate_k,
        list_weights=[dense_w, sparse_w],
    )

    by_id: dict[str, RetrievedChunk] = {c.chunk_id: c for c in dense_chunks}
    for hit in sparse_hits:
        cid = str(hit.get("id", ""))
        if cid and cid not in by_id:
            chunk = _hit_to_chunk(hit, "sparse")
            if chunk is not None:
                by_id[cid] = chunk

    ordered: list[RetrievedChunk] = []
    for doc_id in fused_ids:
        if doc_id in by_id:
            ordered.append(by_id[doc_id])
    return ordered[:candidate_k]


def rerank_chunks(
    settings: RetrieveSettings,
    query: str,
    chunks: list[RetrievedChunk],
    *,
    top_k: int,
) -> list[RetrievedChunk]:
    if not settings.enable_rerank or not settings.reranker_url or not chunks:
        return chunks[:top_k]
    pairs = [{"query": query, "document": c.text[: settings.embed_max_chars]} for c in chunks]
    try:
        with httpx.Client(timeout=30.0, headers=_headers(settings)) as client:
            response = client.post(
                f"{settings.reranker_url.rstrip('/')}/rerank",
                json={"pairs": pairs, "top_k": top_k},
            )
            response.raise_for_status()
            indices = response.json().get("indices", [])
    except httpx.HTTPError:
        return chunks[:top_k]
    reranked: list[RetrievedChunk] = []
    for index in indices:
        if 0 <= index < len(chunks):
            reranked.append(chunks[index])
    return reranked[:top_k]


def search_knowledge_base(
    query: str,
    *,
    top_k: int = 5,
    score_threshold: float | None = None,
    settings: RetrieveSettings | None = None,
) -> list[RetrievedChunk]:
    cfg = settings or RetrieveSettings.from_env()
    limit = max(1, min(top_k, 20))
    candidates = hybrid_retrieve(
        cfg,
        query,
        top_k=limit,
        score_threshold=score_threshold,
    )
    return rerank_chunks(cfg, query, candidates, top_k=limit)


def format_chunks_for_agent(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "No matching passages found in the knowledge base."
    lines = [f"Found {len(chunks)} passage(s):\n"]
    for idx, chunk in enumerate(chunks, start=1):
        title = chunk.title or "Untitled"
        source = chunk.source or "unknown"
        lines.append(
            f"### [{idx}] {title} (score {chunk.score:.3f}, {chunk.retrieval})\n"
            f"Source: `{source}`\n\n{chunk.text}\n"
        )
    return "\n".join(lines)


def fetch_index_status(settings: RetrieveSettings | None = None) -> dict[str, Any]:
    cfg = settings or RetrieveSettings.from_env()
    status: dict[str, Any] = {
        "collection": cfg.qdrant_collection,
        "qdrant_points": 0,
        "sparse_docs": 0,
        "sparse_status": "unknown",
    }
    try:
        with httpx.Client(timeout=10.0, headers=_headers(cfg)) as client:
            response = client.get(
                f"{cfg.qdrant_url.rstrip('/')}/collections/{cfg.qdrant_collection}"
            )
            if response.status_code == 200:
                status["qdrant_points"] = int(
                    response.json()["result"]["points_count"]
                )
    except httpx.HTTPError:
        status["qdrant_status"] = "error"

    if cfg.sparse_index_url:
        try:
            with httpx.Client(timeout=10.0, headers=_headers(cfg)) as client:
                response = client.get(f"{cfg.sparse_index_url.rstrip('/')}/health")
                if response.status_code == 200:
                    body = response.json()
                    status["sparse_docs"] = int(body.get("docs", 0))
                    status["sparse_status"] = str(body.get("status", "ok"))
        except httpx.HTTPError:
            status["sparse_status"] = "error"
    return status

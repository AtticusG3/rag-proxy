"""Hybrid retrieval for MCP tools (embed + Qdrant + sparse + rerank)."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

_REPO_ROOT = Path(__file__).resolve().parents[2]
_root = str(_REPO_ROOT)
if _root not in sys.path:
    sys.path.insert(0, _root)

from rag_proxy.chunk_text import extract_chunk_text
from rag_proxy.clients.retrieve_sync import (
    RetrieveConfig,
    hybrid_retrieve_with_dense_ids,
    rerank_pairs,
)


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


def _to_retrieve_config(settings: RetrieveSettings) -> RetrieveConfig:
    return RetrieveConfig(
        embed_url=settings.embed_url,
        qdrant_url=settings.qdrant_url,
        qdrant_collection=settings.qdrant_collection,
        sparse_index_url=settings.sparse_index_url,
        reranker_url=settings.reranker_url,
        similarity_threshold=settings.similarity_threshold,
        hybrid_dense_weight=settings.hybrid_dense_weight,
        embed_max_chars=settings.embed_max_chars,
        enable_hybrid=settings.enable_hybrid,
        enable_rerank=settings.enable_rerank,
        user_agent=settings.user_agent,
    )


def _headers(settings: RetrieveSettings) -> dict[str, str]:
    return {"User-Agent": settings.user_agent}


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
    config = _to_retrieve_config(settings)
    threshold = (
        settings.similarity_threshold
        if score_threshold is None
        else score_threshold
    )
    hybrid_on = settings.enable_hybrid and bool(settings.sparse_index_url)
    candidate_k = max(top_k, top_k * 4)
    limit = candidate_k if hybrid_on else top_k

    hits, dense_ids = hybrid_retrieve_with_dense_ids(
        config, query, limit=limit, score_threshold=threshold
    )
    chunks: list[RetrievedChunk] = []
    for hit in hits:
        cid = str(hit.get("id", ""))
        retrieval = "dense" if not hybrid_on or cid in dense_ids else "sparse"
        chunk = _hit_to_chunk(hit, retrieval)
        if chunk is not None:
            chunks.append(chunk)
    return chunks


def rerank_chunks(
    settings: RetrieveSettings,
    query: str,
    chunks: list[RetrievedChunk],
    *,
    top_k: int,
) -> list[RetrievedChunk]:
    if not chunks:
        return []
    config = _to_retrieve_config(settings)
    pairs = [
        {"query": query, "document": c.text[: settings.embed_max_chars]}
        for c in chunks
    ]
    indices = rerank_pairs(config, pairs, top_k=top_k)
    return [chunks[i] for i in indices if 0 <= i < len(chunks)][:top_k]


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

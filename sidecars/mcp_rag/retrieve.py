"""Hybrid retrieval for MCP tools (embed + Qdrant + sparse + rerank + MemGraph facts)."""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import httpx
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_root = str(_REPO_ROOT)
if _root not in sys.path:
    sys.path.insert(0, _root)

from rag_proxy.chunk_text import extract_chunk_text
from rag_proxy.clients.retrieve_sync import (
    RetrieveConfig,
    dense_search,
    embed_query,
    hybrid_retrieve_with_dense_ids,
    rerank_pairs,
    sparse_search,
)
from rag_proxy.memgraphrag.cache import get_memory_index

log = logging.getLogger("mcp-rag-context")

_VALID_MODES = frozenset({"passages", "hybrid", "dense", "sparse", "facts"})


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


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
    dense_backend: str = "qdrant"
    turbovec_url: str = ""
    memgraphrag_db_path: str = ""

    @classmethod
    def from_env(cls) -> RetrieveSettings:
        return cls(
            embed_url=os.getenv("EMBED_URL", "http://127.0.0.1:8089"),
            qdrant_url=os.getenv("QDRANT_URL", "http://127.0.0.1:6333"),
            qdrant_collection=os.getenv(
                "QDRANT_COLLECTION", "nomad_knowledge_base"
            ),
            sparse_index_url=os.getenv("SPARSE_INDEX_URL", "").strip(),
            reranker_url=os.getenv("RERANKER_URL", "http://127.0.0.1:8095"),
            hybrid_dense_weight=float(os.getenv("HYBRID_DENSE_WEIGHT", "0.7")),
            similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.65")),
            enable_hybrid=_env_bool("ENABLE_HYBRID_RETRIEVAL", False),
            enable_rerank=_env_bool("ENABLE_RERANKER", False),
            embed_max_chars=int(os.getenv("EMBED_MAX_CHARS", "2000")),
            user_agent=os.getenv("MCP_RAG_USER_AGENT", "mcp-rag-context/1.0"),
            dense_backend=os.getenv("DENSE_BACKEND", "qdrant").strip().lower(),
            turbovec_url=os.getenv("TURBOVEC_URL", "").strip(),
            memgraphrag_db_path=os.getenv(
                "MEMGRAPHRAG_DB_PATH", "/var/lib/rag_proxy/memgraphrag.sqlite"
            ).strip(),
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
        dense_backend=settings.dense_backend,
        turbovec_url=settings.turbovec_url,
    )


def _headers(settings: RetrieveSettings) -> dict[str, str]:
    return {"User-Agent": settings.user_agent}


def _normalize_mode(mode: str) -> str:
    cleaned = (mode or "passages").strip().lower()
    if cleaned not in _VALID_MODES:
        return "passages"
    if cleaned == "hybrid":
        return "passages"
    return cleaned


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
    mode: str = "passages",
) -> list[RetrievedChunk]:
    """Retrieve passages. score_threshold applies to the dense leg only."""
    config = _to_retrieve_config(settings)
    threshold = (
        settings.similarity_threshold
        if score_threshold is None
        else score_threshold
    )
    mode_norm = _normalize_mode(mode)

    if mode_norm == "sparse":
        if not settings.sparse_index_url:
            return []
        hits = sparse_search(config, query, limit=top_k)
        chunks: list[RetrievedChunk] = []
        for hit in hits:
            chunk = _hit_to_chunk(hit, "sparse")
            if chunk is not None:
                chunks.append(chunk)
        return chunks

    if mode_norm == "dense":
        vector = embed_query(config, query)
        if vector is None:
            return []
        hits = dense_search(
            config, vector, limit=top_k, score_threshold=threshold
        )
        chunks = []
        for hit in hits:
            chunk = _hit_to_chunk(hit, "dense")
            if chunk is not None:
                chunks.append(chunk)
        return chunks

    # passages / hybrid: dense-only unless hybrid enabled and sparse URL set
    hybrid_on = settings.enable_hybrid and bool(settings.sparse_index_url)
    candidate_k = max(top_k, top_k * 4)
    limit = candidate_k if hybrid_on else top_k

    retrieve_cfg = config if hybrid_on else replace(config, enable_hybrid=False)
    hits, dense_ids = hybrid_retrieve_with_dense_ids(
        retrieve_cfg, query, limit=limit, score_threshold=threshold
    )

    chunks = []
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


def search_memgraph_facts(
    query: str,
    *,
    top_k: int = 5,
    settings: RetrieveSettings | None = None,
) -> list[RetrievedChunk]:
    """Score MemGraphRAG facts against the query. Fail-open to empty if unavailable."""
    cfg = settings or RetrieveSettings.from_env()
    limit = max(1, min(top_k, 20))
    db_path = cfg.memgraphrag_db_path
    if not db_path or not Path(db_path).is_file():
        log.info("MemGraphRAG DB missing or unset; facts mode returns empty")
        return []

    try:
        index = get_memory_index(db_path)
    except Exception as exc:
        log.warning("MemGraphRAG index load failed: %s", exc)
        return []

    if index.fact_embeddings.size == 0:
        return []

    config = _to_retrieve_config(cfg)
    vector = embed_query(config, query)
    if vector is None:
        return []

    query_emb = np.array(vector, dtype=np.float32)
    query_norm = float(np.linalg.norm(query_emb))
    if query_norm == 0:
        return []
    query_unit = query_emb / query_norm
    scores = index.fact_embeddings @ query_unit
    order = np.argsort(-scores)

    chunks: list[RetrievedChunk] = []
    for i in order:
        if len(chunks) >= limit:
            break
        score = float(scores[i])
        if score <= 0:
            continue
        fact_idx = int(index.fact_indices[i])
        fact = index.memory.facts.get(fact_idx)
        if fact is None:
            continue
        passages = index.memory.get_passages_for_fact(fact_idx)
        excerpt = passages[0].content[: cfg.embed_max_chars] if passages else ""
        body = fact.triple_str
        if excerpt:
            body = f"{fact.triple_str}\n\n{excerpt}"
        chunks.append(
            RetrievedChunk(
                chunk_id=f"fact-{fact_idx}",
                text=body,
                score=score,
                source="memgraphrag",
                title=fact.triple_str,
                retrieval="facts",
            )
        )
    return chunks


def search_knowledge_base(
    query: str,
    *,
    top_k: int = 5,
    score_threshold: float | None = None,
    mode: str = "passages",
    settings: RetrieveSettings | None = None,
) -> list[RetrievedChunk]:
    cfg = settings or RetrieveSettings.from_env()
    limit = max(1, min(top_k, 20))
    mode_norm = _normalize_mode(mode)

    if mode_norm == "facts":
        return search_memgraph_facts(query, top_k=limit, settings=cfg)

    candidates = hybrid_retrieve(
        cfg,
        query,
        top_k=limit,
        score_threshold=score_threshold,
        mode=mode_norm,
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


def _probe_embed(settings: RetrieveSettings) -> str:
    base = settings.embed_url.rstrip("/")
    try:
        with httpx.Client(timeout=5.0, headers=_headers(settings)) as client:
            for path in ("/health", "/v1/models"):
                response = client.get(f"{base}{path}")
                if response.status_code < 500:
                    return "ok" if response.status_code < 400 else f"http_{response.status_code}"
    except httpx.HTTPError:
        return "error"
    return "unknown"


def fetch_index_status(settings: RetrieveSettings | None = None) -> dict[str, Any]:
    cfg = settings or RetrieveSettings.from_env()
    status: dict[str, Any] = {
        "collection": cfg.qdrant_collection,
        "qdrant_points": 0,
        "qdrant_status": "unknown",
        "sparse_docs": 0,
        "sparse_status": "unknown",
        "sparse_truncated": False,
        "sparse_max_points": 0,
        "embed_url": cfg.embed_url,
        "embed_status": "unknown",
        "dense_backend": cfg.dense_backend,
        "enable_hybrid": cfg.enable_hybrid,
        "enable_rerank": cfg.enable_rerank,
        "memgraphrag_db_path": cfg.memgraphrag_db_path,
        "memgraphrag_available": False,
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
                status["qdrant_status"] = "ok"
            else:
                status["qdrant_status"] = f"http_{response.status_code}"
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
                    status["sparse_truncated"] = bool(body.get("truncated", False))
                    status["sparse_max_points"] = int(body.get("max_points", 0) or 0)
                else:
                    status["sparse_status"] = f"http_{response.status_code}"
        except httpx.HTTPError:
            status["sparse_status"] = "error"
    else:
        status["sparse_status"] = "disabled"

    status["embed_status"] = _probe_embed(cfg)

    db_path = cfg.memgraphrag_db_path
    if db_path and Path(db_path).is_file():
        status["memgraphrag_available"] = True

    return status

"""Read-only runtime snapshot for GET /debug."""

from __future__ import annotations

from rag_proxy import upstream_client as uc
from rag_proxy.clients.retrieval_async import embed_cache_stats
from rag_proxy.config import settings


def build_debug_snapshot() -> dict:
    return {
        "cognitive_pipeline": settings.enable_cognitive_pipeline,
        "cognitive_latency_budget_ms": settings.cognitive_latency_budget_ms,
        "stage_exec_timeout_ms": settings.stage_exec_timeout_ms,
        "enable_embed_cache": settings.enable_embed_cache,
        "enable_tokenizer_estimate": settings.enable_tokenizer_estimate,
        "enable_hybrid_retrieval": settings.enable_hybrid_retrieval,
        "enable_reranker": settings.enable_reranker,
        "enable_memgraphrag": settings.enable_memgraphrag,
        "top_k": settings.top_k,
        "similarity_threshold": settings.similarity_threshold,
        "upstream": {
            "max_connections": settings.upstream_max_connections,
            "max_keepalive": settings.upstream_max_keepalive,
            "keepalive_expiry_sec": settings.upstream_keepalive_expiry_sec,
            "active_streams": uc.upstream_active_stream_count(),
        },
        "embed_cache": embed_cache_stats(),
        "urls": {
            "embed": settings.embed_url,
            "qdrant": settings.qdrant_url,
            "qdrant_collection": settings.qdrant_collection,
            "sparse_index": settings.sparse_index_url or None,
            "reranker": settings.reranker_url if settings.enable_reranker else None,
        },
    }

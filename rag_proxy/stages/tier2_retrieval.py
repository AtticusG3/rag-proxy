"""Hybrid retrieval stage."""

from __future__ import annotations

from rag_proxy.registry.models import ModelRegistry
from rag_proxy.clients.qdrant import hybrid_search
from rag_proxy.config import settings
from rag_proxy.context import RequestContext, RetrievalDecision


async def run_retrieval(ctx: RequestContext, _registry: ModelRegistry) -> None:
    if ctx.retrieval == RetrievalDecision.SKIP:
        return

    query = ctx.effective_query()
    if not query:
        return

    limit = ctx.top_k_for_retrieval(
        settings.retrieval_candidate_k if ctx.retrieval == RetrievalDecision.FULL else settings.top_k
    )
    if ctx.retrieval == RetrievalDecision.FULL and settings.enable_reranker:
        limit = max(limit, settings.retrieval_candidate_k)

    hits = await hybrid_search(
        query,
        limit=limit,
        no_cache=ctx.no_cache,
    )
    ctx.hits = hits
    ctx.stage_trace.append(f"retrieve:{len(ctx.hits)}")

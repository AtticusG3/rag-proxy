"""Optional cross-encoder reranking via sidecar."""

from __future__ import annotations

import logging

from rag_proxy.config import settings
from rag_proxy.context import ChunkHit, RequestContext
from rag_proxy.sidecar_client import get_reranker_client

log = logging.getLogger("rag-proxy")


async def run_rerank(ctx: RequestContext) -> None:
    if not settings.enable_reranker or not ctx.hits:
        return

    query = ctx.effective_query() or ""
    pairs = [{"query": query, "document": h.text} for h in ctx.hits if h.text]
    if not pairs:
        return

    try:
        client = get_reranker_client()
        r = await client.post(
            f"{settings.reranker_url.rstrip('/')}/rerank",
            json={"pairs": pairs, "top_k": settings.rerank_top_k},
        )
        r.raise_for_status()
        order = r.json().get("indices", [])
        if order:
            reordered = [ctx.hits[i] for i in order if 0 <= i < len(ctx.hits)]
            ctx.hits = reordered[: settings.rerank_top_k]
            ctx.stage_trace.append("rerank:ok")
            return
    except Exception as e:
        log.warning(f"Rerank failed: {e}")
        ctx.errors.append(f"rerank:{e}")

    ctx.stage_trace.append("rerank:fallback")

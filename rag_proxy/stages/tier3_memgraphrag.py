"""MemGraphRAG pipeline stage.

Uses the three-layer memory (schema/fact/passage) for graph-based retrieval:
  1. Score facts against query via embedding similarity
  2. Rerank facts with cross-encoder
  3. Personalized PageRank on the fact graph
  4. Aggregate passage scores from PPR

When MemGraphRAG finds no scored facts or passages, the stage adds no hits
(fail-open: prior retrieval hits are preserved).
"""

from __future__ import annotations

import logging

from rag_proxy.config import settings
from rag_proxy.context import ChunkHit, RequestContext
from rag_proxy.memgraphrag.cache import get_memory_index
from rag_proxy.memgraphrag.retrieval import MemGraphRetriever

log = logging.getLogger("rag-proxy.stage.memgraphrag")


async def run_memgraphrag(ctx: RequestContext) -> None:
    """Run MemGraphRAG retrieval and append results to ctx.hits."""
    if not ctx.query_text:
        return

    db_path = settings.memgraphrag_db_path
    if not db_path:
        log.debug("MEMGRAPHRAG_DB_PATH not set, skipping")
        return

    try:
        index = get_memory_index(db_path)
        if not index.memory.facts:
            log.info("MemGraphRAG memory is empty, skipping")
            return

        retriever = MemGraphRetriever(
            index=index,
            top_k=settings.top_k,
            fact_top_k=settings.memgraphrag_fact_top_k,
            ppr_damping=settings.memgraphrag_ppr_damping,
            ppr_iterations=settings.memgraphrag_ppr_iterations,
            passage_node_weight=settings.memgraphrag_passage_node_weight,
        )

        hits = await retriever.retrieve(ctx.query_text)
        if hits:
            ctx.hits.extend(hits)
            ctx.stage_trace.append(f"memgraphrag:{len(hits)}")
        else:
            log.info("MemGraphRAG returned no hits")
    except Exception as e:
        log.warning("MemGraphRAG retrieval failed: %s", e)
        ctx.errors.append(f"memgraphrag:{e}")

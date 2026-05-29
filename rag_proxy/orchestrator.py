"""Cognitive pipeline orchestrator."""

from __future__ import annotations

import logging
import time

from rag_proxy.clients.bundle import ClientBundle
from rag_proxy.config import settings
from rag_proxy.context import PipelineTier, RequestContext, RetrievalDecision
from rag_proxy.legacy_rag import extract_query_text, legacy_augment_messages
from rag_proxy.observability import log_pipeline_summary, new_trace_id
from rag_proxy.stages import routing as routing_stage
from rag_proxy.stages import tier0_heuristics, tier1_gating, tier1_intent
from rag_proxy.stages import tier2_context, tier2_rerank, tier2_retrieval, tier2_rewrite
from rag_proxy.stages import tier3_graph, tier3_memory, tier3_tools

log = logging.getLogger("rag-proxy")

_clients = ClientBundle()


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def _budget_remaining(ctx: RequestContext) -> float:
    if ctx.cognitive_start_ms <= 0:
        return float(settings.cognitive_latency_budget_ms)
    spent = _elapsed_ms(ctx.cognitive_start_ms)
    return max(0.0, settings.cognitive_latency_budget_ms - spent)


def build_request_context_from_http(
    data: dict,
    headers: dict[str, str] | None = None,
) -> RequestContext:
    messages = data.get("messages", [])
    hdr = {k.lower(): v for k, v in (headers or {}).items()}
    rag_header = hdr.get("x-rag-mode")
    no_cache = hdr.get("x-no-cache", "").lower() in ("1", "true", "yes")
    conv = hdr.get("x-conversation-id")

    return RequestContext(
        messages=list(messages),
        data=data,
        query_text=extract_query_text(messages),
        requested_model=data.get("model"),
        stream=bool(data.get("stream")),
        rag_mode_header=rag_header,
        no_cache=no_cache,
        conversation_id=conv,
    )


async def run_cognitive_pipeline(ctx: RequestContext) -> None:
    ctx.trace_id = ctx.trace_id or new_trace_id()
    ctx.cognitive_start_ms = time.perf_counter()

    t0 = time.perf_counter()
    await tier0_heuristics.run_tier0(ctx)
    ctx.latency_ms["tier0"] = _elapsed_ms(t0)

    if ctx.retrieval != RetrievalDecision.SKIP:
        ctx.tier = PipelineTier.TIER1_LIGHT

    t1 = time.perf_counter()
    await tier1_intent.run_intent(ctx, _clients)
    ctx.latency_ms["intent"] = _elapsed_ms(t1)

    t2 = time.perf_counter()
    await tier1_gating.run_gating(ctx)
    ctx.latency_ms["gating"] = _elapsed_ms(t2)

    if _budget_remaining(ctx) > 0:
        t3 = time.perf_counter()
        await routing_stage.run_routing(ctx, _clients)
        ctx.latency_ms["routing"] = _elapsed_ms(t3)

    if ctx.retrieval != RetrievalDecision.SKIP and _budget_remaining(ctx) > 20:
        t4 = time.perf_counter()
        await tier2_rewrite.run_rewrite(ctx)
        ctx.latency_ms["rewrite"] = _elapsed_ms(t4)

        if _budget_remaining(ctx) > 50:
            ctx.tier = PipelineTier.TIER2_RETRIEVAL
            t5 = time.perf_counter()
            await tier2_retrieval.run_retrieval(ctx, _clients)
            ctx.latency_ms["retrieve"] = _elapsed_ms(t5)

            if settings.enable_reranker and _budget_remaining(ctx) > settings.rerank_timeout_ms:
                t6 = time.perf_counter()
                await tier2_rerank.run_rerank(ctx)
                ctx.latency_ms["rerank"] = _elapsed_ms(t6)

    if settings.enable_graph_lookup and _budget_remaining(ctx) > 100:
        ctx.tier = PipelineTier.TIER3_HEAVY
        tg = time.perf_counter()
        await tier3_graph.run_graph(ctx)
        ctx.latency_ms["graph"] = _elapsed_ms(tg)

    if settings.enable_tools and _budget_remaining(ctx) > settings.tool_budget_ms:
        tt = time.perf_counter()
        await tier3_tools.run_tools(ctx)
        ctx.latency_ms["tools"] = _elapsed_ms(tt)

    if settings.enable_rolling_memory:
        tm = time.perf_counter()
        await tier3_memory.run_memory(ctx)
        ctx.latency_ms["memory"] = _elapsed_ms(tm)

    if ctx.chunk_texts or ctx.hits:
        tc = time.perf_counter()
        await tier2_context.run_context_assembly(ctx, _clients)
        ctx.latency_ms["context"] = _elapsed_ms(tc)

    ctx.latency_ms["total_cognitive"] = _elapsed_ms(ctx.cognitive_start_ms)
    log_pipeline_summary(ctx)


async def augment_chat_payload(
    data: dict,
    headers: dict[str, str] | None = None,
) -> dict:
    """Augment messages in chat payload; fail-open."""
    messages = data.get("messages", [])
    if not settings.enable_cognitive_pipeline:
        new_messages, meta = await legacy_augment_messages(messages)
        if meta.get("chunks"):
            data = {**data, "messages": new_messages}
            log.info(
                f"RAG: injected {meta['chunks']} chunk(s) "
                f"(scores: {meta['scores']}) | query: {str(meta.get('query', ''))[:80]!r}"
            )
        elif meta.get("query"):
            log.debug(f"RAG: no chunks above threshold={settings.similarity_threshold}")
        return data

    ctx = build_request_context_from_http(data, headers)
    try:
        await _clients.model_registry.refresh()
        await run_cognitive_pipeline(ctx)
        data = {**data, "messages": ctx.messages}
        if ctx.selected_model and settings.model_routing_mode == "force":
            data["model"] = ctx.selected_model
        if ctx.chunk_texts:
            log.info(
                f"RAG: injected {len(ctx.chunk_texts)} chunk(s) "
                f"(scores: {[round(h.score, 3) for h in ctx.hits]}) "
                f"| trace={ctx.trace_id} | query: {(ctx.effective_query() or '')[:80]!r}"
            )
        elif ctx.query_text and ctx.retrieval != RetrievalDecision.SKIP:
            log.debug(f"RAG: no chunks (retrieval={ctx.retrieval.value})")
        elif ctx.retrieval == RetrievalDecision.SKIP:
            log.debug(f"RAG: skipped retrieval (tier={ctx.tier.value})")
    except Exception as e:
        log.warning(f"RAG augmentation error (passing through unmodified): {e}")

    return data

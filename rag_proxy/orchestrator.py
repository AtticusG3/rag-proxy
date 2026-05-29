"""Cognitive pipeline orchestrator."""

from __future__ import annotations

import logging
import time

from rag_proxy.clients.bundle import ClientBundle
from rag_proxy.config import settings
from rag_proxy.context import RequestContext, RetrievalDecision
from rag_proxy.legacy_rag import extract_query_text, legacy_augment_messages
from rag_proxy.observability import log_pipeline_summary, new_trace_id, record_rag_outcome
from rag_proxy.pipeline_stages import build_pipeline_stages

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

    stages = build_pipeline_stages()
    try:
        for stage in stages:
            if not stage.enabled():
                continue
            if not stage.should_run(ctx):
                continue
            if _budget_remaining(ctx) < stage.min_budget_ms:
                continue
            t0 = time.perf_counter()
            await stage.run(ctx, _clients)
            ctx.latency_ms[stage.name] = _elapsed_ms(t0)
    finally:
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
            record_rag_outcome(int(meta["chunks"]))
            log.info(
                f"RAG: injected {meta['chunks']} chunk(s) "
                f"(scores: {meta['scores']}) | query: {str(meta.get('query', ''))[:80]!r}"
            )
        elif meta.get("query"):
            record_rag_outcome(0)
            log.debug(f"RAG: no chunks above threshold={settings.similarity_threshold}")
        return data

    ctx = build_request_context_from_http(data, headers)
    try:
        if settings.enable_model_routing:
            await _clients.model_registry.refresh()
        await run_cognitive_pipeline(ctx)
        data = {**data, "messages": ctx.messages}
        if ctx.selected_model and settings.model_routing_mode == "force":
            data["model"] = ctx.selected_model
        if ctx.hits:
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

"""Structured logging and optional metrics counters."""

from __future__ import annotations

import json
import logging
import threading
import uuid

from rag_proxy.config import settings
from rag_proxy.context import RequestContext, RetrievalDecision

log = logging.getLogger("rag-proxy")

_metrics_lock = threading.Lock()
_requests_total = 0
_chunks_injected_total = 0


def new_trace_id() -> str:
    return uuid.uuid4().hex[:12]


def metrics_enabled() -> bool:
    """ENABLE_METRICS enables GET /metrics on the proxy app."""
    return settings.enable_metrics


def record_rag_outcome(chunks_injected: int) -> None:
    global _requests_total, _chunks_injected_total
    with _metrics_lock:
        _requests_total += 1
        if chunks_injected > 0:
            _chunks_injected_total += chunks_injected


def render_metrics_text() -> str:
    with _metrics_lock:
        reqs = _requests_total
        chunks = _chunks_injected_total
    return (
        f"rag_requests_total {reqs}\n"
        f"rag_chunks_injected_total {chunks}\n"
    )


def log_pipeline_summary(ctx: RequestContext) -> None:
    record_rag_outcome(len(ctx.chunk_texts))

    if not settings.enable_request_trace:
        return
    summary = {
        "trace_id": ctx.trace_id,
        "tier": ctx.tier.value,
        "intent": ctx.intent.value,
        "intent_confidence": round(ctx.intent_confidence, 3),
        "retrieval": ctx.retrieval.value,
        "chunks_injected": len(ctx.chunk_texts),
        "injected_tokens_est": ctx.injected_tokens_est,
        "latency_ms": {k: round(v, 2) for k, v in ctx.latency_ms.items()},
        "scores": [round(h.score, 3) for h in ctx.hits],
        "stage_trace": ctx.stage_trace,
        "cache_hits": ctx.cache_hits,
        "errors": ctx.errors,
        "model_requested": ctx.requested_model,
        "model_routed": ctx.selected_model,
        "gating_would_skip": ctx.gating_would_skip,
    }
    if settings.enable_json_logs:
        log.info(json.dumps(summary, ensure_ascii=False))
    else:
        log.info(
            f"trace={ctx.trace_id} tier={ctx.tier.value} intent={ctx.intent.value} "
            f"retrieval={ctx.retrieval.value} chunks={len(ctx.chunk_texts)} "
            f"latency_ms={summary['latency_ms']} stages={','.join(ctx.stage_trace)}"
        )


def log_rag_request(ctx: RequestContext) -> None:
    """Human-readable RAG outcome log shared by legacy and cognitive paths."""
    if ctx.chunk_texts:
        log.info(
            f"RAG: injected {len(ctx.chunk_texts)} chunk(s) "
            f"(scores: {[round(h.score, 3) for h in ctx.hits]}) "
            f"| trace={ctx.trace_id} | query: {(ctx.effective_query() or ctx.query_text or '')[:80]!r}"
        )
    elif ctx.query_text and ctx.retrieval != RetrievalDecision.SKIP:
        log.debug(
            f"RAG: no chunks (retrieval={ctx.retrieval.value}) "
            f"| threshold={settings.similarity_threshold}"
        )
    elif ctx.retrieval == RetrievalDecision.SKIP:
        log.debug(f"RAG: skipped retrieval (tier={ctx.tier.value})")

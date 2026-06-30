"""Structured logging and optional Prometheus metrics."""

from __future__ import annotations

import json
import logging
import uuid

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from rag_proxy.config import settings
from rag_proxy.context import RequestContext, RetrievalDecision

log = logging.getLogger("rag-proxy")

RAG_REQUESTS = Counter(
    "rag_requests_total",
    "Chat requests that completed the RAG pipeline path",
    ["outcome"],
)
RAG_CHUNKS_INJECTED = Counter(
    "rag_chunks_injected_total",
    "Sum of chunks injected across requests",
)
RAG_AUGMENT_ERRORS = Counter(
    "rag_augment_errors_total",
    "RAG augmentation failures (request forwarded unmodified)",
)
RAG_EMBED_CACHE_HITS = Counter(
    "rag_embed_cache_hits_total",
    "Embed cache hits",
)
RAG_EMBED_CACHE_MISSES = Counter(
    "rag_embed_cache_misses_total",
    "Embed cache misses",
)
RAG_STAGE_LATENCY = Histogram(
    "rag_stage_latency_seconds",
    "Per-stage pipeline latency",
    ["stage"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)
RAG_AUGMENT_DURATION = Histogram(
    "rag_augment_duration_seconds",
    "Wall time for RAG augmentation (pipeline only)",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)
PROXY_REQUEST_DURATION = Histogram(
    "proxy_request_duration_seconds",
    "Full proxy handler wall time (includes upstream for buffered responses)",
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
)
UPSTREAM_ACTIVE_STREAMS = Gauge(
    "upstream_active_streams",
    "Active upstream SSE streams registered for janitor",
)


def new_trace_id() -> str:
    return uuid.uuid4().hex[:12]


def metrics_enabled() -> bool:
    """ENABLE_METRICS enables GET /metrics on the proxy app."""
    return settings.enable_metrics


def metrics_content_type() -> str:
    return CONTENT_TYPE_LATEST


def _rag_outcome(ctx: RequestContext) -> str:
    if ctx.retrieval == RetrievalDecision.SKIP:
        return "skip"
    if ctx.chunk_texts:
        return "hit"
    return "miss"


def record_rag_outcome(ctx: RequestContext) -> None:
    outcome = _rag_outcome(ctx)
    RAG_REQUESTS.labels(outcome=outcome).inc()
    chunks = len(ctx.chunk_texts)
    if chunks > 0:
        RAG_CHUNKS_INJECTED.inc(chunks)


def record_rag_outcome_legacy(chunks_injected: int, outcome: str = "miss") -> None:
    """Test helper and backward-compatible counter bump."""
    RAG_REQUESTS.labels(outcome=outcome).inc()
    if chunks_injected > 0:
        RAG_CHUNKS_INJECTED.inc(chunks_injected)


def record_augment_error() -> None:
    RAG_AUGMENT_ERRORS.inc()


def record_embed_cache_hit() -> None:
    RAG_EMBED_CACHE_HITS.inc()


def record_embed_cache_miss() -> None:
    RAG_EMBED_CACHE_MISSES.inc()


def observe_stage_latency(stage: str, seconds: float) -> None:
    RAG_STAGE_LATENCY.labels(stage=stage).observe(seconds)


def observe_rag_augment_duration(seconds: float) -> None:
    RAG_AUGMENT_DURATION.observe(seconds)


def observe_proxy_request_duration(seconds: float) -> None:
    PROXY_REQUEST_DURATION.observe(seconds)


def set_upstream_active_streams(count: int) -> None:
    UPSTREAM_ACTIVE_STREAMS.set(count)


def render_metrics_text() -> str:
    return generate_latest().decode("utf-8")


def log_pipeline_summary(ctx: RequestContext) -> None:
    record_rag_outcome(ctx)

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

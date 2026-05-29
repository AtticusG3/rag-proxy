"""Structured logging and optional metrics."""

from __future__ import annotations

import json
import logging
import uuid

from rag_proxy.config import settings
from rag_proxy.context import RequestContext

log = logging.getLogger("rag-proxy")


def new_trace_id() -> str:
    return uuid.uuid4().hex[:12]


def log_pipeline_summary(ctx: RequestContext) -> None:
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

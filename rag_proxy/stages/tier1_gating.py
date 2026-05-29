"""Retrieval gating: skip embed/Qdrant when not needed."""

from __future__ import annotations

from rag_proxy.config import settings
from rag_proxy.context import IntentLabel, RequestContext, RetrievalDecision


def _gating_decision(ctx: RequestContext) -> RetrievalDecision:
    if ctx.retrieval == RetrievalDecision.SKIP:
        return RetrievalDecision.SKIP

    query = ctx.query_text or ""
    intent = ctx.intent

    if intent in (
        IntentLabel.INFRA_DEBUG,
        IntentLabel.TROUBLESHOOTING,
        IntentLabel.LOG_ANALYSIS,
        IntentLabel.RETRIEVAL_HEAVY,
    ):
        return RetrievalDecision.FULL

    if intent in (IntentLabel.RESEARCH, IntentLabel.PLANNING, IntentLabel.CODE_REVIEW):
        return RetrievalDecision.FULL

    if intent in (IntentLabel.SIMPLE_CHAT, IntentLabel.CREATIVE) and ctx.intent_confidence >= 0.6:
        return RetrievalDecision.SKIP

    if intent == IntentLabel.CODE_GENERATION and len(query) < 120:
        return RetrievalDecision.LIGHT

    if len(query) < 40 and ctx.intent_confidence < 0.5:
        return RetrievalDecision.SKIP

    return RetrievalDecision.FULL


async def run_gating(ctx: RequestContext) -> None:
    if not settings.enable_retrieval_gating:
        return

    decision = _gating_decision(ctx)
    ctx.gating_would_skip = decision == RetrievalDecision.SKIP

    if settings.gating_log_only:
        ctx.stage_trace.append(f"gating:log_only:would_{decision.value}")
        return

    ctx.retrieval = decision
    ctx.stage_trace.append(f"gating:{decision.value}")

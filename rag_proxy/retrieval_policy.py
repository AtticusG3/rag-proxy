"""Consolidated retrieval skip/light/full policy."""

from __future__ import annotations

import re
from typing import Literal

from rag_proxy.config import settings
from rag_proxy.context import IntentLabel, PipelineTier, RequestContext, RetrievalDecision

# Whole-utterance / prefix greetings (no '?'); KB questions must still retrieve.
_GREETING = re.compile(
    r"^(hi|hello|hey|howdy|thanks|thank you|good morning|good afternoon|"
    r"good evening|good night)\b",
    re.I,
)
_ACK = re.compile(
    r"^(ok(?:ay)?|k{1,2}|sure|yep|yup|cool|great|perfect|got\s+it|noted|"
    r"understood|cheers|np|no\s+problem|thx|ty)[.!]*$",
    re.I,
)
INFRA_SIGNAL = re.compile(
    r"(/[\w.-]+|\\\\[\w.-]+|\b\d{1,3}(?:\.\d{1,3}){3}\b|"
    r"\b(error|exception|failed|traceback|kubectl|docker|systemctl|"
    r"qdrant|llama|nomad|compose)\b|"
    r"\b[A-Z]{2,}\d+\b|v?\d+\.\d+\.\d+)",
    re.I,
)

PolicyPhase = Literal["tier0", "gating"]


def should_bypass_heuristics(ctx: RequestContext) -> bool:
    """True when tier0 should skip retrieval (greetings/acks only, not FAQs)."""
    if ctx.rag_mode_header == "force":
        return False
    if ctx.rag_mode_header == "off":
        return True

    q = (ctx.query_text or "").strip()
    if not q:
        return True

    # Knowledge / troubleshooting questions always retrieve at tier0.
    if "?" in q:
        return False

    if INFRA_SIGNAL.search(q):
        return False

    if len(q) > settings.tier0_max_chars:
        return False

    if _GREETING.match(q) or _ACK.match(q):
        return True

    return False


def gating_decision(ctx: RequestContext) -> RetrievalDecision:
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


def apply_retrieval_policy(ctx: RequestContext, phase: PolicyPhase) -> list[str]:
    """Ordered retrieval policy for tier0 headers/heuristics or intent gating."""
    if phase == "tier0":
        if ctx.rag_mode_header == "force":
            return ["tier0:force_retrieve"]

        if ctx.rag_mode_header == "off":
            ctx.tier = PipelineTier.TIER0_BYPASS
            ctx.retrieval = RetrievalDecision.SKIP
            return ["tier0:header_off"]

        if settings.enable_tier0_heuristics and should_bypass_heuristics(ctx):
            ctx.tier = PipelineTier.TIER0_BYPASS
            ctx.retrieval = RetrievalDecision.SKIP
            return ["tier0:bypass"]

        return []

    if ctx.rag_mode_header == "force":
        ctx.retrieval = RetrievalDecision.FULL
        ctx.gating_would_skip = False
        return ["gating:forced_full"]

    if not settings.enable_retrieval_gating:
        return []

    decision = gating_decision(ctx)
    ctx.gating_would_skip = decision == RetrievalDecision.SKIP

    if settings.gating_log_only:
        return [f"gating:log_only:would_{decision.value}"]

    ctx.retrieval = decision
    return [f"gating:{decision.value}"]

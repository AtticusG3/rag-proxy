"""Tier 0: heuristic fast path (no embed/Qdrant)."""

from __future__ import annotations

import re

from rag_proxy.config import settings
from rag_proxy.context import PipelineTier, RequestContext, RetrievalDecision

_GREETING = re.compile(
    r"^(hi|hello|hey|thanks|thank you|good morning|good evening)\b",
    re.I,
)
_INFRA_SIGNAL = re.compile(
    r"(/[\w.-]+|\\\\[\w.-]+|\b\d{1,3}(?:\.\d{1,3}){3}\b|"
    r"\b(error|exception|failed|traceback|kubectl|docker|systemctl|"
    r"qdrant|llama|nomad|compose)\b|"
    r"\b[A-Z]{2,}\d+\b|v?\d+\.\d+\.\d+)",
    re.I,
)
_SIMPLE_FAQ = re.compile(
    r"^what is [a-z][a-z0-9 -]{0,40}\??$",
    re.I,
)


def should_bypass_heuristics(ctx: RequestContext) -> bool:
    if ctx.rag_mode_header == "force":
        return False
    if ctx.rag_mode_header == "off":
        return True

    query = ctx.query_text
    if not query:
        return True

    if _INFRA_SIGNAL.search(query):
        return False

    q = query.strip()
    if len(q) <= settings.tier0_max_chars and _GREETING.match(q):
        return True
    if len(q) <= settings.tier0_max_chars and _SIMPLE_FAQ.match(q):
        return True
    if len(q) <= 40 and "?" in q and not _INFRA_SIGNAL.search(q):
        words = q.split()
        if len(words) <= 8:
            return True

    return False


async def run_tier0(ctx: RequestContext) -> None:
    if not settings.enable_tier0_heuristics:
        return
    if ctx.rag_mode_header == "force":
        ctx.stage_trace.append("tier0:force_retrieve")
        return
    if ctx.rag_mode_header == "off":
        ctx.tier = PipelineTier.TIER0_BYPASS
        ctx.retrieval = RetrievalDecision.SKIP
        ctx.stage_trace.append("tier0:header_off")
        return

    if should_bypass_heuristics(ctx):
        ctx.tier = PipelineTier.TIER0_BYPASS
        ctx.retrieval = RetrievalDecision.SKIP
        ctx.stage_trace.append("tier0:bypass")

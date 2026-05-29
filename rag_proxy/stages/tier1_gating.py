"""Retrieval gating: skip embed/Qdrant when not needed."""

from __future__ import annotations

from rag_proxy.config import settings
from rag_proxy.context import RequestContext
from rag_proxy.retrieval_policy import apply_late_policy, gating_decision

# Re-export for tests
__all__ = ["gating_decision", "run_gating"]


async def run_gating(ctx: RequestContext) -> None:
    if not settings.enable_retrieval_gating:
        return

    ctx.stage_trace.extend(apply_late_policy(ctx))

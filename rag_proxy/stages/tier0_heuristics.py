"""Tier 0: heuristic fast path (no embed/Qdrant)."""

from __future__ import annotations

from rag_proxy.config import settings
from rag_proxy.context import RequestContext
from rag_proxy.retrieval_policy import apply_retrieval_policy, should_bypass_heuristics

# Re-export for tests
__all__ = ["should_bypass_heuristics", "run_tier0"]


async def run_tier0(ctx: RequestContext) -> None:
    # Header overrides are operator controls and must apply even
    # when heuristic bypass is disabled.
    if ctx.rag_mode_header in ("off", "force"):
        ctx.stage_trace.extend(apply_retrieval_policy(ctx, "tier0"))
        return

    if not settings.enable_tier0_heuristics:
        return

    ctx.stage_trace.extend(apply_retrieval_policy(ctx, "tier0"))

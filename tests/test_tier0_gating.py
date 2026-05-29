"""Unit tests for tier0 heuristics and retrieval gating."""

import asyncio

from rag_proxy.config import settings
from rag_proxy.context import IntentLabel, RequestContext, RetrievalDecision
from rag_proxy.stages import tier0_heuristics
from rag_proxy.stages.tier1_gating import _gating_decision


def test_tier0_bypasses_short_greeting():
    ctx = RequestContext(query_text="hello there")
    assert tier0_heuristics.should_bypass_heuristics(ctx)


def test_tier0_does_not_bypass_infra_error():
    ctx = RequestContext(query_text="Error E1234 in /var/log/syslog docker failed")
    assert not tier0_heuristics.should_bypass_heuristics(ctx)


def test_tier0_header_off_skips_retrieval():
    orig = settings.enable_tier0_heuristics
    settings.enable_tier0_heuristics = True
    try:
        ctx = RequestContext(query_text="kubectl pods crash", rag_mode_header="off")
        asyncio.run(tier0_heuristics.run_tier0(ctx))
        assert ctx.retrieval == RetrievalDecision.SKIP
    finally:
        settings.enable_tier0_heuristics = orig


def test_gating_skips_simple_chat_with_confidence():
    ctx = RequestContext(
        query_text="tell me a joke",
        intent=IntentLabel.SIMPLE_CHAT,
        intent_confidence=0.7,
    )
    assert _gating_decision(ctx) == RetrievalDecision.SKIP


def test_gating_full_for_infra_debug():
    ctx = RequestContext(
        query_text="why is qdrant down",
        intent=IntentLabel.INFRA_DEBUG,
        intent_confidence=0.8,
    )
    assert _gating_decision(ctx) == RetrievalDecision.FULL

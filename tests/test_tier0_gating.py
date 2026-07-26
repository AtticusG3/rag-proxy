"""Unit tests for tier0 heuristics and retrieval gating."""

import asyncio

from rag_proxy.context import IntentLabel, RequestContext, RetrievalDecision
from rag_proxy.retrieval_policy import gating_decision, should_bypass_heuristics
from rag_proxy.stages import tier0_heuristics
from rag_proxy.stages.tier1_gating import gating_decision as gating_export


def test_tier0_bypasses_short_greeting():
    """Greetings skip retrieval so chat fluff does not burn embed/Qdrant."""
    ctx = RequestContext(query_text="hello there")
    assert should_bypass_heuristics(ctx)


def test_tier0_bypasses_ack():
    """Short acknowledgements skip retrieval (greetings/acks-only policy)."""
    assert should_bypass_heuristics(RequestContext(query_text="ok"))
    assert should_bypass_heuristics(RequestContext(query_text="Got it."))
    assert should_bypass_heuristics(RequestContext(query_text="sure"))


def test_tier0_does_not_bypass_knowledge_questions():
    """KB FAQs must retrieve; tier0 is greetings/acks only, not short-? skip."""
    assert not should_bypass_heuristics(
        RequestContext(query_text="why is the sky blue?")
    )
    assert not should_bypass_heuristics(
        RequestContext(query_text="what is the capital of france?")
    )
    assert not should_bypass_heuristics(RequestContext(query_text="what is rag?"))
    assert not should_bypass_heuristics(RequestContext(query_text="fix this?"))


def test_tier0_does_not_bypass_infra_error():
    ctx = RequestContext(query_text="Error E1234 in /var/log/syslog docker failed")
    assert not should_bypass_heuristics(ctx)


def test_tier0_infra_beats_greeting_prefix():
    """Infra signals force retrieve even when the utterance starts politely."""
    ctx = RequestContext(query_text="hi, please debug docker")
    assert not should_bypass_heuristics(ctx)


def test_tier0_header_off_skips_retrieval(monkeypatch):
    monkeypatch.setattr("rag_proxy.config.settings.enable_tier0_heuristics", True)
    ctx = RequestContext(query_text="kubectl pods crash", rag_mode_header="off")
    asyncio.run(tier0_heuristics.run_tier0(ctx))
    assert ctx.retrieval == RetrievalDecision.SKIP


def test_tier0_header_off_skips_when_heuristics_disabled(monkeypatch):
    monkeypatch.setattr("rag_proxy.config.settings.enable_tier0_heuristics", False)
    ctx = RequestContext(query_text="kubectl pods crash", rag_mode_header="off")
    asyncio.run(tier0_heuristics.run_tier0(ctx))
    assert ctx.retrieval == RetrievalDecision.SKIP


def test_tier0_header_force_does_not_skip(monkeypatch):
    monkeypatch.setattr("rag_proxy.config.settings.enable_tier0_heuristics", True)
    ctx = RequestContext(query_text="hi", rag_mode_header="force")
    asyncio.run(tier0_heuristics.run_tier0(ctx))
    assert ctx.retrieval == RetrievalDecision.FULL
    assert any("force" in t for t in ctx.stage_trace)


def test_gating_skips_simple_chat_with_confidence():
    ctx = RequestContext(
        query_text="tell me a joke",
        intent=IntentLabel.SIMPLE_CHAT,
        intent_confidence=0.7,
    )
    assert gating_decision(ctx) == RetrievalDecision.SKIP
    assert gating_export(ctx) == RetrievalDecision.SKIP


def test_gating_full_for_infra_debug():
    ctx = RequestContext(
        query_text="why is qdrant down",
        intent=IntentLabel.INFRA_DEBUG,
        intent_confidence=0.8,
    )
    assert gating_decision(ctx) == RetrievalDecision.FULL

"""Tests for consolidated retrieval policy."""

from rag_proxy.context import IntentLabel, RequestContext, RetrievalDecision
from rag_proxy.retrieval_policy import apply_early_policy, apply_late_policy


def test_early_policy_header_off():
    ctx = RequestContext(query_text="hello", rag_mode_header="off")
    trace = apply_early_policy(ctx)
    assert ctx.retrieval == RetrievalDecision.SKIP
    assert "tier0:header_off" in trace


def test_late_policy_log_only_does_not_mutate_retrieval(monkeypatch):
    monkeypatch.setattr("rag_proxy.config.settings.enable_retrieval_gating", True)
    monkeypatch.setattr("rag_proxy.config.settings.gating_log_only", True)
    ctx = RequestContext(
        query_text="hi",
        intent=IntentLabel.SIMPLE_CHAT,
        intent_confidence=0.9,
        retrieval=RetrievalDecision.FULL,
    )
    trace = apply_late_policy(ctx)
    assert ctx.retrieval == RetrievalDecision.FULL
    assert any("log_only" in t for t in trace)

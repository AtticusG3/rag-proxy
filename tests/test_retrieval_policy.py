"""Tests for consolidated retrieval policy."""

from rag_proxy.context import IntentLabel, RequestContext, RetrievalDecision
from rag_proxy.retrieval_policy import (
    apply_retrieval_policy,
    should_bypass_heuristics,
)


def test_tier0_policy_header_off():
    ctx = RequestContext(query_text="hello", rag_mode_header="off")
    trace = apply_retrieval_policy(ctx, "tier0")
    assert ctx.retrieval == RetrievalDecision.SKIP
    assert "tier0:header_off" in trace


def test_gating_policy_log_only_does_not_mutate_retrieval(monkeypatch):
    monkeypatch.setattr("rag_proxy.config.settings.enable_retrieval_gating", True)
    monkeypatch.setattr("rag_proxy.config.settings.gating_log_only", True)
    ctx = RequestContext(
        query_text="hi",
        intent=IntentLabel.SIMPLE_CHAT,
        intent_confidence=0.9,
        retrieval=RetrievalDecision.FULL,
    )
    trace = apply_retrieval_policy(ctx, "gating")
    assert ctx.retrieval == RetrievalDecision.FULL
    assert any("log_only" in t for t in trace)


def test_should_bypass_whitespace_only():
    ctx = RequestContext(query_text="   ")
    assert should_bypass_heuristics(ctx)


def test_gating_policy_honors_force_header(monkeypatch):
    monkeypatch.setattr("rag_proxy.config.settings.enable_retrieval_gating", True)
    monkeypatch.setattr("rag_proxy.config.settings.gating_log_only", False)
    ctx = RequestContext(
        query_text="tell me a joke",
        rag_mode_header="force",
        intent=IntentLabel.SIMPLE_CHAT,
        intent_confidence=0.9,
        retrieval=RetrievalDecision.FULL,
    )
    trace = apply_retrieval_policy(ctx, "gating")
    assert ctx.retrieval == RetrievalDecision.FULL
    assert "gating:forced_full" in trace

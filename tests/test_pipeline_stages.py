"""Tests for declarative pipeline stage gating."""

from rag_proxy.config import settings
from rag_proxy.context import RequestContext, RetrievalDecision
from rag_proxy.pipeline_stages import build_pipeline_stages


def test_retrieve_stage_requires_active_retrieval():
    stages = build_pipeline_stages()
    retrieve = next(s for s in stages if s.name == "retrieve")
    ctx_skip = RequestContext(retrieval=RetrievalDecision.SKIP)
    ctx_full = RequestContext(retrieval=RetrievalDecision.FULL)
    assert not retrieve.should_run(ctx_skip)
    assert retrieve.should_run(ctx_full)


def test_rerank_stage_requires_hits():
    stages = build_pipeline_stages()
    rerank = next(s for s in stages if s.name == "rerank")
    ctx_empty = RequestContext(retrieval=RetrievalDecision.FULL, hits=[])
    assert not rerank.should_run(ctx_empty)


def test_stage_budget_defaults_match_legacy():
    assert settings.stage_budget_rewrite_ms == 20
    assert settings.stage_budget_retrieve_ms == 50
    assert settings.stage_budget_graph_ms == 100


def test_rewrite_stage_disabled_when_flag_off(monkeypatch):
    monkeypatch.setattr("rag_proxy.config.settings.enable_query_rewrite", False)
    stages = build_pipeline_stages()
    rewrite = next(s for s in stages if s.name == "rewrite")
    assert not rewrite.enabled()


def test_rewrite_stage_enabled_when_flag_on(monkeypatch):
    monkeypatch.setattr("rag_proxy.config.settings.enable_query_rewrite", True)
    stages = build_pipeline_stages()
    rewrite = next(s for s in stages if s.name == "rewrite")
    assert rewrite.enabled()

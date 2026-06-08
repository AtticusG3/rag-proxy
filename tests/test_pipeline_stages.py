"""Tests for declarative pipeline stage gating."""

from rag_proxy.config import Settings
from rag_proxy.context import RequestContext, RetrievalDecision
from rag_proxy.pipeline_stages import build_pipeline_stages


def test_retrieve_stage_requires_active_retrieval():
    """Retrieve stage runs only when retrieval is not skipped."""
    stages = build_pipeline_stages()
    retrieve = next(s for s in stages if s.name == "retrieve")
    ctx_skip = RequestContext(retrieval=RetrievalDecision.SKIP)
    ctx_full = RequestContext(retrieval=RetrievalDecision.FULL)
    assert not retrieve.should_run(ctx_skip)
    assert retrieve.should_run(ctx_full)


def test_rerank_stage_requires_hits():
    """Rerank stage requires non-empty hits."""
    stages = build_pipeline_stages()
    rerank = next(s for s in stages if s.name == "rerank")
    ctx_empty = RequestContext(retrieval=RetrievalDecision.FULL, hits=[])
    assert not rerank.should_run(ctx_empty)


def test_invalid_float_env_falls_back_to_default(monkeypatch):
    """Invalid SIMILARITY_THRESHOLD falls back to the default."""
    monkeypatch.setenv("SIMILARITY_THRESHOLD", "not-a-float")
    s = Settings()
    assert s.similarity_threshold == 0.65


def test_stage_budget_defaults_match_legacy(monkeypatch):
    """Stage budget defaults match legacy values when unset."""
    monkeypatch.delenv("STAGE_BUDGET_REWRITE_MS", raising=False)
    monkeypatch.delenv("STAGE_BUDGET_RETRIEVE_MS", raising=False)
    monkeypatch.delenv("STAGE_BUDGET_GRAPH_MS", raising=False)
    s = Settings()
    assert s.stage_budget_rewrite_ms == 20
    assert s.stage_budget_retrieve_ms == 50
    assert s.stage_budget_graph_ms == 100


def test_rewrite_stage_disabled_when_flag_off(monkeypatch):
    """Rewrite stage is disabled when ENABLE_QUERY_REWRITE is off."""
    monkeypatch.setattr("rag_proxy.config.settings.enable_query_rewrite", False)
    stages = build_pipeline_stages()
    rewrite = next(s for s in stages if s.name == "rewrite")
    assert not rewrite.enabled()


def test_rewrite_stage_enabled_when_flag_on(monkeypatch):
    """Rewrite stage is enabled when ENABLE_QUERY_REWRITE is on."""
    monkeypatch.setattr("rag_proxy.config.settings.enable_query_rewrite", True)
    stages = build_pipeline_stages()
    rewrite = next(s for s in stages if s.name == "rewrite")
    assert rewrite.enabled()


def test_intent_stage_disabled_when_flag_off(monkeypatch):
    """Intent stage is disabled when ENABLE_INTENT_ROUTER is off."""
    monkeypatch.setattr("rag_proxy.config.settings.enable_intent_router", False)
    stages = build_pipeline_stages()
    intent = next(s for s in stages if s.name == "intent")
    assert not intent.enabled()


def test_gating_stage_disabled_when_flag_off(monkeypatch):
    """Gating stage is disabled when ENABLE_RETRIEVAL_GATING is off."""
    monkeypatch.setattr("rag_proxy.config.settings.enable_retrieval_gating", False)
    stages = build_pipeline_stages()
    gating = next(s for s in stages if s.name == "gating")
    assert not gating.enabled()


def test_graph_stage_requires_infra_intent_and_query():
    """Graph stage requires infra intent and non-empty query."""
    from rag_proxy.context import IntentLabel

    stages = build_pipeline_stages()
    graph = next(s for s in stages if s.name == "graph")
    ctx_skip_intent = RequestContext(
        query_text="why is qdrant down",
        intent=IntentLabel.SIMPLE_CHAT,
    )
    ctx_infra = RequestContext(
        query_text="why is qdrant down",
        intent=IntentLabel.INFRA_DEBUG,
    )
    ctx_no_query = RequestContext(intent=IntentLabel.INFRA_DEBUG)
    assert not graph.should_run(ctx_skip_intent)
    assert graph.should_run(ctx_infra)
    assert not graph.should_run(ctx_no_query)


def test_tools_stage_requires_active_retrieval():
    """Tools stage runs only when retrieval is active."""
    stages = build_pipeline_stages()
    tools = next(s for s in stages if s.name == "tools")
    ctx_skip = RequestContext(retrieval=RetrievalDecision.SKIP)
    ctx_full = RequestContext(retrieval=RetrievalDecision.FULL)
    assert not tools.should_run(ctx_skip)
    assert tools.should_run(ctx_full)


def test_memory_stage_requires_conversation_id():
    """Memory stage requires a conversation id."""
    stages = build_pipeline_stages()
    memory = next(s for s in stages if s.name == "memory")
    assert not memory.should_run(RequestContext())
    assert memory.should_run(RequestContext(conversation_id="conv-1"))

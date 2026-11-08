"""Orchestrator pipeline behavior."""

import asyncio

import pytest

from rag_proxy.registry.models import ModelRegistry
from rag_proxy.config import settings
from rag_proxy.context import ChunkHit, RequestContext
from rag_proxy.orchestrator import (
    augment_chat_payload_with_context,
    build_legacy_pipeline_stages,
    run_cognitive_pipeline,
)
from rag_proxy.pipeline_stages import PipelineStage


async def _slow_stage(ctx: RequestContext, _registry: ModelRegistry) -> None:
    await asyncio.sleep(0.05)
    ctx.stage_trace.append("slow:ok")


async def _expensive_stage(ctx: RequestContext, _registry: ModelRegistry) -> None:
    ctx.stage_trace.append("expensive:ran")


def test_orchestrator_skips_stage_when_budget_exhausted(monkeypatch):
    """Stages above remaining COGNITIVE_LATENCY_BUDGET_MS must not run."""

    monkeypatch.setattr(settings, "cognitive_latency_budget_ms", 10)
    monkeypatch.setattr(
        "rag_proxy.orchestrator._pipeline_stages_for_mode",
        lambda: [
            PipelineStage(
                name="slow",
                min_budget_ms=0,
                enabled=lambda: True,
                should_run=lambda _ctx: True,
                run=_slow_stage,
            ),
            PipelineStage(
                name="expensive",
                min_budget_ms=100,
                enabled=lambda: True,
                should_run=lambda _ctx: True,
                run=_expensive_stage,
            ),
        ],
    )
    monkeypatch.setattr("rag_proxy.orchestrator.log_pipeline_summary", lambda _ctx: None)

    ctx = RequestContext(query_text="test")
    asyncio.run(run_cognitive_pipeline(ctx))

    assert "slow:ok" in ctx.stage_trace
    assert "expensive:ran" not in ctx.stage_trace
    assert "slow" in ctx.latency_ms
    assert "expensive" not in ctx.latency_ms
    assert ctx.latency_ms["slow"] >= 40


def test_orchestrator_stage_error_fail_open_continues(monkeypatch):
    """A mid-pipeline exception must not abort later stages (partial RAG still possible)."""
    summary_calls: list[str] = []

    async def boom(_ctx: RequestContext, _registry: ModelRegistry) -> None:
        raise RuntimeError("stage failed")

    async def ok(ctx: RequestContext, _registry: ModelRegistry) -> None:
        ctx.stage_trace.append("ok:ran")

    monkeypatch.setattr(
        "rag_proxy.orchestrator.log_pipeline_summary",
        lambda ctx: summary_calls.append(ctx.trace_id or ""),
    )
    monkeypatch.setattr(
        "rag_proxy.orchestrator._pipeline_stages_for_mode",
        lambda: [
            PipelineStage(
                name="fail",
                min_budget_ms=0,
                enabled=lambda: True,
                should_run=lambda _ctx: True,
                run=boom,
            ),
            PipelineStage(
                name="ok",
                min_budget_ms=0,
                enabled=lambda: True,
                should_run=lambda _ctx: True,
                run=ok,
            ),
        ],
    )
    ctx = RequestContext(query_text="test")
    asyncio.run(run_cognitive_pipeline(ctx))

    assert any("fail:stage failed" in err for err in ctx.errors)
    assert "ok:ran" in ctx.stage_trace
    assert "fail" in ctx.latency_ms
    assert len(summary_calls) == 1
    assert "total_cognitive" in ctx.latency_ms


def test_stage_exec_timeout_uses_timeout_ms_not_min_budget(monkeypatch):
    """Entrance budget must not double as the hard exec timeout (retrieve footgun)."""
    from rag_proxy.orchestrator import _stage_exec_timeout_sec

    monkeypatch.setattr(settings, "stage_exec_timeout_ms", 30000)
    stage = PipelineStage(
        name="retrieve",
        min_budget_ms=50,
        enabled=lambda: True,
        should_run=lambda _ctx: True,
        run=_expensive_stage,
        timeout_ms=5000,
    )
    assert _stage_exec_timeout_sec(stage) == 5.0

    fallback = PipelineStage(
        name="tier0",
        min_budget_ms=0,
        enabled=lambda: True,
        should_run=lambda _ctx: True,
        run=_expensive_stage,
        timeout_ms=0,
    )
    assert _stage_exec_timeout_sec(fallback) == 30.0


async def _timeout_stage(_ctx: RequestContext, _registry: ModelRegistry) -> None:
    await asyncio.sleep(0.2)


def test_orchestrator_times_out_slow_stage(monkeypatch):
    monkeypatch.setattr(settings, "stage_exec_timeout_ms", 50)
    monkeypatch.setattr(
        "rag_proxy.orchestrator._pipeline_stages_for_mode",
        lambda: [
            PipelineStage(
                name="slow",
                min_budget_ms=0,
                enabled=lambda: True,
                should_run=lambda _ctx: True,
                run=_timeout_stage,
            )
        ],
    )
    monkeypatch.setattr("rag_proxy.orchestrator.log_pipeline_summary", lambda _ctx: None)

    ctx = RequestContext(query_text="test")
    asyncio.run(run_cognitive_pipeline(ctx))

    assert any("slow:timeout" in err for err in ctx.errors)
    assert "slow" in ctx.latency_ms


def test_legacy_pipeline_has_retrieve_and_context_only():
    stages = build_legacy_pipeline_stages()
    names = [s.name for s in stages]
    assert names == ["retrieve", "context"]


def test_pipeline_stages_for_mode_legacy_when_cognitive_off(monkeypatch):
    monkeypatch.setattr(settings, "enable_cognitive_pipeline", False)
    from rag_proxy.orchestrator import _pipeline_stages_for_mode

    names = [s.name for s in _pipeline_stages_for_mode()]
    assert names == ["retrieve", "context"]


_KB_CHUNK = "Example KB chunk about vector search."


def _stub_retrieval(monkeypatch, *, cognitive: bool) -> list[str]:
    """Fake embed + search at the stage seams; returns the list of retrieved queries."""
    calls: list[str] = []

    async def fake_embed(query, **_kwargs):
        calls.append(f"embed:{query}")
        return [0.1, 0.2]

    async def fake_search(query, **_kwargs):
        calls.append(f"search:{query}")
        return [ChunkHit(id="doc-1", text=_KB_CHUNK, score=0.9)]

    async def no_refresh(*_args, **_kwargs):
        return None

    monkeypatch.setattr(settings, "enable_cognitive_pipeline", cognitive)
    monkeypatch.setattr(settings, "enable_model_routing", False)
    monkeypatch.setattr(settings, "enable_tokenizer_estimate", False)
    for flag in (
        "enable_intent_router",
        "enable_query_rewrite",
        "enable_reranker",
        "enable_graph_lookup",
        "enable_memgraphrag",
        "enable_tools",
        "enable_rolling_memory",
    ):
        monkeypatch.setattr(settings, flag, False)
    monkeypatch.setattr(settings, "enable_tier0_heuristics", True)
    monkeypatch.setattr(settings, "enable_retrieval_gating", True)
    monkeypatch.setattr(settings, "gating_log_only", False)
    monkeypatch.setattr("rag_proxy.orchestrator._registry.refresh", no_refresh)
    monkeypatch.setattr("rag_proxy.pipeline_stages.hybrid_search", fake_search)
    monkeypatch.setattr("rag_proxy.stages.tier2_retrieval.embed_text", fake_embed)
    monkeypatch.setattr("rag_proxy.stages.tier2_retrieval.hybrid_search", fake_search)
    return calls


def _chat(query: str, rag_mode: str | None) -> tuple[dict, RequestContext]:
    headers = {"X-RAG-Mode": rag_mode} if rag_mode else {}
    data = {"model": "example-model", "messages": [{"role": "user", "content": query}]}
    return asyncio.run(augment_chat_payload_with_context(data, headers))


def _injected(data: dict) -> bool:
    return any(_KB_CHUNK in str(m.get("content", "")) for m in data["messages"])


def test_rag_mode_off_pipeline_off(monkeypatch):
    """Open WebUI task prompts send off: no embed, no search, no injected context."""
    calls = _stub_retrieval(monkeypatch, cognitive=False)
    data, ctx = _chat("what is qdrant", "off")
    assert calls == []
    assert not _injected(data)
    assert ctx.stage_trace == ["tier0:header_off"]


def test_rag_mode_force_pipeline_off(monkeypatch):
    calls = _stub_retrieval(monkeypatch, cognitive=False)
    data, ctx = _chat("hello", "force")
    assert calls == ["search:hello"]
    assert _injected(data)
    assert "tier0:force_retrieve" in ctx.stage_trace


@pytest.mark.parametrize("rag_mode", [None, "auto"])
def test_rag_mode_auto_pipeline_off(monkeypatch, rag_mode):
    """Legacy default retrieves even for a greeting: tier0/gating are cognitive-only."""
    calls = _stub_retrieval(monkeypatch, cognitive=False)
    data, _ctx = _chat("hello", rag_mode)
    assert calls == ["search:hello"]
    assert _injected(data)


@pytest.mark.parametrize(
    ("rag_mode", "retrieves", "trace"),
    [
        ("off", False, "tier0:header_off"),
        # force beats the tier0 greeting bypass and gating skip.
        ("force", True, "gating:forced_full"),
        ("auto", False, "tier0:bypass"),
        (None, False, "tier0:bypass"),
    ],
)
def test_rag_mode_pipeline_on(monkeypatch, rag_mode, retrieves, trace):
    calls = _stub_retrieval(monkeypatch, cognitive=True)
    data, ctx = _chat("hello", rag_mode)
    assert (calls == ["embed:hello", "search:hello"]) is retrieves
    assert _injected(data) is retrieves
    assert ctx.stage_trace.count(trace) == 1

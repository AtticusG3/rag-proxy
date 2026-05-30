"""Orchestrator pipeline behavior."""

import asyncio

import pytest

from rag_proxy.clients.bundle import ClientBundle
from rag_proxy.config import settings
from rag_proxy.context import RequestContext
from rag_proxy.orchestrator import (
    build_legacy_pipeline_stages,
    run_cognitive_pipeline,
)
from rag_proxy.pipeline_stages import PipelineStage


def test_pipeline_summary_on_stage_error(monkeypatch):
    summary_calls: list[str] = []

    async def boom(_ctx: RequestContext, _clients: ClientBundle) -> None:
        raise RuntimeError("stage failed")

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
            )
        ],
    )
    ctx = RequestContext(query_text="test")
    with pytest.raises(RuntimeError, match="stage failed"):
        asyncio.run(run_cognitive_pipeline(ctx))
    assert len(summary_calls) == 1
    assert "total_cognitive" in ctx.latency_ms


def test_legacy_pipeline_has_retrieve_and_context_only():
    stages = build_legacy_pipeline_stages()
    names = [s.name for s in stages]
    assert names == ["retrieve", "context"]


def test_pipeline_stages_for_mode_legacy_when_cognitive_off(monkeypatch):
    monkeypatch.setattr(settings, "enable_cognitive_pipeline", False)
    from rag_proxy.orchestrator import _pipeline_stages_for_mode

    names = [s.name for s in _pipeline_stages_for_mode()]
    assert names == ["retrieve", "context"]

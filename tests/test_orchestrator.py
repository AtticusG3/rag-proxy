"""Orchestrator pipeline behavior."""

import asyncio

import pytest

from rag_proxy.clients.bundle import ClientBundle
from rag_proxy.context import RequestContext
from rag_proxy.orchestrator import run_cognitive_pipeline
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
        "rag_proxy.orchestrator.build_pipeline_stages",
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

"""Performance regression benchmarks (optional; requires pytest-benchmark plugin)."""

from __future__ import annotations

import asyncio

import pytest

from rag_proxy.clients.qdrant import merge_fused_with_sparse_reserve
from rag_proxy.config import settings
from rag_proxy.context import ChunkHit, RequestContext
from rag_proxy.orchestrator import run_cognitive_pipeline
from rag_proxy.pipeline_stages import PipelineStage
from rag_proxy.registry.models import ModelRegistry
from rag_proxy.stages.tier2_context import apply_context_budget, dedupe_chunks


@pytest.mark.benchmark(group="context-assembly")
def test_benchmark_context_assembly_budget(benchmark, request: pytest.FixtureRequest):
    if hasattr(request.config.option, "benchmark_only") and not request.config.option.benchmark_only:
        pytest.skip("benchmarks run only with --benchmark-only")
    old_estimate = settings.enable_tokenizer_estimate
    settings.enable_tokenizer_estimate = False
    try:
        hits = [
            ChunkHit(id=f"doc-{i}", text=("chunk text " * 80) + str(i), score=1.0 - (i * 0.001))
            for i in range(200)
        ]

        def _work() -> int:
            out = dedupe_chunks(hits, enable_semantic=False)
            kept = apply_context_budget(out, 8000)
            return len(kept)

        kept_count = benchmark(_work)
        assert kept_count > 0
    finally:
        settings.enable_tokenizer_estimate = old_estimate


@pytest.mark.benchmark(group="hybrid-merge")
def test_benchmark_hybrid_merge_reserve(benchmark, request: pytest.FixtureRequest):
    if hasattr(request.config.option, "benchmark_only") and not request.config.option.benchmark_only:
        pytest.skip("benchmarks run only with --benchmark-only")
    fused_ids = [f"dense-{i}" for i in range(400)]
    sparse_only = [f"sparse-{i}" for i in range(120)]

    def _work() -> list[str]:
        return merge_fused_with_sparse_reserve(fused_ids, sparse_only, limit=120)

    merged = benchmark(_work)
    assert len(merged) == 120
    assert any(doc.startswith("sparse-") for doc in merged)


async def _cheap_stage(_ctx: RequestContext, _registry: ModelRegistry) -> None:
    return


async def _burn_budget_stage(_ctx: RequestContext, _registry: ModelRegistry) -> None:
    await asyncio.sleep(0.002)


@pytest.mark.benchmark(group="orchestrator-budget")
def test_benchmark_orchestrator_budget_skip(benchmark, request: pytest.FixtureRequest, monkeypatch):
    if hasattr(request.config.option, "benchmark_only") and not request.config.option.benchmark_only:
        pytest.skip("benchmarks run only with --benchmark-only")
    monkeypatch.setattr(settings, "stage_exec_timeout_ms", 1000)
    monkeypatch.setattr(settings, "cognitive_latency_budget_ms", 1)
    monkeypatch.setattr("rag_proxy.orchestrator.log_pipeline_summary", lambda _ctx: None)
    monkeypatch.setattr(
        "rag_proxy.orchestrator._pipeline_stages_for_mode",
        lambda: [
            PipelineStage(
                name="burn",
                min_budget_ms=0,
                enabled=lambda: True,
                should_run=lambda _ctx: True,
                run=_burn_budget_stage,
            ),
            PipelineStage(
                name="skipped",
                min_budget_ms=10,
                enabled=lambda: True,
                should_run=lambda _ctx: True,
                run=_cheap_stage,
            ),
        ],
    )

    def _work() -> int:
        ctx = RequestContext(query_text="bench")
        asyncio.run(run_cognitive_pipeline(ctx))
        return len(ctx.stage_trace)

    stage_count = benchmark(_work)
    assert stage_count == 0

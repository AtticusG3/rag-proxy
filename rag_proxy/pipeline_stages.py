"""Declarative cognitive pipeline stage registry."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from rag_proxy.clients.bundle import ClientBundle
from rag_proxy.config import settings
from rag_proxy.context import PipelineTier, RequestContext, RetrievalDecision
from rag_proxy.stages import routing as routing_stage
from rag_proxy.stages import tier0_heuristics, tier1_gating, tier1_intent
from rag_proxy.stages import tier2_context, tier2_rerank, tier2_retrieval, tier2_rewrite
from rag_proxy.stages import tier3_graph, tier3_memory, tier3_tools


@dataclass(frozen=True)
class PipelineStage:
    name: str
    min_budget_ms: float
    enabled: Callable[[], bool]
    should_run: Callable[[RequestContext], bool]
    run: Callable[[RequestContext, ClientBundle], Awaitable[None]]


def _retrieval_active(ctx: RequestContext) -> bool:
    return ctx.retrieval != RetrievalDecision.SKIP


async def _run_tier0(ctx: RequestContext, clients: ClientBundle) -> None:
    await tier0_heuristics.run_tier0(ctx)
    if ctx.retrieval != RetrievalDecision.SKIP:
        ctx.tier = PipelineTier.TIER1_LIGHT


async def _run_retrieve(ctx: RequestContext, clients: ClientBundle) -> None:
    ctx.tier = PipelineTier.TIER2_RETRIEVAL
    await tier2_retrieval.run_retrieval(ctx, clients)


async def _run_graph(ctx: RequestContext, clients: ClientBundle) -> None:
    ctx.tier = PipelineTier.TIER3_HEAVY
    await tier3_graph.run_graph(ctx)


def build_pipeline_stages() -> list[PipelineStage]:
    return [
        PipelineStage(
            name="tier0",
            min_budget_ms=0,
            enabled=lambda: True,
            should_run=lambda _ctx: True,
            run=_run_tier0,
        ),
        PipelineStage(
            name="intent",
            min_budget_ms=0,
            enabled=lambda: True,
            should_run=lambda _ctx: True,
            run=lambda ctx, clients: tier1_intent.run_intent(ctx, clients),
        ),
        PipelineStage(
            name="gating",
            min_budget_ms=0,
            enabled=lambda: True,
            should_run=lambda _ctx: True,
            run=lambda ctx, _clients: tier1_gating.run_gating(ctx),
        ),
        PipelineStage(
            name="routing",
            min_budget_ms=float(settings.stage_budget_routing_ms),
            enabled=lambda: settings.enable_model_routing,
            should_run=lambda _ctx: True,
            run=lambda ctx, clients: routing_stage.run_routing(ctx, clients),
        ),
        PipelineStage(
            name="rewrite",
            min_budget_ms=float(settings.stage_budget_rewrite_ms),
            enabled=lambda: settings.enable_query_rewrite,
            should_run=_retrieval_active,
            run=lambda ctx, _clients: tier2_rewrite.run_rewrite(ctx),
        ),
        PipelineStage(
            name="retrieve",
            min_budget_ms=float(settings.stage_budget_retrieve_ms),
            enabled=lambda: True,
            should_run=_retrieval_active,
            run=_run_retrieve,
        ),
        PipelineStage(
            name="rerank",
            min_budget_ms=float(settings.rerank_timeout_ms),
            enabled=lambda: settings.enable_reranker,
            should_run=lambda ctx: _retrieval_active(ctx) and bool(ctx.hits),
            run=lambda ctx, _clients: tier2_rerank.run_rerank(ctx),
        ),
        PipelineStage(
            name="graph",
            min_budget_ms=float(settings.stage_budget_graph_ms),
            enabled=lambda: settings.enable_graph_lookup,
            should_run=lambda _ctx: True,
            run=_run_graph,
        ),
        PipelineStage(
            name="tools",
            min_budget_ms=float(settings.tool_budget_ms),
            enabled=lambda: settings.enable_tools,
            should_run=lambda _ctx: True,
            run=lambda ctx, _clients: tier3_tools.run_tools(ctx),
        ),
        PipelineStage(
            name="memory",
            min_budget_ms=0,
            enabled=lambda: settings.enable_rolling_memory,
            should_run=lambda _ctx: True,
            run=lambda ctx, _clients: tier3_memory.run_memory(ctx),
        ),
        PipelineStage(
            name="context",
            min_budget_ms=0,
            enabled=lambda: True,
            should_run=lambda ctx: bool(ctx.hits),
            run=lambda ctx, clients: tier2_context.run_context_assembly(ctx, clients),
        ),
    ]

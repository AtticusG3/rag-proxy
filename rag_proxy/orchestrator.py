"""Cognitive pipeline orchestrator."""

from __future__ import annotations

import asyncio
import logging
import time

from rag_proxy.registry.models import ModelRegistry
from rag_proxy.config import settings
from rag_proxy.context import RequestContext
from rag_proxy.legacy_rag import extract_query_text
from rag_proxy.observability import (
    log_pipeline_summary,
    log_rag_request,
    new_trace_id,
    observe_rag_augment_duration,
    observe_stage_latency,
)
from rag_proxy.pipeline_stages import (
    PipelineStage,
    build_legacy_pipeline_stages,
    build_pipeline_stages,
)

log = logging.getLogger("rag-proxy")

_registry = ModelRegistry()


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def _budget_remaining(ctx: RequestContext) -> float:
    if ctx.cognitive_start_ms <= 0:
        return float(settings.cognitive_latency_budget_ms)
    spent = _elapsed_ms(ctx.cognitive_start_ms)
    return max(0.0, settings.cognitive_latency_budget_ms - spent)


def _stage_exec_timeout_sec(stage: PipelineStage) -> float | None:
    if stage.min_budget_ms > 0:
        ms = int(stage.min_budget_ms)
    else:
        ms = settings.stage_exec_timeout_ms
    if ms <= 0:
        return None
    return ms / 1000.0


async def _run_stage(stage: PipelineStage, ctx: RequestContext) -> None:
    t0 = time.perf_counter()
    timeout_sec = _stage_exec_timeout_sec(stage)
    try:
        if timeout_sec is None:
            await stage.run(ctx, _registry)
        else:
            await asyncio.wait_for(stage.run(ctx, _registry), timeout=timeout_sec)
    except asyncio.TimeoutError:
        log.warning("Stage %s timed out after %.3fs", stage.name, timeout_sec or 0.0)
        ctx.errors.append(f"{stage.name}:timeout")
        return
    elapsed = _elapsed_ms(t0)
    ctx.latency_ms[stage.name] = elapsed
    observe_stage_latency(stage.name, elapsed / 1000.0)


def build_request_context_from_http(
    data: dict,
    headers: dict[str, str] | None = None,
) -> RequestContext:
    messages = data.get("messages", [])
    hdr = {k.lower(): v for k, v in (headers or {}).items()}
    rag_header = hdr.get("x-rag-mode")
    no_cache = hdr.get("x-no-cache", "").lower() in ("1", "true", "yes")
    conv = hdr.get("x-conversation-id")

    return RequestContext(
        messages=list(messages),
        data=data,
        query_text=extract_query_text(messages),
        requested_model=data.get("model"),
        stream=bool(data.get("stream")),
        rag_mode_header=rag_header,
        no_cache=no_cache,
        conversation_id=conv,
    )


def _pipeline_stages_for_mode() -> list[PipelineStage]:
    if settings.enable_cognitive_pipeline:
        return build_pipeline_stages()
    return build_legacy_pipeline_stages()


def _needs_model_registry_refresh() -> bool:
    return settings.enable_model_routing or settings.enable_cognitive_pipeline


async def run_cognitive_pipeline(ctx: RequestContext) -> None:
    ctx.trace_id = ctx.trace_id or new_trace_id()
    ctx.cognitive_start_ms = time.perf_counter()

    stages = _pipeline_stages_for_mode()
    try:
        for stage in stages:
            if not stage.enabled():
                continue
            if not stage.should_run(ctx):
                continue
            if _budget_remaining(ctx) < stage.min_budget_ms:
                continue
            await _run_stage(stage, ctx)
    finally:
        ctx.latency_ms["total_cognitive"] = _elapsed_ms(ctx.cognitive_start_ms)
        log_pipeline_summary(ctx)


def apply_context_to_payload(data: dict, ctx: RequestContext) -> dict:
    """Single boundary: copy pipeline context back into the chat payload."""
    data = {**data, "messages": ctx.messages}
    if ctx.selected_model and settings.model_routing_mode == "force":
        data["model"] = ctx.selected_model
    return data


async def augment_chat_payload_with_context(
    data: dict,
    headers: dict[str, str] | None = None,
) -> tuple[dict, RequestContext]:
    """Augment messages and return the context used for capture/observability."""
    ctx = build_request_context_from_http(data, headers)
    if _needs_model_registry_refresh():
        await _registry.refresh()
    augment_t0 = time.perf_counter()
    await run_cognitive_pipeline(ctx)
    observe_rag_augment_duration(time.perf_counter() - augment_t0)
    data = apply_context_to_payload(data, ctx)
    log_rag_request(ctx)
    return data, ctx

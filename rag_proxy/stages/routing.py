"""Capability-aware model routing."""

from __future__ import annotations

import logging

from rag_proxy.clients.bundle import ClientBundle
from rag_proxy.config import settings
from rag_proxy.context import IntentLabel, RequestContext

log = logging.getLogger("rag-proxy")

_INTENT_DEFAULT_ROUTES: dict[IntentLabel, str] = {
    IntentLabel.CODE_GENERATION: "bonsai-8b",
    IntentLabel.CODE_REVIEW: "bonsai-8b",
    IntentLabel.REASONING_HEAVY: "qwen3.5-9b",
    IntentLabel.SUMMARIZATION: "phi4-mini",
}


async def run_routing(ctx: RequestContext, clients: ClientBundle) -> None:
    if not settings.enable_model_routing:
        return

    routes = settings.model_routes()
    target = routes.get(ctx.intent.value) or _INTENT_DEFAULT_ROUTES.get(ctx.intent)
    if not target:
        return
    if not clients.model_registry.model_exists(target):
        ctx.errors.append(f"route:unknown_model:{target}")
        return

    ctx.selected_model = target
    ctx.stage_trace.append(f"route:{target}:{settings.model_routing_mode}")

    if settings.model_routing_mode == "force" and ctx.data is not None:
        ctx.data["model"] = target
    elif settings.model_routing_mode == "suggest":
        log.info(
            f"trace={ctx.trace_id} suggested_model={target} for intent={ctx.intent.value}"
        )

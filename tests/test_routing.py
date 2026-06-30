"""Model routing: intent defaults must resolve via IntentLabel keys."""

import asyncio
from unittest.mock import MagicMock

from rag_proxy.config import settings
from rag_proxy.context import IntentLabel, RequestContext
from rag_proxy.orchestrator import apply_context_to_payload
from rag_proxy.stages.routing import run_routing


def test_default_route_resolves_intent_enum_key(monkeypatch):
    """CODE_GENERATION maps to bonsai-8b when MODEL_ROUTES_JSON is empty."""
    monkeypatch.setattr("rag_proxy.config.settings.enable_model_routing", True)
    monkeypatch.setattr("rag_proxy.config.settings.model_routes_json", "")

    ctx = RequestContext(intent=IntentLabel.CODE_GENERATION, data={"model": "requested"})
    registry = MagicMock()
    registry.model_exists.return_value = True

    asyncio.run(run_routing(ctx, registry))

    assert ctx.selected_model == "bonsai-8b"
    assert ctx.data["model"] == "requested"
    assert any(s.startswith("route:bonsai-8b:") for s in ctx.stage_trace)


def test_force_route_does_not_mutate_ctx_data(monkeypatch):
    monkeypatch.setattr("rag_proxy.config.settings.enable_model_routing", True)
    monkeypatch.setattr("rag_proxy.config.settings.model_routing_mode", "force")
    monkeypatch.setattr("rag_proxy.config.settings.model_routes_json", "")

    ctx = RequestContext(intent=IntentLabel.CODE_GENERATION, data={"model": "requested"})
    registry = MagicMock()
    registry.model_exists.return_value = True

    asyncio.run(run_routing(ctx, registry))

    assert ctx.selected_model == "bonsai-8b"
    assert ctx.data["model"] == "requested"


def test_apply_context_to_payload_force_route(monkeypatch):
    monkeypatch.setattr(settings, "model_routing_mode", "force")
    data = {"model": "requested", "messages": []}
    ctx = RequestContext(selected_model="bonsai-8b", messages=[{"role": "user", "content": "hi"}])
    out = apply_context_to_payload(data, ctx)
    assert out["model"] == "bonsai-8b"

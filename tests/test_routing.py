"""Model routing: intent defaults must resolve via IntentLabel keys."""

import asyncio
from unittest.mock import MagicMock

from rag_proxy.context import IntentLabel, RequestContext
from rag_proxy.stages.routing import run_routing


def test_default_route_resolves_intent_enum_key(monkeypatch):
    """CODE_GENERATION maps to bonsai-8b when MODEL_ROUTES_JSON is empty."""
    monkeypatch.setattr("rag_proxy.config.settings.enable_model_routing", True)
    monkeypatch.setattr("rag_proxy.config.settings.model_routes_json", "")

    ctx = RequestContext(intent=IntentLabel.CODE_GENERATION)
    clients = MagicMock()
    clients.model_registry.model_exists.return_value = True

    asyncio.run(run_routing(ctx, clients))

    assert ctx.selected_model == "bonsai-8b"
    assert any(s.startswith("route:bonsai-8b:") for s in ctx.stage_trace)

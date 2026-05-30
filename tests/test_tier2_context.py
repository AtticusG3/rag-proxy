"""Tests for context budget estimation."""

from rag_proxy.clients.bundle import ClientBundle
from rag_proxy.config import settings
from rag_proxy.context import RequestContext
from rag_proxy.registry.models import ModelCapabilities
from rag_proxy.stages.tier2_context import estimate_message_chars, resolve_inject_budget_chars


def test_estimate_message_chars_clamps_unbroken_text(monkeypatch):
    monkeypatch.setattr(settings, "enable_tokenizer_estimate", True)
    log_line = "ERROR: " + "x" * 500
    total = estimate_message_chars([{"role": "user", "content": log_line}])
    assert total >= len(log_line)


def test_estimate_message_chars_counts_string_content_once(monkeypatch):
    monkeypatch.setattr(settings, "enable_tokenizer_estimate", False)
    content = "hello world"
    total = estimate_message_chars([{"role": "user", "content": content}])
    assert total == len(content)


def test_resolve_inject_budget_unknown_model_uses_char_fallback(monkeypatch):
    monkeypatch.setattr(settings, "context_fallback_chars", 8000)
    monkeypatch.setattr(settings, "default_completion_reserve", 1024)
    ctx = RequestContext(messages=[], requested_model="unknown-model")
    clients = ClientBundle()
    budget = resolve_inject_budget_chars(ctx, clients)
    assert budget == settings.context_fallback_chars - (settings.default_completion_reserve * 4)


def test_resolve_inject_budget_uses_model_context_tokens(monkeypatch):
    monkeypatch.setattr(settings, "context_budget_ratio", 0.25)
    monkeypatch.setattr(settings, "default_completion_reserve", 1024)
    ctx = RequestContext(messages=[], requested_model="test-model")
    clients = ClientBundle()
    clients.model_registry._cache["test-model"] = ModelCapabilities(
        model_id="test-model",
        context_length=8192,
    )
    char_budget = int(8192 * settings.context_budget_ratio) * 4
    reserve = settings.default_completion_reserve * 4
    assert resolve_inject_budget_chars(ctx, clients) == char_budget - reserve

"""Tests for context budget estimation."""

from rag_proxy.config import settings
from rag_proxy.context import RequestContext
from rag_proxy.registry.models import ModelCapabilities, ModelRegistry
from rag_proxy.stages.tier2_context import (
    estimate_message_chars,
    estimate_message_tokens,
    resolve_inject_budget_chars,
)


def test_estimate_message_chars_counts_string_content(monkeypatch):
    monkeypatch.setattr(settings, "enable_tokenizer_estimate", False)
    content = "hello world"
    total = estimate_message_chars([{"role": "user", "content": content}])
    assert total == len(content)


def test_estimate_message_tokens_when_enabled(monkeypatch):
    monkeypatch.setattr(settings, "enable_tokenizer_estimate", True)
    total = estimate_message_tokens([{"role": "user", "content": "hello world"}])
    assert total >= 2


def test_resolve_inject_budget_unknown_model_uses_char_fallback(monkeypatch):
    monkeypatch.setattr(settings, "enable_tokenizer_estimate", False)
    monkeypatch.setattr(settings, "context_fallback_chars", 8000)
    monkeypatch.setattr(settings, "default_completion_reserve", 1024)
    ctx = RequestContext(messages=[], requested_model="unknown-model")
    registry = ModelRegistry()
    budget = resolve_inject_budget_chars(ctx, registry)
    assert budget == settings.context_fallback_chars


def test_resolve_inject_budget_unknown_model_long_chat_is_capped(monkeypatch):
    monkeypatch.setattr(settings, "enable_tokenizer_estimate", False)
    monkeypatch.setattr(settings, "context_fallback_chars", 8000)
    monkeypatch.setattr(settings, "context_budget_ratio", 0.25)
    monkeypatch.setattr(settings, "default_completion_reserve", 1024)
    ctx = RequestContext(
        messages=[{"role": "user", "content": "x" * 20000}],
        requested_model="unknown-model",
    )
    # Assumed window 8000 / 0.25 = 32000 chars, reserve 4096, conversation 20000.
    assert resolve_inject_budget_chars(ctx, ModelRegistry()) == 32000 - 4096 - 20000


def test_resolve_inject_budget_unknown_model_full_window_is_zero(monkeypatch):
    monkeypatch.setattr(settings, "enable_tokenizer_estimate", False)
    monkeypatch.setattr(settings, "context_fallback_chars", 8000)
    monkeypatch.setattr(settings, "context_budget_ratio", 0.25)
    monkeypatch.setattr(settings, "default_completion_reserve", 1024)
    ctx = RequestContext(
        messages=[{"role": "user", "content": "x" * 40000}],
        requested_model="unknown-model",
    )
    assert resolve_inject_budget_chars(ctx, ModelRegistry()) == 0


def _known_registry() -> ModelRegistry:
    registry = ModelRegistry()
    registry._cache["test-model"] = ModelCapabilities(model_id="test-model", context_length=8192)
    return registry


def test_resolve_inject_budget_uses_model_context_tokens(monkeypatch):
    monkeypatch.setattr(settings, "enable_tokenizer_estimate", False)
    monkeypatch.setattr(settings, "context_budget_ratio", 0.25)
    monkeypatch.setattr(settings, "default_completion_reserve", 1024)
    ctx = RequestContext(messages=[], requested_model="test-model")
    assert resolve_inject_budget_chars(ctx, _known_registry()) == int(8192 * 0.25) * 4


def test_resolve_inject_budget_known_model_long_chat_keeps_slice(monkeypatch):
    monkeypatch.setattr(settings, "enable_tokenizer_estimate", False)
    monkeypatch.setattr(settings, "context_budget_ratio", 0.25)
    monkeypatch.setattr(settings, "default_completion_reserve", 1024)
    # 12000 chars is past the old "slice minus conversation" cutoff but well inside the window.
    ctx = RequestContext(
        messages=[{"role": "user", "content": "x" * 12000}], requested_model="test-model"
    )
    assert resolve_inject_budget_chars(ctx, _known_registry()) == int(8192 * 0.25) * 4


def test_resolve_inject_budget_known_model_full_window_is_zero(monkeypatch):
    monkeypatch.setattr(settings, "enable_tokenizer_estimate", False)
    monkeypatch.setattr(settings, "default_completion_reserve", 1024)
    ctx = RequestContext(
        messages=[{"role": "user", "content": "x" * (8192 * 4)}], requested_model="test-model"
    )
    assert resolve_inject_budget_chars(ctx, _known_registry()) == 0


def test_resolve_inject_budget_known_model_capped_by_room_left(monkeypatch):
    monkeypatch.setattr(settings, "enable_tokenizer_estimate", False)
    monkeypatch.setattr(settings, "context_budget_ratio", 0.25)
    monkeypatch.setattr(settings, "default_completion_reserve", 1024)
    # Window 32768 chars, reserve 4096, conversation 27000 -> 1672 chars left.
    ctx = RequestContext(
        messages=[{"role": "user", "content": "x" * 27000}], requested_model="test-model"
    )
    assert resolve_inject_budget_chars(ctx, _known_registry()) == 8192 * 4 - 4096 - 27000


def test_run_context_assembly_traces_zero_budget(monkeypatch):
    import asyncio

    from rag_proxy.context import ChunkHit
    from rag_proxy.stages.tier2_context import run_context_assembly

    monkeypatch.setattr(settings, "enable_tokenizer_estimate", False)
    monkeypatch.setattr(settings, "default_completion_reserve", 1024)
    ctx = RequestContext(
        messages=[{"role": "user", "content": "x" * (8192 * 4)}], requested_model="test-model"
    )
    ctx.hits = [ChunkHit(id="a", text="some passage", score=0.9, source="dense")]
    asyncio.run(run_context_assembly(ctx, _known_registry()))
    assert ctx.hits == []
    assert "inject:budget0" in ctx.stage_trace
    assert "inject:empty_text" not in ctx.stage_trace


def test_run_context_assembly_traces_empty_text_not_budget0(monkeypatch):
    import asyncio

    from rag_proxy.context import ChunkHit
    from rag_proxy.stages.tier2_context import run_context_assembly

    monkeypatch.setattr(settings, "enable_tokenizer_estimate", False)
    ctx = RequestContext(messages=[{"role": "user", "content": "hi"}], requested_model="test-model")
    ctx.hits = [ChunkHit(id="a", text="   ", score=0.9, source="dense")]
    asyncio.run(run_context_assembly(ctx, _known_registry()))
    assert ctx.hits == []
    assert "inject:empty_text" in ctx.stage_trace
    assert "inject:budget0" not in ctx.stage_trace

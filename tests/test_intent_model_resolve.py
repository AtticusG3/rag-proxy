"""Tests for intent model resolution ('auto' via /running) and endpoint selection."""

import asyncio

import pytest

import rag_proxy.clients.llama_swap as llama_swap
from rag_proxy.clients.llama_swap import _running_models, resolve_intent_model
from rag_proxy.config import settings


def _clear_auto_cache():
    llama_swap._auto_model_cache.clear()


def test_explicit_intent_model_passes_through_without_lookup(monkeypatch):
    """A concrete model id is used verbatim and never queries /running."""
    _clear_auto_cache()
    monkeypatch.setattr(settings, "intent_model", "owl-alpha")

    async def fail_fetch(*_a, **_k):
        raise AssertionError("/running must not be queried for an explicit model id")

    monkeypatch.setattr(llama_swap, "fetch_running_model", fail_fetch)
    assert asyncio.run(resolve_intent_model()) == "owl-alpha"


def test_empty_intent_model_resolves_to_none(monkeypatch):
    """No configured model means the intent LLM call is skipped."""
    _clear_auto_cache()
    monkeypatch.setattr(settings, "intent_model", "")
    assert asyncio.run(resolve_intent_model()) is None


def test_auto_uses_running_model_from_intent_endpoint(monkeypatch):
    """'auto' reuses whichever model llama-swap has loaded, at the intent endpoint."""
    _clear_auto_cache()
    monkeypatch.setattr(settings, "intent_model", "auto")
    monkeypatch.setattr(settings, "intent_model_url", "http://intent.test:9000")

    captured: dict = {}

    async def fake_running(base_url=None):
        captured["base_url"] = base_url
        return "qwen3-8b"

    monkeypatch.setattr(llama_swap, "fetch_running_model", fake_running)

    assert asyncio.run(resolve_intent_model()) == "qwen3-8b"
    assert captured["base_url"] == "http://intent.test:9000"


def test_auto_returns_none_when_nothing_loaded(monkeypatch):
    """When no model is warm, auto skips the LLM instead of forcing a swap."""
    _clear_auto_cache()
    monkeypatch.setattr(settings, "intent_model", "auto")

    async def idle(base_url=None):
        return None

    monkeypatch.setattr(llama_swap, "fetch_running_model", idle)
    assert asyncio.run(resolve_intent_model()) is None


def test_auto_result_is_cached_within_ttl(monkeypatch):
    """Repeated 'auto' resolution reuses the cached id instead of re-querying /running."""
    _clear_auto_cache()
    monkeypatch.setattr(settings, "intent_model", "auto")
    monkeypatch.setattr(settings, "intent_model_auto_ttl_sec", 300)

    calls = {"n": 0}

    async def counting(base_url=None):
        calls["n"] += 1
        return "loaded"

    monkeypatch.setattr(llama_swap, "fetch_running_model", counting)

    assert asyncio.run(resolve_intent_model()) == "loaded"
    assert asyncio.run(resolve_intent_model()) == "loaded"
    assert calls["n"] == 1


def test_auto_refetches_after_ttl_expiry(monkeypatch):
    """A zero-length cache window means every 'auto' call re-queries the loaded model."""
    _clear_auto_cache()
    monkeypatch.setattr(settings, "intent_model", "auto")
    monkeypatch.setattr(settings, "intent_model_auto_ttl_sec", 0)

    seq = iter(["model-a", "model-b"])

    async def swapping(base_url=None):
        return next(seq)

    monkeypatch.setattr(llama_swap, "fetch_running_model", swapping)

    assert asyncio.run(resolve_intent_model()) == "model-a"
    assert asyncio.run(resolve_intent_model()) == "model-b"


def test_intent_base_url_prefers_override_then_falls_back(monkeypatch):
    """Dedicated endpoint is used when set; otherwise the chat upstream."""
    monkeypatch.setattr(settings, "llama_swap_url", "http://swap.test:8080/")
    monkeypatch.setattr(settings, "intent_model_url", "")
    assert settings.intent_base_url() == "http://swap.test:8080"

    monkeypatch.setattr(settings, "intent_model_url", "http://intent.test:9000/")
    assert settings.intent_base_url() == "http://intent.test:9000"


@pytest.mark.parametrize(
    "payload, expected",
    [
        ({}, []),
        ({"model": "DeepSeek-V3", "state": "ready"}, ["DeepSeek-V3"]),
        ({"model": "DeepSeek-V3", "state": "starting"}, []),
        ({"model": "DeepSeek-V3", "state": "stopping"}, []),
        (
            {"running": [{"model": "a", "state": "ready"}, {"model": "b", "state": "stopped"}]},
            ["a"],
        ),
        ({"running": ["llama3", "nomic-embed"]}, ["llama3", "nomic-embed"]),
        ({"running": []}, []),
        ({"running": [{"id": "by-id", "state": "ready"}]}, ["by-id"]),
        ("garbage", []),
    ],
)
def test_running_models_parses_llama_swap_shapes(payload, expected):
    """/running parser accepts the object/list shapes across llama-swap builds."""
    assert _running_models(payload) == expected

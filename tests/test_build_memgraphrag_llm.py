"""Tests for MemGraphRAG build LLM request shaping and JSON parse helpers."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import patch

from scripts.build_memgraphrag_index import LLMClient, _extract_json_object


def test_extract_json_object_strips_markdown_fence() -> None:
    """Models sometimes wrap JSON in fences; extraction must still parse it."""
    raw = '```json\n{"entities": [{"text": "Paris", "type": "LOCATION"}]}\n```'
    data = _extract_json_object(raw)
    assert data == {"entities": [{"text": "Paris", "type": "LOCATION"}]}


def test_extract_json_object_rejects_non_object() -> None:
    """Arrays alone are not a valid extraction payload."""
    assert _extract_json_object("[1, 2]") is None


def test_llm_client_disables_thinking_and_caps_answer_tokens() -> None:
    """Extraction must not inherit coding-model's huge CLI reasoning budget."""
    captured: dict[str, Any] = {}

    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {
                "choices": [
                    {"message": {"content": '{"entities":[]}', "reasoning_content": ""}}
                ]
            }

    class _FakeClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def post(self, url: str, json: dict[str, Any] | None = None, headers: dict | None = None) -> _FakeResponse:
            captured["url"] = url
            captured["json"] = json
            return _FakeResponse()

        async def __aenter__(self) -> _FakeClient:
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

    async def _run() -> str:
        with patch("scripts.build_memgraphrag_index.httpx.AsyncClient", _FakeClient):
            client = LLMClient("http://nugget:8081/v1", "coding-model")
            return await client.chat("sys", "user")

    text = asyncio.run(_run())
    assert text == '{"entities":[]}'
    payload = captured["json"]
    assert payload["max_tokens"] == 768
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert "thinking_budget_tokens" not in payload
    # Ensure we did not leave an unbounded generation path.
    assert payload["max_tokens"] < 4096

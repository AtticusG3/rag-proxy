"""HTTP capture integration for chat proxy responses."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from rag_proxy.app import app
from rag_proxy.config import settings
from rag_proxy.context import ChunkHit

from conftest import buffered_upstream_response, capture_upstream_body, pooled_client_mock


def test_chat_capture_writes_sanitized_finetune_and_rag_records(tmp_path, monkeypatch):
    """Completed chat requests should persist useful data without injected RAG context."""
    monkeypatch.setattr(settings, "enable_transcript_capture", True)
    monkeypatch.setattr(settings, "finetune_log_path", str(tmp_path / "finetune.jsonl"))
    monkeypatch.setattr(settings, "rag_improvement_log_path", str(tmp_path / "rag.jsonl"))
    monkeypatch.setattr(settings, "enable_cognitive_pipeline", False)

    async def fake_hybrid(_query, limit, score_threshold=None, no_cache=False):
        return [
            ChunkHit(
                id="doc-1",
                text="private deployment chunk",
                score=0.91,
                source="dense",
            )
        ]

    upstream_body = json.dumps(
        {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Use systemd."},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"total_tokens": 20},
        }
    ).encode()
    mock_response = buffered_upstream_response(upstream_body)
    mock_client = pooled_client_mock(mock_response)
    captured = capture_upstream_body(mock_client)

    with patch("rag_proxy.pipeline_stages.hybrid_search", fake_hybrid):
        with patch("rag_proxy.upstream_client.httpx.AsyncClient", return_value=mock_client):
            with TestClient(app) as client:
                resp = client.post(
                    "/v1/chat/completions",
                    headers={"X-Conversation-Id": "conv-1"},
                    json={
                        "model": "chat-demo",
                        "messages": [{"role": "user", "content": "How do I deploy?"}],
                    },
                )

    assert resp.status_code == 200
    upstream = json.loads(captured[0])
    assert "private deployment chunk" in upstream["messages"][0]["content"]

    ft = json.loads((tmp_path / "finetune.jsonl").read_text(encoding="utf-8").strip())
    rag = json.loads((tmp_path / "rag.jsonl").read_text(encoding="utf-8").strip())
    assert ft["record_type"] == "finetune_turn"
    assert ft["messages"] == [{"role": "user", "content": "How do I deploy?"}]
    assert ft["assistant"] == {"role": "assistant", "content": "Use systemd."}
    assert rag["record_type"] == "rag_turn"
    assert rag["conversation_id"] == "conv-1"
    assert rag["qa_pair"] == {"question": "How do I deploy?", "answer": "Use systemd."}
    assert rag["hits"][0]["text_preview"] == "private deployment chunk"


def test_chat_capture_error_does_not_change_response(monkeypatch):
    """Capture failure must not affect the upstream response delivered to clients."""
    monkeypatch.setattr(settings, "enable_transcript_capture", True)
    monkeypatch.setattr(settings, "enable_cognitive_pipeline", False)
    mock_response = buffered_upstream_response(
        b'{"choices":[{"message":{"role":"assistant","content":"ok"}}]}'
    )
    mock_client = pooled_client_mock(mock_response)

    async def fake_hybrid(_query, limit, score_threshold=None, no_cache=False):
        return []

    with patch("rag_proxy.pipeline_stages.hybrid_search", fake_hybrid):
        with patch("rag_proxy.upstream_client.httpx.AsyncClient", return_value=mock_client):
            with patch("rag_proxy.capture.enqueue_records", side_effect=OSError("disk full")):
                with TestClient(app) as client:
                    resp = client.post(
                        "/v1/chat/completions",
                        json={
                            "model": "chat-demo",
                            "messages": [{"role": "user", "content": "Hi"}],
                        },
                    )

    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "ok"


def test_chat_capture_writes_streaming_completion_record(tmp_path, monkeypatch):
    """SSE chat completions should capture after the stream finishes."""
    monkeypatch.setattr(settings, "enable_transcript_capture", True)
    monkeypatch.setattr(settings, "finetune_log_path", str(tmp_path / "finetune.jsonl"))
    monkeypatch.setattr(settings, "rag_improvement_log_path", str(tmp_path / "rag.jsonl"))
    monkeypatch.setattr(settings, "enable_cognitive_pipeline", False)

    async def fake_hybrid(_query, limit, score_threshold=None, no_cache=False):
        return []

    mock_response = _streaming_upstream_response(
        [
            b'data: {"choices":[{"delta":{"role":"assistant","content":"Hel"}}]}\n\n',
            b'data: {"choices":[{"delta":{"content":"lo"},"finish_reason":"stop"}]}\n\n',
            b"data: [DONE]\n\n",
        ]
    )
    mock_client = pooled_client_mock(mock_response)

    with patch("rag_proxy.pipeline_stages.hybrid_search", fake_hybrid):
        with patch("rag_proxy.upstream_client.httpx.AsyncClient", return_value=mock_client):
            with TestClient(app) as client:
                resp = client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "chat-demo",
                        "stream": True,
                        "messages": [{"role": "user", "content": "Say hello"}],
                    },
                )

    assert resp.status_code == 200
    assert resp.content.endswith(b"data: [DONE]\n\n")
    ft = json.loads((tmp_path / "finetune.jsonl").read_text(encoding="utf-8").strip())
    rag = json.loads((tmp_path / "rag.jsonl").read_text(encoding="utf-8").strip())
    assert ft["assistant"] == {"role": "assistant", "content": "Hello"}
    assert ft["stream"] is True
    assert rag["assistant_text"] == "Hello"


async def _async_iter(items):
    for item in items:
        yield item


def _streaming_upstream_response(chunks: list[bytes]):
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "text/event-stream; charset=utf-8"}
    mock_response.aiter_bytes = MagicMock(return_value=_async_iter(chunks))
    mock_response.aread = AsyncMock()
    mock_response.aclose = AsyncMock()
    return mock_response

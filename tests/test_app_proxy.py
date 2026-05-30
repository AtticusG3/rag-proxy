"""Proxy route uses shared upstream client without per-request client allocation."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from rag_proxy import upstream_client as uc
from rag_proxy.app import app


def _buffered_upstream_response(body: bytes = b'{"ok":true}') -> AsyncMock:
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "application/json"}
    mock_response.aread = AsyncMock(return_value=body)
    mock_response.aclose = AsyncMock()
    return mock_response


def _streaming_upstream_response(chunks: list[bytes]) -> AsyncMock:
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "text/event-stream; charset=utf-8"}
    mock_response.aiter_bytes = MagicMock(return_value=_async_iter(chunks))
    mock_response.aread = AsyncMock()
    mock_response.aclose = AsyncMock()
    return mock_response


def _pooled_client_mock(send_response: AsyncMock) -> MagicMock:
    mock_client = MagicMock()
    mock_client.build_request = MagicMock(return_value="req")
    mock_client.send = AsyncMock(return_value=send_response)
    mock_client.aclose = AsyncMock()
    return mock_client


def _assert_send_stream_true(mock_client: MagicMock) -> None:
    for call in mock_client.send.await_args_list:
        assert call.kwargs.get("stream") is True


def test_proxy_shared_client_two_gets_one_async_client_ctor():
    """Two polls in one session must construct one httpx.AsyncClient for the pool."""

    mock_response = _buffered_upstream_response()
    mock_client = _pooled_client_mock(mock_response)

    with patch("rag_proxy.upstream_client.httpx.AsyncClient", return_value=mock_client) as ctor:
        with TestClient(app) as client:
            r1 = client.get("/v1/models")
            r2 = client.get("/v1/models")

    assert ctor.call_count == 1
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r1.json() == {"ok": True}
    assert r2.json() == {"ok": True}
    assert mock_client.send.await_count == 2
    _assert_send_stream_true(mock_client)
    mock_client.aclose.assert_awaited_once()


def test_proxy_buffered_returns_body_and_closes_response():
    """Non-SSE upstream responses must return body and release the upstream socket."""

    mock_response = _buffered_upstream_response()
    mock_client = _pooled_client_mock(mock_response)

    with patch("rag_proxy.upstream_client.httpx.AsyncClient", return_value=mock_client):
        with TestClient(app) as client:
            resp = client.get("/v1/models")

    assert resp.status_code == 200
    assert resp.json() == {"ok": True}
    mock_response.aread.assert_awaited_once()
    mock_response.aclose.assert_awaited_once()


def test_proxy_closes_upstream_when_send_raises():
    """Upstream cleanup must run when proxy forwarding fails mid-request."""

    mock_response = _buffered_upstream_response()
    mock_client = _pooled_client_mock(mock_response)
    mock_client.send = AsyncMock(side_effect=RuntimeError("upstream send failed"))

    with patch("rag_proxy.upstream_client.httpx.AsyncClient", return_value=mock_client):
        with patch("rag_proxy.app.close_upstream_response", AsyncMock()) as close_mock:
            with TestClient(app, raise_server_exceptions=False) as client:
                resp = client.get("/v1/models")

    assert resp.status_code == 500
    close_mock.assert_awaited_once_with(None)


def test_proxy_sse_streams_via_real_relay_upstream():
    """SSE must flow through production relay_upstream, not buffered aread."""

    chunks = [b"data: ", b"hello\n\n"]
    mock_response = _streaming_upstream_response(chunks)
    mock_client = _pooled_client_mock(mock_response)

    with patch("rag_proxy.upstream_client.httpx.AsyncClient", return_value=mock_client) as ctor:
        with TestClient(app) as client:
            resp = client.get("/v1/chat/completions")

    assert ctor.call_count == 1
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers.get("content-type", "")
    assert resp.content == b"".join(chunks)
    mock_response.aread.assert_not_called()
    mock_response.aclose.assert_awaited()
    _assert_send_stream_true(mock_client)


def test_lifespan_starts_janitor_and_closes_pool_on_shutdown():
    """App lifespan must start one pool + janitor and tear both down on exit."""

    mock_client = MagicMock()
    mock_client.aclose = AsyncMock()

    with patch("rag_proxy.upstream_client.httpx.AsyncClient", return_value=mock_client) as ctor:
        with TestClient(app):
            assert ctor.call_count == 1
            assert uc._janitor_task is not None
            assert not uc._janitor_task.done()

    mock_client.aclose.assert_awaited_once()
    assert uc._upstream_client is None
    assert uc._janitor_task is None


async def _async_iter(items):
    for item in items:
        yield item

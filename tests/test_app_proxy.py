"""Proxy route uses shared upstream client without per-request client allocation."""

from __future__ import annotations

import asyncio
import gzip
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

from conftest import pooled_ctor_side_effect
from rag_proxy import upstream_client as uc
from rag_proxy.app import app, proxy
from rag_proxy.config import settings


def _buffered_upstream_response(body: bytes = b'{"ok":true}') -> AsyncMock:
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "application/json", "content-length": str(len(body))}
    mock_response.aread = AsyncMock(return_value=body)
    mock_response.aiter_bytes = MagicMock(side_effect=lambda: _async_iter([body]))
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

    with patch(
        "rag_proxy.upstream_client.httpx.AsyncClient",
        side_effect=pooled_ctor_side_effect(mock_client),
    ) as ctor:
        with TestClient(app) as client:
            r1 = client.get("/v1/models")
            r2 = client.get("/v1/models")

    assert ctor.call_count == 3
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

    with patch(
        "rag_proxy.upstream_client.httpx.AsyncClient",
        side_effect=pooled_ctor_side_effect(mock_client),
    ):
        with TestClient(app) as client:
            resp = client.get("/v1/models")

    assert resp.status_code == 200
    assert resp.json() == {"ok": True}
    mock_response.aiter_bytes.assert_called_once()
    mock_response.aclose.assert_awaited_once()


def test_proxy_closes_buffered_upstream_when_read_is_cancelled():
    """Cancel during a non-SSE body read must release the upstream socket.

    asyncio.CancelledError is a BaseException. An except Exception handler
    skips it, and leaked responses fill UPSTREAM_MAX_CONNECTIONS.
    """

    async def _run() -> None:
        mock_response = _buffered_upstream_response()
        async def _cancelled_iter():
            raise asyncio.CancelledError()
            yield b""

        mock_response.aiter_bytes = MagicMock(return_value=_cancelled_iter())
        mock_client = _pooled_client_mock(mock_response)

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": "/v1/models",
            "raw_path": b"/v1/models",
            "query_string": b"",
            "headers": [],
            "client": ("127.0.0.1", 1234),
            "server": ("127.0.0.1", 8088),
        }
        request = Request(scope, receive)
        with patch(
            "rag_proxy.app.ensure_upstream_client",
            AsyncMock(return_value=mock_client),
        ):
            with pytest.raises(asyncio.CancelledError):
                await proxy(request, "v1/models")
        mock_response.aclose.assert_awaited()

    asyncio.run(_run())


def test_proxy_closes_upstream_when_send_raises():
    """Upstream cleanup must run when proxy forwarding fails mid-request."""

    mock_response = _buffered_upstream_response()
    mock_client = _pooled_client_mock(mock_response)
    mock_client.send = AsyncMock(side_effect=RuntimeError("upstream send failed"))

    with patch(
        "rag_proxy.upstream_client.httpx.AsyncClient",
        side_effect=pooled_ctor_side_effect(mock_client),
    ):
        with patch("rag_proxy.app.close_upstream_response", AsyncMock()) as close_mock:
            with TestClient(app, raise_server_exceptions=False) as client:
                resp = client.get("/v1/models")

    assert resp.status_code == 500
    close_mock.assert_awaited_once_with(None)


def test_proxy_strips_content_encoding_on_buffered_passthrough():
    """gzip'd upstream bodies (llama.cpp UI) must drop Content-Encoding.

    httpx auto-decodes the upstream body before aread() returns, so the relayed
    body is always plaintext; keeping the header makes the client decompress
    plaintext bytes (net::ERR_CONTENT_DECODING_FAILED in the llama-swap web UI).
    transfer-encoding and content-length are hop-by-hop and must still be dropped.
    """

    # aread() returns the DECODED body (httpx decodes gzip before returning),
    # while the upstream headers still carry content-encoding.
    decoded = b"hello world"
    mock_response = _buffered_upstream_response(body=decoded)
    mock_response.headers = {
        "content-type": "application/javascript",
        "content-encoding": "gzip",
        "content-length": str(len(gzip.compress(decoded))),
        "transfer-encoding": "chunked",
    }
    mock_client = _pooled_client_mock(mock_response)

    with patch(
        "rag_proxy.upstream_client.httpx.AsyncClient",
        side_effect=pooled_ctor_side_effect(mock_client),
    ):
        with TestClient(app) as client:
            resp = client.get("/index.js")

    assert resp.status_code == 200
    # Header dropped because the relayed body is already decoded plaintext.
    assert resp.headers.get("content-encoding") is None
    assert resp.content == b"hello world"
    assert "transfer-encoding" not in resp.headers


def test_proxy_sse_streams_via_real_relay_upstream():
    """SSE must flow through production relay_upstream, not buffered aread."""

    # aiter_bytes() yields DECODED chunks (httpx decodes gzip before relaying),
    # while the upstream headers still carry content-encoding.
    decoded = b"data: hello\n\n"
    chunks = [decoded[:6], decoded[6:]]
    mock_response = _streaming_upstream_response(chunks)
    mock_response.headers = {
        "content-type": "text/event-stream; charset=utf-8",
        "content-encoding": "gzip",
        "transfer-encoding": "chunked",
    }
    mock_client = _pooled_client_mock(mock_response)

    with patch(
        "rag_proxy.upstream_client.httpx.AsyncClient",
        side_effect=pooled_ctor_side_effect(mock_client),
    ) as ctor:
        with TestClient(app) as client:
            resp = client.get("/v1/chat/completions")

    assert ctor.call_count == 3
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers.get("content-type", "")
    assert resp.headers.get("content-encoding") is None
    assert resp.content == b"data: hello\n\n"
    assert "transfer-encoding" not in resp.headers
    mock_response.aread.assert_not_called()
    mock_response.aclose.assert_awaited()
    _assert_send_stream_true(mock_client)


def test_proxy_streams_chunked_text_plain_without_buffering():
    """A chunked text/plain body (log tail) must be relayed, not read to the end."""

    async def _never_ending():
        yield b"line 1\n"
        yield b"line 2\n"
        # Reading past here would hang a buffering proxy; the test client stops early.
        await asyncio.sleep(3600)

    mock_response = _streaming_upstream_response([])
    mock_response.headers = {"content-type": "text/plain", "transfer-encoding": "chunked"}
    mock_response.aiter_bytes = MagicMock(return_value=_never_ending())
    mock_client = _pooled_client_mock(mock_response)

    async def _run() -> list[bytes]:
        sent_body = False

        async def receive():
            nonlocal sent_body
            if not sent_body:
                sent_body = True
                return {"type": "http.request", "body": b"", "more_body": False}
            await asyncio.sleep(3600)

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": "/logs/stream",
            "raw_path": b"/logs/stream",
            "query_string": b"",
            "headers": [],
            "client": ("127.0.0.1", 1234),
            "server": ("127.0.0.1", 8088),
        }
        request = Request(scope, receive)
        with patch("rag_proxy.app.ensure_upstream_client", AsyncMock(return_value=mock_client)):
            resp = await asyncio.wait_for(proxy(request, "logs/stream"), timeout=2)
        assert resp.media_type == "text/plain"
        got = []
        it = resp.body_iterator
        got.append(await asyncio.wait_for(it.__anext__(), timeout=2))
        got.append(await asyncio.wait_for(it.__anext__(), timeout=2))
        # Client disconnect: closing the generator must release the upstream response.
        await it.aclose()
        return got

    chunks = asyncio.run(_run())

    assert chunks == [b"line 1\n", b"line 2\n"]
    mock_response.aread.assert_not_called()
    mock_response.aclose.assert_awaited()


_CHAT_REQ = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}


@pytest.mark.parametrize(
    ("label", "method", "path", "headers", "capture", "streamed"),
    [
        ("accel-buffering-no", "GET", "/events", {"content-type": "text/plain", "content-length": "9", "x-accel-buffering": "no"}, False, True),
        ("ndjson-with-length", "GET", "/feed", {"content-type": "application/x-ndjson", "content-length": "9"}, False, True),
        ("chunked-json-non-chat", "GET", "/api/items", {"content-type": "application/json", "transfer-encoding": "chunked"}, False, True),
        # Capture needs the whole body, so a captured non-SSE chat completion stays buffered.
        ("captured-chat-chunked", "POST", "/v1/chat/completions", {"content-type": "application/json", "transfer-encoding": "chunked"}, True, False),
        ("bounded-json", "GET", "/api/items", {"content-type": "application/json", "content-length": "9"}, False, False),
    ],
)
def test_proxy_stream_or_buffer_by_headers(label, method, path, headers, capture, streamed):
    """Each stream_it branch streams (aiter_bytes) or buffers (aread) as intended."""

    mock_response = _buffered_upstream_response(b"abc\ndef\n\n")
    mock_response.headers = headers
    mock_response.aiter_bytes = MagicMock(return_value=_async_iter([b"abc\n", b"def\n\n"]))
    mock_client = _pooled_client_mock(mock_response)
    captured = MagicMock()
    ctx = MagicMock(stream=False)

    with (
        patch("rag_proxy.upstream_client.httpx.AsyncClient", side_effect=pooled_ctor_side_effect(mock_client)),
        patch("rag_proxy.app.capture_enabled", return_value=capture),
        patch("rag_proxy.app.augment_chat_payload_with_context", AsyncMock(side_effect=lambda d, h: (d, ctx))),
        patch("rag_proxy.app.capture_chat_response", captured),
        TestClient(app) as client,
    ):
        resp = client.request(method, path, json=_CHAT_REQ if method == "POST" else None)

    assert resp.status_code == 200
    assert resp.content == b"abc\ndef\n\n"
    if streamed:
        mock_response.aread.assert_not_called()
    else:
        mock_response.aiter_bytes.assert_called()
    # The captured chat body must reach capture; this fails if the guard is dropped.
    assert captured.call_count == (1 if capture else 0)


@pytest.mark.parametrize(
    ("method", "status"),
    [("HEAD", 200), ("GET", 204), ("GET", 304)],
)
def test_proxy_bodyless_responses_stream_with_no_body(method, status):
    """HEAD, 204 and 304 have no Content-Length here, so they take the stream path safely."""

    mock_response = _streaming_upstream_response([])
    mock_response.status_code = status
    mock_response.headers = {"content-type": "text/html"}
    mock_client = _pooled_client_mock(mock_response)

    with (
        patch("rag_proxy.upstream_client.httpx.AsyncClient", side_effect=pooled_ctor_side_effect(mock_client)),
        TestClient(app) as client,
    ):
        resp = client.request(method, "/ui/")

    assert resp.status_code == status
    assert resp.content == b""
    mock_response.aread.assert_not_called()
    mock_response.aclose.assert_awaited()


def test_lifespan_starts_janitor_and_closes_pool_on_shutdown():
    """App lifespan must start one pool + janitor and tear both down on exit."""

    mock_client = MagicMock()
    mock_client.aclose = AsyncMock()

    with patch(
        "rag_proxy.upstream_client.httpx.AsyncClient",
        side_effect=pooled_ctor_side_effect(mock_client),
    ) as ctor:
        with TestClient(app):
            assert ctor.call_count == 3
            assert uc._janitor_task is not None
            assert not uc._janitor_task.done()

    assert mock_client.aclose.await_count == 1
    assert uc._upstream_client is None
    assert uc._janitor_task is None


class _SilentStream(httpx.AsyncByteStream):
    async def __aiter__(self):
        await asyncio.sleep(3600)
        yield b""


class _SilentTransport(httpx.AsyncBaseTransport):
    def __init__(self):
        self.request: httpx.Request | None = None

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.request = request
        return httpx.Response(
            200,
            headers={"content-type": "text/plain"},
            stream=_SilentStream(),
            request=request,
        )


def test_proxy_stream_survives_read_timeout(monkeypatch):
    """A silent non-SSE stream must not be cut by the configured read timeout."""

    monkeypatch.setattr(settings, "upstream_timeout_sec", 0.1)
    transport = _SilentTransport()
    client = httpx.AsyncClient(transport=transport, timeout=httpx.Timeout(0.1))

    async def _run() -> None:
        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": "/logs/stream",
            "raw_path": b"/logs/stream",
            "query_string": b"",
            "headers": [],
            "client": ("127.0.0.1", 1234),
            "server": ("127.0.0.1", 8088),
        }
        request = Request(scope, receive)
        with patch("rag_proxy.app.ensure_upstream_client", AsyncMock(return_value=client)):
            resp = await proxy(request, "logs/stream")
        # The request must carry read=None so a silent stream is not cut.
        assert transport.request is not None
        assert transport.request.extensions["timeout"]["read"] is None
        it = resp.body_iterator
        try:
            await asyncio.wait_for(it.__anext__(), timeout=0.2)
        except asyncio.TimeoutError:
            pass
        else:
            raise AssertionError("expected the silent stream to still be waiting")
        await it.aclose()
        await client.aclose()

    asyncio.run(_run())


def test_proxy_disconnect_releases_silent_stream():
    """A closed client on a silent stream must release the upstream and registry."""

    async def _silent():
        await asyncio.sleep(3600)
        yield b""

    async def _run() -> None:
        uc._stream_registry.clear()
        uc._idle_exempt_keys.clear()

        mock_response = _streaming_upstream_response([])
        mock_response.headers = {"content-type": "text/plain", "transfer-encoding": "chunked"}
        mock_response.aiter_bytes = MagicMock(return_value=_silent())
        mock_client = _pooled_client_mock(mock_response)

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": "/logs/stream",
            "raw_path": b"/logs/stream",
            "query_string": b"",
            "headers": [],
            "client": ("127.0.0.1", 1234),
            "server": ("127.0.0.1", 8088),
        }
        request = Request(scope, receive)
        with patch("rag_proxy.app.ensure_upstream_client", AsyncMock(return_value=mock_client)):
            resp = await proxy(request, "logs/stream")

        sent_disconnect = False

        async def receive_disconnect():
            nonlocal sent_disconnect
            if not sent_disconnect:
                sent_disconnect = True
                await asyncio.sleep(0.05)
                return {"type": "http.disconnect"}
            await asyncio.sleep(3600)

        async def send(_message):
            pass

        await asyncio.wait_for(resp(scope, receive_disconnect, send), timeout=2)
        mock_response.aclose.assert_awaited()
        assert uc.upstream_active_stream_count() == 0

    asyncio.run(_run())


async def _async_iter(items):
    for item in items:
        yield item


def test_proxy_bounds_header_wait_when_backend_never_answers():
    """A backend that accepts the connection but never sends headers must not hang proxy().

    read=None keeps the streaming body open, but httpx's read timeout also covers
    waiting for response headers. Without its own bound on client.send(), a silent
    backend holds the request open forever, so the header wait is timed out.
    """

    async def _run() -> None:
        async def _never_send(*args, **kwargs):
            await asyncio.Event().wait()

        mock_client = MagicMock()
        mock_client.build_request = MagicMock(return_value="req")
        mock_client.send = _never_send
        mock_client.aclose = AsyncMock()

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": "/v1/models",
            "raw_path": b"/v1/models",
            "query_string": b"",
            "headers": [],
            "client": ("127.0.0.1", 1234),
            "server": ("127.0.0.1", 8088),
        }
        request = Request(scope, receive)
        started = asyncio.get_running_loop().time()
        with (
            patch("rag_proxy.app.ensure_upstream_client", AsyncMock(return_value=mock_client)),
            patch.object(settings, "upstream_timeout_sec", 0.05),
        ):
            with pytest.raises(httpx.ReadTimeout):
                await asyncio.wait_for(proxy(request, "v1/models"), timeout=2)
        elapsed = asyncio.get_running_loop().time() - started
        # The 2 s guard must not be what fired; the header bound must be.
        assert elapsed < 1.0, f"header wait was not bounded by upstream_timeout_sec ({elapsed:.2f}s)"

    asyncio.run(_run())


def test_bounded_body_timeout_applies_per_chunk_not_total():
    """A bounded body may take longer than the read timeout in total, as long as each
    chunk arrives within it. A stalled chunk must still time out."""

    async def _run() -> None:
        body = b"abcdefghij"

        def _slow_but_steady():
            async def _gen():
                for i in range(0, len(body), 4):
                    await asyncio.sleep(0.03)
                    yield body[i : i + 4]
            return _gen()

        mock_response = _buffered_upstream_response(body)
        mock_response.aiter_bytes = MagicMock(side_effect=_slow_but_steady)
        mock_client = _pooled_client_mock(mock_response)
        with (
            patch("rag_proxy.upstream_client.httpx.AsyncClient", side_effect=pooled_ctor_side_effect(mock_client)),
            patch.object(settings, "upstream_timeout_sec", 0.05),
            TestClient(app) as client,
        ):
            resp = client.get("/api/items")
        assert resp.status_code == 200
        assert resp.content == body

        def _stalled():
            async def _gen():
                yield b"start"
                await asyncio.Event().wait()
                yield b"never"
            return _gen()

        mock_response2 = _buffered_upstream_response(body)
        mock_response2.aiter_bytes = MagicMock(side_effect=_stalled)
        mock_client2 = _pooled_client_mock(mock_response2)
        with (
            patch("rag_proxy.upstream_client.httpx.AsyncClient", side_effect=pooled_ctor_side_effect(mock_client2)),
            patch.object(settings, "upstream_timeout_sec", 0.05),
            TestClient(app, raise_server_exceptions=False) as client,
        ):
            resp2 = client.get("/api/items")
        assert resp2.status_code == 500

    asyncio.run(_run())


def test_bounded_body_chunk_timeout_raises_httpx_read_timeout():
    """A stalled bounded body must surface as httpx.ReadTimeout, not asyncio.TimeoutError.

    The buffered path waits per chunk with asyncio.wait_for, which raises
    asyncio.TimeoutError. Re-raise the httpx type so callers keep the module's
    exception contract instead of seeing an unhandled 500.
    """

    async def _run() -> None:
        def _stalled():
            async def _gen():
                yield b"start"
                await asyncio.Event().wait()
                yield b"never"
            return _gen()

        mock_response = _buffered_upstream_response(b"abcdefghij")
        mock_response.aiter_bytes = MagicMock(side_effect=_stalled)
        mock_client = _pooled_client_mock(mock_response)

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": "/api/items",
            "raw_path": b"/api/items",
            "query_string": b"",
            "headers": [],
            "client": ("127.0.0.1", 1234),
            "server": ("127.0.0.1", 8088),
        }
        request = Request(scope, receive)
        with (
            patch("rag_proxy.app.ensure_upstream_client", AsyncMock(return_value=mock_client)),
            patch.object(settings, "upstream_timeout_sec", 0.05),
        ):
            with pytest.raises(httpx.ReadTimeout):
                await asyncio.wait_for(proxy(request, "api/items"), timeout=2)

    asyncio.run(_run())

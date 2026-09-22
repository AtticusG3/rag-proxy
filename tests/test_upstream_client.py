"""Upstream client pooling and abandoned-stream cleanup."""

from __future__ import annotations

import asyncio
import contextlib
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rag_proxy import upstream_client as uc


def test_buffered_upstream_closes_response_not_shared_client():
    """One-shot polls must release upstream sockets without tearing down the pool."""

    async def _run() -> None:
        closed = []

        class FakeResponse:
            is_closed = False

            async def aclose(self):
                self.is_closed = True
                closed.append("response")

        class FakeClient:
            is_closed = False

            async def aclose(self):
                self.is_closed = True
                closed.append("client")

        uc._upstream_client = FakeClient()
        resp = FakeResponse()
        await uc.close_upstream_response(resp)  # type: ignore[arg-type]

        assert resp.is_closed
        assert uc._upstream_client.is_closed is False
        assert closed == ["response"]

    asyncio.run(_run())


def test_relay_upstream_closes_when_client_disconnects():
    """Abandoned SSE streams must not leak upstream FDs when the client goes away."""

    async def _run() -> None:
        chunks = [b"a", b"b", b"c"]
        upstream = AsyncMock()
        upstream.aiter_bytes = MagicMock(return_value=_async_iter(chunks))

        request = AsyncMock()
        request.is_disconnected = AsyncMock(side_effect=[False, True])

        out = []
        async for part in uc.relay_upstream(request, upstream):
            out.append(part)

        assert out == [b"a"]
        upstream.aclose.assert_awaited_once()
        assert not uc._stream_registry

    asyncio.run(_run())


def test_janitor_reaps_idle_registered_streams():
    """Streams idle past UPSTREAM_STREAM_ABANDON_SEC are closed by the janitor."""

    async def _run() -> None:
        upstream = AsyncMock()
        key = await uc.register_stream(upstream)
        uc._stream_registry[key] = (upstream, time.monotonic() - 9999, True)

        closed = await uc.reap_abandoned_streams()

        assert closed == 1
        upstream.aclose.assert_awaited_once()
        assert key not in uc._stream_registry

    asyncio.run(_run())


def test_janitor_does_not_reap_before_first_byte():
    """Cold llama-swap (no SSE yet) must survive the inter-chunk idle limit.

    Time-to-first-token is not abandonment. Reaping at UPSTREAM_STREAM_ABANDON_SEC
    kills a live generation during model load.
    """

    async def _run() -> None:
        upstream = AsyncMock()
        key = await uc.register_stream(upstream)
        uc._stream_registry[key] = (upstream, time.monotonic() - 130, False)

        with patch.object(uc.settings, "upstream_stream_abandon_sec", 120):
            with patch.object(uc.settings, "upstream_first_byte_sec", 600):
                closed = await uc.reap_abandoned_streams()

        assert closed == 0
        upstream.aclose.assert_not_called()
        assert key in uc._stream_registry

    asyncio.run(_run())


def test_janitor_reaps_after_bytes_once_inter_chunk_idle_elapses():
    """After the first byte, silence past UPSTREAM_STREAM_ABANDON_SEC is abandoned."""

    async def _run() -> None:
        upstream = AsyncMock()
        key = await uc.register_stream(upstream)
        uc._stream_registry[key] = (upstream, time.monotonic() - 130, True)

        with patch.object(uc.settings, "upstream_stream_abandon_sec", 120):
            with patch.object(uc.settings, "upstream_first_byte_sec", 600):
                closed = await uc.reap_abandoned_streams()

        assert closed == 1
        upstream.aclose.assert_awaited_once()
        assert key not in uc._stream_registry

    asyncio.run(_run())


def test_janitor_keeps_active_long_streams():
    """Long-lived SSE with recent activity must not be closed by idle reaper."""

    async def _run() -> None:
        upstream = AsyncMock()
        key = await uc.register_stream(upstream)
        await uc.touch_stream(key)

        closed = await uc.reap_abandoned_streams()

        assert closed == 0
        upstream.aclose.assert_not_called()
        assert key in uc._stream_registry

    asyncio.run(_run())


def test_ensure_upstream_client_requires_lifespan_startup():
    """Proxy must start the pool via lifespan; ensure does not lazy-init."""

    async def _run() -> None:
        with pytest.raises(RuntimeError, match="upstream client not started"):
            await uc.ensure_upstream_client()

    asyncio.run(_run())


def test_janitor_loop_reaps_abandoned_streams():
    """Background janitor must call reap on idle streams without direct test hooks."""

    async def _run() -> None:
        upstream = AsyncMock()
        real_sleep = uc.asyncio.sleep

        async def yield_sleep(_delay: float) -> None:
            await real_sleep(0)

        with patch.object(uc.settings, "upstream_idle_sweep_sec", 0.05):
            with patch.object(uc.settings, "upstream_stream_abandon_sec", 0.01):
                with patch.object(uc.asyncio, "sleep", side_effect=yield_sleep):
                    await uc.startup_upstream_client()
                    try:
                        key = await uc.register_stream(upstream)
                        uc._stream_registry[key] = (upstream, time.monotonic() - 9999, True)
                        deadline = time.monotonic() + 1.0
                        while key in uc._stream_registry and time.monotonic() < deadline:
                            await asyncio.sleep(0)
                        upstream.aclose.assert_awaited()
                        assert key not in uc._stream_registry
                    finally:
                        await uc.shutdown_upstream_client()

    asyncio.run(_run())


def test_relay_upstream_throttles_stream_registry_touch():
    """SSE must not take the registry lock on every chunk (hot-path churn)."""

    async def _run() -> None:
        chunks = [b"a", b"b", b"c"]
        upstream = AsyncMock()
        upstream.aiter_bytes = MagicMock(return_value=_async_iter(chunks))

        request = AsyncMock()
        request.is_disconnected = AsyncMock(return_value=False)

        reg_key = id(upstream)
        # register, last_touch init, per-chunk now checks, touch_stream update on 3rd chunk
        times = iter([0.0, 0.0, 0.1, 0.2, 1.5, 1.5])
        last = [1.5]

        def fake_monotonic() -> float:
            with contextlib.suppress(StopIteration):
                last[0] = next(times)
            return last[0]

        with patch.object(uc.time, "monotonic", side_effect=fake_monotonic):
            out = []
            async for part in uc.relay_upstream(request, upstream):
                out.append(part)
                ts = uc._stream_registry[reg_key][1]
                if part == b"a":
                    assert ts == 0.0
                elif part == b"b":
                    assert ts == 0.0
                elif part == b"c":
                    assert ts == 1.5

        assert out == chunks

    asyncio.run(_run())


def test_relay_upstream_capture_reports_complete_body_and_closes():
    """Capture tee must not change streamed bytes or leak the upstream response."""

    async def _run() -> None:
        chunks = [b"data: a\n\n", b"data: b\n\n"]
        upstream = AsyncMock()
        upstream.aiter_bytes = MagicMock(return_value=_async_iter(chunks))

        request = AsyncMock()
        request.is_disconnected = AsyncMock(return_value=False)

        captured = []

        async def on_complete(body: bytes) -> None:
            captured.append(body)

        out = []
        async for part in uc.relay_upstream_capture(request, upstream, on_complete):
            out.append(part)

        assert out == chunks
        assert captured == [b"".join(chunks)]
        upstream.aclose.assert_awaited_once()
        assert not uc._stream_registry

    asyncio.run(_run())


def test_sidecar_pools_keepalive_when_upstream_keepalive_is_zero(monkeypatch):
    """Embed and Qdrant are hit every chat turn; upstream keepalive 0 must not apply.

    UPSTREAM_MAX_KEEPALIVE=0 exists so llama-swap one-shot polls do not hold
    sockets. Reusing it on sidecar pools opens a new TCP connection per retrieval.
    """
    from rag_proxy import sidecar_client as sc

    monkeypatch.setattr(uc.settings, "upstream_max_keepalive", 0)
    monkeypatch.setattr(uc.settings, "sidecar_max_keepalive", 20)
    monkeypatch.setattr(uc.settings, "upstream_max_connections", 50)
    sc._embed_client = None
    sc._qdrant_client = None
    sc._sparse_client = None
    sc._turbovec_client = None
    sc._reranker_client = None
    limits: list[object] = []

    def ctor(*_args, **kwargs):
        limits.append(kwargs.get("limits"))
        client = MagicMock()
        client.aclose = AsyncMock()
        return client

    async def _run() -> None:
        with patch("rag_proxy.sidecar_client.httpx.AsyncClient", side_effect=ctor):
            await sc.startup_sidecar_clients()
            await sc.shutdown_sidecar_clients()

    asyncio.run(_run())
    assert limits
    assert all(item is not None and item.max_keepalive_connections == 20 for item in limits)
    assert all(item.max_connections == 50 for item in limits)


async def _async_iter(items):
    for item in items:
        yield item

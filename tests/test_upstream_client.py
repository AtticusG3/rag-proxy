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
        uc._stream_registry[key] = (upstream, time.monotonic() - 9999)

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
                        uc._stream_registry[key] = (upstream, time.monotonic() - 9999)
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


async def _async_iter(items):
    for item in items:
        yield item

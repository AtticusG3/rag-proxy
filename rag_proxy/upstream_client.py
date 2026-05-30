"""Shared upstream httpx client, streaming relay, and idle connection janitor."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import AsyncGenerator

import httpx
from starlette.requests import Request

from rag_proxy.config import settings

log = logging.getLogger("rag-proxy")

_STREAM_TOUCH_INTERVAL_SEC = 1.0

_upstream_client: httpx.AsyncClient | None = None
_janitor_task: asyncio.Task[None] | None = None
_stream_registry: dict[int, tuple[httpx.Response, float]] = {}
_stream_registry_lock: asyncio.Lock | None = None


def _registry_lock() -> asyncio.Lock:
    global _stream_registry_lock
    if _stream_registry_lock is None:
        _stream_registry_lock = asyncio.Lock()
    return _stream_registry_lock


def _build_limits() -> httpx.Limits:
    return httpx.Limits(
        max_connections=settings.upstream_max_connections,
        max_keepalive_connections=settings.upstream_max_keepalive,
        keepalive_expiry=settings.upstream_keepalive_expiry_sec,
    )


def get_upstream_client() -> httpx.AsyncClient:
    if _upstream_client is None:
        raise RuntimeError("upstream client not started")
    return _upstream_client


async def ensure_upstream_client() -> httpx.AsyncClient:
    """Return the lifespan-started client; does not lazy-init."""
    return get_upstream_client()


async def startup_upstream_client() -> None:
    global _upstream_client, _janitor_task
    if _upstream_client is not None:
        return
    _upstream_client = httpx.AsyncClient(
        timeout=httpx.Timeout(settings.upstream_timeout_sec),
        limits=_build_limits(),
    )
    _janitor_task = asyncio.create_task(_upstream_connection_janitor())
    log.info(
        "upstream pool max_conn=%s keepalive=%s expiry=%ss sweep=%ss",
        settings.upstream_max_connections,
        settings.upstream_max_keepalive,
        settings.upstream_keepalive_expiry_sec,
        settings.upstream_idle_sweep_sec,
    )


async def shutdown_upstream_client() -> None:
    global _upstream_client, _janitor_task
    if _janitor_task is not None:
        _janitor_task.cancel()
        try:
            await _janitor_task
        except asyncio.CancelledError:
            pass
        _janitor_task = None
    async with _registry_lock():
        streams = list(_stream_registry.values())
        _stream_registry.clear()
    for resp, _ in streams:
        await close_upstream_response(resp)
    if _upstream_client is not None:
        await _upstream_client.aclose()
        _upstream_client = None


async def close_upstream_response(response: httpx.Response | None) -> None:
    if response is None:
        return
    try:
        await response.aclose()
    except Exception:
        log.debug("upstream response close failed", exc_info=True)


async def reap_abandoned_streams() -> int:
    """Close upstream streams idle longer than UPSTREAM_STREAM_ABANDON_SEC."""
    cutoff = time.monotonic() - settings.upstream_stream_abandon_sec
    closed = 0
    async with _registry_lock():
        stale_keys = [
            key for key, (_, last_activity) in _stream_registry.items() if last_activity < cutoff
        ]
        stale_entries = [
            _stream_registry.pop(key) for key in stale_keys if key in _stream_registry
        ]
    for resp, _ in stale_entries:
        await close_upstream_response(resp)
        closed += 1
    if closed:
        log.info("janitor: closed %s idle upstream stream(s)", closed)
    return closed


async def _upstream_connection_janitor() -> None:
    while True:
        try:
            await asyncio.sleep(settings.upstream_idle_sweep_sec)
        except asyncio.CancelledError:
            raise
        if _upstream_client is None:
            continue
        await reap_abandoned_streams()


async def register_stream(response: httpx.Response) -> int:
    key = id(response)
    now = time.monotonic()
    async with _registry_lock():
        _stream_registry[key] = (response, now)
    return key


async def touch_stream(key: int) -> None:
    async with _registry_lock():
        entry = _stream_registry.get(key)
        if entry is not None:
            response, _ = entry
            _stream_registry[key] = (response, time.monotonic())


async def unregister_stream(key: int) -> None:
    async with _registry_lock():
        _stream_registry.pop(key, None)


async def relay_upstream(
    request: Request,
    upstream: httpx.Response,
) -> AsyncGenerator[bytes, None]:
    reg_key = await register_stream(upstream)
    last_touch = time.monotonic()
    try:
        async for chunk in upstream.aiter_bytes():
            now = time.monotonic()
            if now - last_touch >= _STREAM_TOUCH_INTERVAL_SEC:
                await touch_stream(reg_key)
                last_touch = now
            if await request.is_disconnected():
                log.debug("client disconnected; closing upstream stream")
                break
            yield chunk
    finally:
        await close_upstream_response(upstream)
        await unregister_stream(reg_key)

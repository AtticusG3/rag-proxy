"""Shared httpx clients for embed and reranker sidecars."""

from __future__ import annotations

import logging

import httpx

from rag_proxy.config import settings

log = logging.getLogger("rag-proxy")

_embed_client: httpx.AsyncClient | None = None
_reranker_client: httpx.AsyncClient | None = None


def _build_limits() -> httpx.Limits:
    return httpx.Limits(
        max_connections=settings.upstream_max_connections,
        max_keepalive_connections=settings.upstream_max_keepalive,
        keepalive_expiry=settings.upstream_keepalive_expiry_sec,
    )


def _needs_embed_pool() -> bool:
    return settings.enable_cognitive_pipeline


def _needs_reranker_pool() -> bool:
    return settings.enable_cognitive_pipeline and settings.enable_reranker


def get_embed_client() -> httpx.AsyncClient:
    if _embed_client is None:
        raise RuntimeError("embed sidecar client not started")
    return _embed_client


def get_reranker_client() -> httpx.AsyncClient:
    if _reranker_client is None:
        raise RuntimeError("reranker sidecar client not started")
    return _reranker_client


async def startup_sidecar_clients() -> None:
    global _embed_client, _reranker_client
    if _needs_embed_pool() and _embed_client is None:
        _embed_client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0),
            limits=_build_limits(),
        )
        log.info("embed sidecar pool started url=%s", settings.embed_url)
    if _needs_reranker_pool() and _reranker_client is None:
        timeout_sec = settings.rerank_timeout_ms / 1000.0 + 0.5
        _reranker_client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_sec),
            limits=_build_limits(),
        )
        log.info("reranker sidecar pool started url=%s", settings.reranker_url)


async def shutdown_sidecar_clients() -> None:
    global _embed_client, _reranker_client
    if _embed_client is not None:
        await _embed_client.aclose()
        _embed_client = None
    if _reranker_client is not None:
        await _reranker_client.aclose()
        _reranker_client = None

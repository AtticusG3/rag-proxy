"""Settings-backed async retrieval HTTP (embed, dense, sparse)."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time

import httpx

from rag_proxy.clients.retrieval_core import (
    dense_search_payload,
    embed_input_too_large,
    embed_payload,
    parse_dense_hits,
    parse_embedding,
    parse_sparse_hits,
    prepare_embed_text,
    sparse_search_payload,
)
from ingest.embed_lifecycle import ensure_embed_urls, touch_embed_activity
from rag_proxy.config import settings
from rag_proxy.observability import record_embed_cache_hit, record_embed_cache_miss
from rag_proxy.sidecar_client import get_embed_client, get_qdrant_client, get_sparse_client

log = logging.getLogger("rag-proxy")

_embed_cache: dict[str, tuple[list[float], float]] = {}
_EMBED_CACHE_TTL = 600.0


def embed_cache_stats() -> dict[str, int | float | bool]:
    return {
        "enabled": settings.enable_embed_cache,
        "entries": len(_embed_cache),
        "ttl_sec": _EMBED_CACHE_TTL,
    }


def _cache_key(text: str) -> str:
    return hashlib.sha256(text.strip().encode()).hexdigest()


async def get_embedding(text: str) -> list[float] | None:
    """Embed text via the standalone nomic-embed server."""
    ensure_embed_urls([settings.embed_url.rstrip("/")])
    char_limits = [settings.embed_max_chars]
    if settings.embed_max_chars > 1200:
        char_limits.append(1200)

    client = get_embed_client()
    for max_chars in char_limits:
        chunk = prepare_embed_text(text, max_chars)
        payload = embed_payload(chunk)
        saw_too_large = False

        for attempt in range(settings.embed_retries):
            if attempt:
                await asyncio.sleep(0.5)
            try:
                r = await client.post(
                    f"{settings.embed_url}/v1/embeddings",
                    json=payload,
                )
                body = r.text or ""
                if r.status_code >= 500:
                    if embed_input_too_large(body):
                        saw_too_large = True
                        break
                    if attempt + 1 < settings.embed_retries:
                        log.warning(
                            f"Embedding HTTP {r.status_code}, retry {attempt + 2}/"
                            f"{settings.embed_retries}: {body[:200]!r}"
                        )
                        continue
                r.raise_for_status()
                touch_embed_activity()
                return parse_embedding(r.json())
            except httpx.HTTPStatusError as e:
                body = (e.response.text or "")[:200]
                if embed_input_too_large(e.response.text or ""):
                    saw_too_large = True
                    break
                if e.response.status_code >= 500 and attempt + 1 < settings.embed_retries:
                    log.warning(
                        f"Embedding HTTP {e.response.status_code}, retry: {body!r}"
                    )
                    continue
                log.warning(f"Embedding failed HTTP {e.response.status_code}: {body!r}")
                return None
            except Exception as e:
                if attempt + 1 < settings.embed_retries:
                    log.warning(f"Embedding failed, retry: {e}")
                    continue
                log.warning(f"Embedding failed: {e}")
                return None

        if not saw_too_large:
            return None

    return None


async def embed_text(
    text: str,
    no_cache: bool = False,
    cache_hits: list[str] | None = None,
) -> list[float] | None:
    if settings.enable_embed_cache and not no_cache:
        key = _cache_key(text)
        now = time.time()
        cached = _embed_cache.get(key)
        if cached and now - cached[1] < _EMBED_CACHE_TTL:
            record_embed_cache_hit()
            if cache_hits is not None:
                cache_hits.append("embed")
            return cached[0]
        record_embed_cache_miss()
        vector = await get_embedding(text)
        if vector is not None:
            _embed_cache[key] = (vector, now)
        return vector
    return await get_embedding(text)


async def search_qdrant_dense(
    vector: list[float],
    limit: int | None = None,
    score_threshold: float | None = None,
) -> list[dict]:
    """Return top-k chunks from Qdrant above the similarity threshold."""
    limit = limit if limit is not None else settings.top_k
    score_threshold = (
        score_threshold if score_threshold is not None else settings.similarity_threshold
    )
    body = dense_search_payload(vector, limit, score_threshold)
    try:
        client = get_qdrant_client()
        r = await client.post(
            f"{settings.qdrant_url}/collections/{settings.qdrant_collection}/points/search",
            json=body,
        )
        r.raise_for_status()
        return parse_dense_hits(r.json())
    except Exception as e:
        log.warning(f"Qdrant search failed: {e}")
        return []


async def sparse_search(query: str, limit: int) -> list[dict]:
    """Optional BM25 sidecar; fail-open to empty list."""
    if not settings.sparse_index_url:
        return []
    body = sparse_search_payload(query, limit, settings.qdrant_collection)
    try:
        client = get_sparse_client()
        r = await client.post(
            f"{settings.sparse_index_url.rstrip('/')}/search",
            json=body,
        )
        r.raise_for_status()
        return parse_sparse_hits(r.json())
    except Exception as e:
        log.warning(f"Sparse search failed: {e}")
        return []

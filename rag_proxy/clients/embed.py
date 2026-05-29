"""Embedding client with optional cache."""

from __future__ import annotations

import hashlib
import time

from rag_proxy.config import settings
from rag_proxy.legacy_rag import get_embedding

_embed_cache: dict[str, tuple[list[float], float]] = {}
_EMBED_CACHE_TTL = 600.0


def _cache_key(text: str) -> str:
    return hashlib.sha256(text.strip().encode()).hexdigest()


async def embed_text(text: str, no_cache: bool = False) -> list[float] | None:
    if settings.enable_embed_cache and not no_cache:
        key = _cache_key(text)
        now = time.time()
        cached = _embed_cache.get(key)
        if cached and now - cached[1] < _EMBED_CACHE_TTL:
            return cached[0]
        vector = await get_embedding(text)
        if vector is not None:
            _embed_cache[key] = (vector, now)
        return vector
    return await get_embedding(text)

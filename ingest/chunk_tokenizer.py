"""Tokenizer resolution and token counting for ingest chunking."""

from __future__ import annotations

import logging
from functools import lru_cache

from chonkie import RecursiveChunker

from ingest.chunk_config import (
    TOKENIZER_FALLBACKS,
    ChunkConfig,
    load_chunk_config,
)

log = logging.getLogger("ingest.chunking")

_PROBE_TEXT = "tokenizer probe text."


@lru_cache(maxsize=16)
def resolve_tokenizer(preferred: str) -> str:
    """Return the first working tokenizer; log loudly when not the configured one."""
    for candidate in (preferred, *TOKENIZER_FALLBACKS):
        try:
            RecursiveChunker(tokenizer=candidate, chunk_size=64)(_PROBE_TEXT)
        except Exception as exc:
            log.warning(
                "INGEST_CHUNK_TOKENIZER probe failed for %r: %s",
                candidate,
                exc,
            )
            continue
        if candidate != preferred:
            log.warning(
                "INGEST_CHUNK_TOKENIZER FALLBACK: configured=%s active=%s "
                "(chunk token counts may not match nomic-embed; install chonkie[tokenizers])",
                preferred,
                candidate,
            )
        return candidate
    log.error(
        "INGEST_CHUNK_TOKENIZER: all probes failed (configured=%s); using word last-resort",
        preferred,
    )
    return "word"


@lru_cache(maxsize=16)
def _counter_chunker(tokenizer: str) -> RecursiveChunker:
    return RecursiveChunker(tokenizer=tokenizer, chunk_size=10_000_000)


def count_tokens(text: str, tokenizer: str) -> int:
    """Count tokens for a text slice using the active ingest tokenizer."""
    if not text.strip():
        return 0
    chunks = _counter_chunker(tokenizer)(text)
    return sum(chunk.token_count for chunk in chunks)


def warmup_chunking(config: ChunkConfig | None = None) -> str:
    """Resolve tokenizer at startup and log the active ingest chunk profile."""
    cfg = config or load_chunk_config()
    active = resolve_tokenizer(cfg.tokenizer)
    if active == cfg.tokenizer:
        log.info(
            "ingest chunking ready: tokenizer=%s chunk_size=%s overlap=%s "
            "min_chunk_tokens=%s semantic=%s",
            active,
            cfg.chunk_size,
            cfg.chunk_overlap,
            cfg.min_chunk_tokens,
            cfg.semantic_enabled,
        )
    else:
        log.warning(
            "ingest chunking ready WITH TOKENIZER FALLBACK: configured=%s active=%s "
            "chunk_size=%s overlap=%s min_chunk_tokens=%s semantic=%s",
            cfg.tokenizer,
            active,
            cfg.chunk_size,
            cfg.chunk_overlap,
            cfg.min_chunk_tokens,
            cfg.semantic_enabled,
        )
    return active

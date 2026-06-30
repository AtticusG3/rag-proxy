"""Text chunking for ingest via Chonkie with per-document strategy selection."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from chonkie import Pipeline, RecursiveChunker, SentenceChunker, TokenChunker

from ingest.chunk_config import (
    TOKENIZER_FALLBACKS,
    ChunkConfig,
    DEFAULT_CHUNK_OVERLAP_TOKENS,
    DEFAULT_CHUNK_SIZE_TOKENS,
    load_chunk_config,
)
from ingest.chunking_strategy import ChunkContext, ChunkStrategy, select_chunk_strategy

log = logging.getLogger("ingest.chunking")

# Backward-compatible aliases (token counts, not characters).
DEFAULT_CHUNK_SIZE = DEFAULT_CHUNK_SIZE_TOKENS
DEFAULT_CHUNK_OVERLAP = DEFAULT_CHUNK_OVERLAP_TOKENS

_FALLBACK_CHAIN: dict[ChunkStrategy, tuple[ChunkStrategy, ...]] = {
    ChunkStrategy.SEMANTIC: (
        ChunkStrategy.SENTENCE,
        ChunkStrategy.RECURSIVE,
        ChunkStrategy.TOKEN,
    ),
    ChunkStrategy.CODE: (ChunkStrategy.RECURSIVE, ChunkStrategy.TOKEN),
    ChunkStrategy.SENTENCE: (ChunkStrategy.RECURSIVE, ChunkStrategy.TOKEN),
    ChunkStrategy.RECURSIVE: (ChunkStrategy.SENTENCE, ChunkStrategy.TOKEN),
    ChunkStrategy.TOKEN: (ChunkStrategy.RECURSIVE,),
}


@lru_cache(maxsize=1)
def _semantic_chunker_available() -> bool:
    try:
        from chonkie import SemanticChunker  # noqa: F401
    except ImportError:
        return False
    return True


@lru_cache(maxsize=1)
def _code_chunker_available() -> bool:
    try:
        from chonkie import CodeChunker  # noqa: F401
    except ImportError:
        return False
    return True


@lru_cache(maxsize=16)
def _resolved_tokenizer(preferred: str) -> str:
    from chonkie import RecursiveChunker

    for candidate in (preferred, *TOKENIZER_FALLBACKS):
        try:
            RecursiveChunker(tokenizer=candidate, chunk_size=64)("tokenizer probe text.")
            return candidate
        except Exception:
            continue
    return "word"


@lru_cache(maxsize=32)
def _cached_runner(
    strategy: str,
    tokenizer: str,
    chunk_size: int,
    chunk_overlap: int,
    semantic_model: str,
) -> Any:
    strat = ChunkStrategy(strategy)
    if strat is ChunkStrategy.SEMANTIC:
        from chonkie import SemanticChunker

        return SemanticChunker(
            tokenizer=tokenizer,
            embedding_model=semantic_model,
            chunk_size=chunk_size,
        )
    if strat is ChunkStrategy.CODE:
        from chonkie import CodeChunker

        return CodeChunker(tokenizer=tokenizer, chunk_size=chunk_size)
    if strat is ChunkStrategy.SENTENCE:
        return SentenceChunker(
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    if strat is ChunkStrategy.TOKEN:
        return TokenChunker(
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    if chunk_overlap > 0:
        return (
            Pipeline()
            .chunk_with("recursive", tokenizer=tokenizer, chunk_size=chunk_size)
            .refine_with("overlap", context_size=chunk_overlap)
        )
    return RecursiveChunker(tokenizer=tokenizer, chunk_size=chunk_size)


def _strategy_supported(strategy: ChunkStrategy, config: ChunkConfig) -> bool:
    if strategy is ChunkStrategy.SEMANTIC:
        return config.semantic_enabled and _semantic_chunker_available()
    if strategy is ChunkStrategy.CODE:
        return _code_chunker_available()
    return True


def _run_strategy(
    strategy: ChunkStrategy,
    text: str,
    *,
    tokenizer: str,
    config: ChunkConfig,
) -> list[str]:
    runner = _cached_runner(
        strategy.value,
        tokenizer,
        config.chunk_size,
        config.chunk_overlap,
        config.semantic_model,
    )
    if isinstance(runner, Pipeline):
        raw = [chunk.text for chunk in runner.run(texts=text).chunks]
    else:
        raw = [chunk.text for chunk in runner(text)]
    return [piece for piece in raw if piece.strip()]


def chunk_text(
    text: str,
    size: int | None = None,
    overlap: int | None = None,
    *,
    context: ChunkContext | None = None,
    strategy: ChunkStrategy | str | None = None,
    config: ChunkConfig | None = None,
) -> list[str]:
    """Split text into chunks using a Chonkie strategy selected for the document."""
    normalized = text.strip()
    if not normalized:
        return []

    base_config = config or load_chunk_config()
    chunk_config = ChunkConfig(
        chunk_size=size if size is not None else base_config.chunk_size,
        chunk_overlap=overlap if overlap is not None else base_config.chunk_overlap,
        tokenizer=base_config.tokenizer,
        semantic_model=base_config.semantic_model,
        semantic_enabled=base_config.semantic_enabled,
    )
    tokenizer = _resolved_tokenizer(chunk_config.tokenizer)

    ctx = context or ChunkContext()
    if strategy is None:
        primary = select_chunk_strategy(ctx, normalized)
    elif isinstance(strategy, ChunkStrategy):
        primary = strategy
    else:
        primary = ChunkStrategy(strategy)

    chain = (primary, *_FALLBACK_CHAIN.get(primary, (ChunkStrategy.RECURSIVE, ChunkStrategy.TOKEN)))
    for candidate in chain:
        if not _strategy_supported(candidate, chunk_config):
            continue
        try:
            pieces = _run_strategy(
                candidate,
                normalized,
                tokenizer=tokenizer,
                config=chunk_config,
            )
        except Exception as exc:
            log.warning(
                "chunk strategy %s failed for %s: %s",
                candidate.value,
                ctx.source_path or ctx.file_type,
                exc,
            )
            continue
        if pieces:
            if candidate is not primary:
                log.info(
                    "chunk fallback %s -> %s for %s",
                    primary.value,
                    candidate.value,
                    ctx.source_path or ctx.file_type,
                )
            else:
                log.debug(
                    "chunk strategy=%s tokenizer=%s size=%s overlap=%s path=%s",
                    candidate.value,
                    tokenizer,
                    chunk_config.chunk_size,
                    chunk_config.chunk_overlap,
                    ctx.source_path or ctx.file_type,
                )
            return pieces

    return []

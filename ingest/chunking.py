"""Text chunking for ingest via Chonkie with per-document strategy selection."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from chonkie import Pipeline, RecursiveChunker, SentenceChunker, TokenChunker

from ingest.chunk_config import (
    ChunkConfig,
    DEFAULT_CHUNK_OVERLAP_TOKENS,
    DEFAULT_CHUNK_SIZE_TOKENS,
    load_chunk_config,
)
from ingest.chunk_tokenizer import count_tokens, resolve_tokenizer, warmup_chunking
from ingest.chunking_strategy import ChunkContext, ChunkStrategy, select_chunk_strategy
from ingest.scrape_cleanup import strip_scrape_boilerplate

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


def _prepare_text(strategy: ChunkStrategy, text: str) -> str:
    if strategy is ChunkStrategy.TOKEN:
        return strip_scrape_boilerplate(text)
    return text


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


def merge_undersized_chunks(
    pieces: list[str],
    *,
    tokenizer: str,
    min_tokens: int,
    max_tokens: int,
) -> list[str]:
    """Merge adjacent chunks below min_tokens (semantic long-tail guard)."""
    if min_tokens <= 0 or len(pieces) <= 1:
        return pieces

    merged: list[str] = []
    buffer = ""
    buffer_tokens = 0

    for piece in pieces:
        piece_tokens = count_tokens(piece, tokenizer)
        if not buffer:
            buffer = piece
            buffer_tokens = piece_tokens
            continue
        combined = f"{buffer}\n\n{piece}"
        combined_tokens = count_tokens(combined, tokenizer)
        if buffer_tokens < min_tokens and combined_tokens <= max_tokens:
            buffer = combined
            buffer_tokens = combined_tokens
        else:
            merged.append(buffer)
            buffer = piece
            buffer_tokens = piece_tokens

    if buffer:
        merged.append(buffer)
    return merged


def _log_chunk_stats(
    pieces: list[str],
    *,
    strategy: ChunkStrategy,
    tokenizer: str,
    config: ChunkConfig,
    context: ChunkContext,
) -> None:
    if not pieces:
        return
    token_counts = [count_tokens(piece, tokenizer) for piece in pieces]
    short = sum(1 for n in token_counts if n < config.min_chunk_tokens)
    if strategy is not ChunkStrategy.SEMANTIC and short == 0:
        return
    label = context.source_path or context.file_type
    log.info(
        "chunk stats strategy=%s path=%s n=%s min=%s max=%s short_below_%s=%s",
        strategy.value,
        label,
        len(pieces),
        min(token_counts),
        max(token_counts),
        config.min_chunk_tokens,
        short,
    )
    if strategy is ChunkStrategy.SEMANTIC and short:
        log.info(
            "chunk stats hint: %s undersized semantic chunks on %s; "
            "merge pass applied (raise INGEST_CHUNK_MIN_TOKENS or check chonkie SDPM)",
            short,
            label,
        )


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
        min_chunk_tokens=base_config.min_chunk_tokens,
    )
    tokenizer = resolve_tokenizer(chunk_config.tokenizer)

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
            prepared = _prepare_text(candidate, normalized)
            if not prepared.strip():
                continue
            pieces = _run_strategy(
                candidate,
                prepared,
                tokenizer=tokenizer,
                config=chunk_config,
            )
            pieces = merge_undersized_chunks(
                pieces,
                tokenizer=tokenizer,
                min_tokens=chunk_config.min_chunk_tokens,
                max_tokens=chunk_config.chunk_size,
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
            _log_chunk_stats(
                pieces,
                strategy=candidate,
                tokenizer=tokenizer,
                config=chunk_config,
                context=ctx,
            )
            return pieces

    return []


__all__ = [
    "DEFAULT_CHUNK_OVERLAP",
    "DEFAULT_CHUNK_SIZE",
    "chunk_text",
    "merge_undersized_chunks",
    "warmup_chunking",
]

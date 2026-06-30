"""Text chunking for ingest via Chonkie with per-document strategy selection."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from chonkie import Pipeline, RecursiveChunker, SentenceChunker, TokenChunker

from ingest.chunking_strategy import ChunkContext, ChunkStrategy, select_chunk_strategy
from ingest.scrape_cleanup import strip_scrape_boilerplate

log = logging.getLogger("ingest.chunking")

# nomic-embed-text-v1.5 is tuned for ~512-token inputs; 64 tokens ~= 12.5% overlap.
DEFAULT_CHUNK_SIZE_TOKENS = 512
DEFAULT_CHUNK_OVERLAP_TOKENS = 64
DEFAULT_CHUNK_TOKENIZER = "nomic-ai/nomic-embed-text-v1.5"
DEFAULT_SEMANTIC_MODEL = "minishlab/potion-base-32M"
DEFAULT_MIN_CHUNK_TOKENS = 100
TOKENIZER_FALLBACKS = ("gpt2", "word")

_PROBE_TEXT = "tokenizer probe text."


def _env_bool_from_str(raw: str | None, default: bool) -> bool:
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ChunkConfig:
    chunk_size: int = DEFAULT_CHUNK_SIZE_TOKENS
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP_TOKENS
    tokenizer: str = DEFAULT_CHUNK_TOKENIZER
    semantic_model: str = DEFAULT_SEMANTIC_MODEL
    semantic_enabled: bool = True
    min_chunk_tokens: int = DEFAULT_MIN_CHUNK_TOKENS


def chunk_config_from_values(values: dict[str, str]) -> ChunkConfig:
    """Build chunk config from settings/env key-value map."""
    return ChunkConfig(
        chunk_size=int(values.get("INGEST_CHUNK_SIZE_TOKENS", str(DEFAULT_CHUNK_SIZE_TOKENS))),
        chunk_overlap=int(
            values.get("INGEST_CHUNK_OVERLAP_TOKENS", str(DEFAULT_CHUNK_OVERLAP_TOKENS))
        ),
        tokenizer=values.get("INGEST_CHUNK_TOKENIZER", DEFAULT_CHUNK_TOKENIZER),
        semantic_model=values.get("INGEST_CHUNK_SEMANTIC_MODEL", DEFAULT_SEMANTIC_MODEL),
        semantic_enabled=_env_bool_from_str(values.get("INGEST_CHUNK_SEMANTIC"), True),
        min_chunk_tokens=int(
            values.get("INGEST_CHUNK_MIN_TOKENS", str(DEFAULT_MIN_CHUNK_TOKENS))
        ),
    )


def load_chunk_config() -> ChunkConfig:
    """Load chunk settings from environment."""
    return chunk_config_from_values(
        {
            "INGEST_CHUNK_SIZE_TOKENS": os.getenv(
                "INGEST_CHUNK_SIZE_TOKENS", str(DEFAULT_CHUNK_SIZE_TOKENS)
            ),
            "INGEST_CHUNK_OVERLAP_TOKENS": os.getenv(
                "INGEST_CHUNK_OVERLAP_TOKENS", str(DEFAULT_CHUNK_OVERLAP_TOKENS)
            ),
            "INGEST_CHUNK_TOKENIZER": os.getenv("INGEST_CHUNK_TOKENIZER", DEFAULT_CHUNK_TOKENIZER),
            "INGEST_CHUNK_SEMANTIC_MODEL": os.getenv(
                "INGEST_CHUNK_SEMANTIC_MODEL", DEFAULT_SEMANTIC_MODEL
            ),
            "INGEST_CHUNK_SEMANTIC": os.getenv("INGEST_CHUNK_SEMANTIC"),
            "INGEST_CHUNK_MIN_TOKENS": os.getenv(
                "INGEST_CHUNK_MIN_TOKENS", str(DEFAULT_MIN_CHUNK_TOKENS)
            ),
        }
    )


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
    "ChunkConfig",
    "DEFAULT_CHUNK_OVERLAP",
    "DEFAULT_CHUNK_SIZE",
    "chunk_config_from_values",
    "chunk_text",
    "count_tokens",
    "load_chunk_config",
    "merge_undersized_chunks",
    "resolve_tokenizer",
    "warmup_chunking",
]

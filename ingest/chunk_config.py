"""Ingest chunk sizing and tokenizer settings (env-backed)."""

from __future__ import annotations

import os
from dataclasses import dataclass

# nomic-embed-text-v1.5 is tuned for ~512-token inputs; 64 tokens ~= 12.5% overlap.
DEFAULT_CHUNK_SIZE_TOKENS = 512
DEFAULT_CHUNK_OVERLAP_TOKENS = 64
DEFAULT_CHUNK_TOKENIZER = "nomic-ai/nomic-embed-text-v1.5"
DEFAULT_SEMANTIC_MODEL = "minishlab/potion-base-32M"
TOKENIZER_FALLBACKS = ("gpt2", "word")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ChunkConfig:
    chunk_size: int = DEFAULT_CHUNK_SIZE_TOKENS
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP_TOKENS
    tokenizer: str = DEFAULT_CHUNK_TOKENIZER
    semantic_model: str = DEFAULT_SEMANTIC_MODEL
    semantic_enabled: bool = True


def load_chunk_config() -> ChunkConfig:
    """Load chunk settings from environment."""
    return ChunkConfig(
        chunk_size=int(os.getenv("INGEST_CHUNK_SIZE_TOKENS", str(DEFAULT_CHUNK_SIZE_TOKENS))),
        chunk_overlap=int(
            os.getenv("INGEST_CHUNK_OVERLAP_TOKENS", str(DEFAULT_CHUNK_OVERLAP_TOKENS))
        ),
        tokenizer=os.getenv("INGEST_CHUNK_TOKENIZER", DEFAULT_CHUNK_TOKENIZER),
        semantic_model=os.getenv("INGEST_CHUNK_SEMANTIC_MODEL", DEFAULT_SEMANTIC_MODEL),
        semantic_enabled=_env_bool("INGEST_CHUNK_SEMANTIC", True),
    )

"""Ingest chunk sizing and tokenizer settings (env-backed)."""

from __future__ import annotations

import os
from dataclasses import dataclass

# nomic-embed-text-v1.5 is tuned for ~512-token inputs; 64 tokens ~= 12.5% overlap.
DEFAULT_CHUNK_SIZE_TOKENS = 512
DEFAULT_CHUNK_OVERLAP_TOKENS = 64
DEFAULT_CHUNK_TOKENIZER = "nomic-ai/nomic-embed-text-v1.5"
DEFAULT_SEMANTIC_MODEL = "minishlab/potion-base-32M"
DEFAULT_MIN_CHUNK_TOKENS = 100
TOKENIZER_FALLBACKS = ("gpt2", "word")


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

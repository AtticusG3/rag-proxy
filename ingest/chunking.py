"""Text chunking for ingest via Chonkie RecursiveChunker."""

from __future__ import annotations

from functools import lru_cache

from chonkie import Pipeline, RecursiveChunker

# Keep below llama-server per-slot token limit when --parallel divides context
# (e.g. -c 8096 --parallel 16 -> ~512 tokens per input).
DEFAULT_CHUNK_SIZE = 400
DEFAULT_CHUNK_OVERLAP = 64


@lru_cache(maxsize=8)
def _get_chunker(size: int, overlap: int) -> tuple[Pipeline | RecursiveChunker, bool]:
    if overlap > 0:
        pipe = (
            Pipeline()
            .chunk_with("recursive", tokenizer="character", chunk_size=size)
            .refine_with("overlap", context_size=overlap)
        )
        return pipe, True
    return RecursiveChunker(tokenizer="character", chunk_size=size), False


def chunk_text(
    text: str,
    size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping chunks using Chonkie recursive rules."""
    normalized = text.strip()
    if not normalized:
        return []

    chunker, use_pipeline = _get_chunker(size, overlap)
    if use_pipeline:
        raw = [chunk.text for chunk in chunker.run(texts=normalized).chunks]
    else:
        raw = [chunk.text for chunk in chunker(normalized)]
    return [piece for piece in raw if piece.strip()]

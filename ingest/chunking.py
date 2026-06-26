"""Text chunking for ingest (paragraph-aware, overlapping)."""

from __future__ import annotations

import re

# Keep below llama-server per-slot token limit when --parallel divides context
# (e.g. -c 8096 --parallel 16 -> ~512 tokens per input).
DEFAULT_CHUNK_SIZE = 400
DEFAULT_CHUNK_OVERLAP = 64


def _split_fixed(text: str, size: int, overlap: int) -> list[str]:
    """Split long text into fixed-size overlapping slices."""
    if not text:
        return []
    if size <= 0:
        return [text]
    if overlap >= size:
        overlap = max(0, size // 4)
    chunks: list[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + size, length)
        chunks.append(text[start:end])
        if end >= length:
            break
        start = end - overlap
    return chunks


def chunk_text(
    text: str,
    size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping chunks at paragraph boundaries."""
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    chunks: list[str] = []
    current = ""
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        if len(para) > size:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_split_fixed(para, size, overlap))
            continue
        if current and len(current) + len(para) + 2 > size:
            chunks.append(current)
            if len(current) > overlap:
                current = current[-overlap:] + "\n\n" + para
            else:
                current = para
        else:
            current = current + "\n\n" + para if current else para

    if current:
        chunks.append(current)
    return chunks

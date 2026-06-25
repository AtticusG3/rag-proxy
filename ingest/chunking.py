"""Text chunking for ingest (paragraph-aware, overlapping)."""

from __future__ import annotations

import re

DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 64


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

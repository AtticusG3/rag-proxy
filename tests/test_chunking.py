"""Tests for ingest chunking."""

from __future__ import annotations

from ingest.chunking import DEFAULT_CHUNK_SIZE, chunk_text


def test_chunk_text_splits_huge_paragraph():
    huge = "x" * 10_000
    chunks = chunk_text(f"{huge}\n\nafter")
    assert all(len(c) <= DEFAULT_CHUNK_SIZE for c in chunks)
    assert chunks[0].startswith("x")
    assert any("after" in c for c in chunks)


def test_chunk_text_respects_size_limit():
    text = "word " * 200
    chunks = chunk_text(text, size=100, overlap=10)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)

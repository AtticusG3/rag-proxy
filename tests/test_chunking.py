"""Tests for ingest chunking."""

from __future__ import annotations

from ingest.chunking import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, ChunkConfig, chunk_text
from ingest.chunking_strategy import ChunkStrategy


def test_chunk_text_splits_long_prose():
    prose = ("Sentence one about homelab deployment. " * 200) + "\n\nFinal paragraph here."
    chunks = chunk_text(
        prose,
        size=100,
        overlap=10,
        config=ChunkConfig(tokenizer="word", semantic_enabled=False),
        strategy=ChunkStrategy.SENTENCE,
    )
    assert len(chunks) > 1
    assert all(len(c) > 0 for c in chunks)
    assert any("Final paragraph" in c for c in chunks)


def test_chunk_text_respects_token_budget():
    text = "word " * 800
    chunks = chunk_text(
        text,
        size=100,
        overlap=10,
        config=ChunkConfig(tokenizer="word", semantic_enabled=False),
        strategy=ChunkStrategy.TOKEN,
    )
    assert len(chunks) > 1
    assert all(len(c.split()) <= 110 for c in chunks)


def test_default_token_targets_match_nomic_guidance():
    assert DEFAULT_CHUNK_SIZE == 512
    assert DEFAULT_CHUNK_OVERLAP == 64

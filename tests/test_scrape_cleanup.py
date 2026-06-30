"""Tests for scrape cleanup and chunk post-processing."""

from __future__ import annotations

from ingest.chunk_config import ChunkConfig
from ingest.chunking import merge_undersized_chunks
from ingest.chunk_tokenizer import resolve_tokenizer, warmup_chunking
from ingest.scrape_cleanup import strip_scrape_boilerplate


def test_strip_scrape_boilerplate_removes_cookie_lines():
    text = (
        "Real article paragraph about a topic.\n\n"
        "We use cookies to improve your experience.\n\n"
        "More substantive content here."
    )
    cleaned = strip_scrape_boilerplate(text)
    assert "cookies" not in cleaned.lower()
    assert "substantive content" in cleaned


def test_merge_undersized_chunks_combines_adjacent_small_pieces():
    short = "word " * 20
    long = "word " * 200
    merged = merge_undersized_chunks(
        [short, long],
        tokenizer="word",
        min_tokens=50,
        max_tokens=300,
    )
    assert len(merged) == 1
    assert "word" in merged[0]


def test_warmup_chunking_logs_active_tokenizer(caplog):
    import logging

    caplog.set_level(logging.INFO, logger="ingest.chunking")
    active = warmup_chunking(ChunkConfig(tokenizer="word"))
    assert active == "word"
    assert any("ingest chunking ready" in r.message for r in caplog.records)


def test_resolve_tokenizer_warns_on_fallback(caplog):
    import logging

    caplog.set_level(logging.WARNING, logger="ingest.chunking")
    active = resolve_tokenizer("definitely-not-a-real-tokenizer-name")
    assert active in ("gpt2", "word")
    assert any("FALLBACK" in r.message for r in caplog.records)

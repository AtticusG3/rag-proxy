"""Tests for scrape cleanup and chunk post-processing."""

from __future__ import annotations

from ingest.chunking import ChunkConfig, merge_undersized_chunks, resolve_tokenizer, warmup_chunking
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


def test_merge_undersized_chunks_counts_tokens_linearly(monkeypatch) -> None:
    """Merge pass should not re-tokenize full combined buffers (O(n) not O(n^2))."""
    from ingest import chunking

    calls = 0
    real_count = chunking.count_tokens

    def counting_tokens(text: str, tokenizer: str) -> int:
        nonlocal calls
        calls += 1
        return real_count(text, tokenizer)

    monkeypatch.setattr(chunking, "count_tokens", counting_tokens)
    pieces = [f"segment {index} " * 5 for index in range(12)]
    merge_undersized_chunks(
        pieces,
        tokenizer="word",
        min_tokens=5,
        max_tokens=500,
    )
    # One count per piece plus one for the "\n\n" separator probe.
    assert calls == len(pieces) + 1


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

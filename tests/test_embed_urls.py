"""Tests for ingest embed URL parsing."""

from __future__ import annotations

from ingest.embed_urls import parse_ingest_embed_urls


def test_parse_ingest_embed_urls_defaults_to_single():
    assert parse_ingest_embed_urls(embed_url="http://127.0.0.1:18089/") == [
        "http://127.0.0.1:18089"
    ]


def test_parse_ingest_embed_urls_splits_list():
    urls = parse_ingest_embed_urls(
        embed_url="http://127.0.0.1:18089",
        ingest_embed_urls="http://a:1, http://b:2/",
    )
    assert urls == ["http://a:1", "http://b:2"]

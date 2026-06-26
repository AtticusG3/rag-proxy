"""Parse embed endpoint list for ingest load balancing."""

from __future__ import annotations

import os


def parse_ingest_embed_urls(
    *,
    embed_url: str,
    ingest_embed_urls: str | None = None,
) -> list[str]:
    """Return ingest embed endpoints; INGEST_EMBED_URLS overrides a single EMBED_URL."""
    raw = ingest_embed_urls
    if raw is None:
        raw = os.getenv("INGEST_EMBED_URLS", "").strip()
    if raw:
        urls = [part.strip().rstrip("/") for part in raw.split(",") if part.strip()]
        if urls:
            return urls
    return [embed_url.rstrip("/")]

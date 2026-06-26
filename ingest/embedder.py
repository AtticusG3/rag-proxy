"""Sync embedding client for ingest worker."""

from __future__ import annotations

import time

import httpx

DEFAULT_EMBED_MODEL = "nomic-embed-text-v1.5"
DEFAULT_MAX_CHARS = 2000


def _post_embeddings(
    client: httpx.Client,
    *,
    embed_url: str,
    model: str,
    trimmed: list[str],
) -> list[list[float]]:
    response = client.post(
        f"{embed_url.rstrip('/')}/v1/embeddings",
        json={"model": model, "input": trimmed},
    )
    response.raise_for_status()
    data = response.json()["data"]
    return [item["embedding"] for item in data]


def embed_texts(
    texts: list[str],
    *,
    embed_url: str,
    model: str = DEFAULT_EMBED_MODEL,
    max_chars: int = DEFAULT_MAX_CHARS,
    retries: int = 2,
    client: httpx.Client | None = None,
) -> list[list[float]]:
    """Batch-embed texts via OpenAI-compatible embeddings API."""
    trimmed = [t.strip()[:max_chars] for t in texts]
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        if attempt:
            time.sleep(1.0)
        try:
            if client is not None:
                return _post_embeddings(
                    client, embed_url=embed_url, model=model, trimmed=trimmed
                )
            with httpx.Client(timeout=120.0) as owned:
                return _post_embeddings(
                    owned, embed_url=embed_url, model=model, trimmed=trimmed
                )
        except Exception as exc:
            last_err = exc
    raise RuntimeError(f"embed batch failed after {retries + 1} attempts: {last_err}") from last_err

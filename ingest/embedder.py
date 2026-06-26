"""Sync embedding client for ingest worker."""

from __future__ import annotations

import time

import httpx

DEFAULT_EMBED_MODEL = "nomic-embed-text-v1.5"
DEFAULT_MAX_CHARS = 2000
_CONTEXT_SHRINK_LIMITS = (400, 300, 200, 100)


def _exceed_context_size(response_text: str) -> bool:
    lower = response_text.lower()
    return "exceed_context_size" in lower or "max context size" in lower


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


def _embed_batch_resilient(
    client: httpx.Client,
    *,
    embed_url: str,
    model: str,
    texts: list[str],
    max_chars: int,
) -> list[list[float]]:
    """Embed a batch, bisecting or truncating on per-input context overflow."""
    trimmed = [t.strip()[:max_chars] for t in texts]
    try:
        return _post_embeddings(
            client, embed_url=embed_url, model=model, trimmed=trimmed
        )
    except httpx.HTTPStatusError as exc:
        body = exc.response.text or ""
        if exc.response.status_code != 400 or not _exceed_context_size(body):
            raise
        if len(trimmed) == 1:
            text = trimmed[0]
            for limit in _CONTEXT_SHRINK_LIMITS:
                if limit >= len(text):
                    continue
                try:
                    return _post_embeddings(
                        client,
                        embed_url=embed_url,
                        model=model,
                        trimmed=[text[:limit]],
                    )
                except httpx.HTTPStatusError as retry_exc:
                    retry_body = retry_exc.response.text or ""
                    if retry_exc.response.status_code == 400 and _exceed_context_size(
                        retry_body
                    ):
                        continue
                    raise
            raise
        mid = len(trimmed) // 2
        left = _embed_batch_resilient(
            client,
            embed_url=embed_url,
            model=model,
            texts=texts[:mid],
            max_chars=max_chars,
        )
        right = _embed_batch_resilient(
            client,
            embed_url=embed_url,
            model=model,
            texts=texts[mid:],
            max_chars=max_chars,
        )
        return left + right


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
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        if attempt:
            time.sleep(1.0)
        try:
            if client is not None:
                return _embed_batch_resilient(
                    client,
                    embed_url=embed_url,
                    model=model,
                    texts=texts,
                    max_chars=max_chars,
                )
            owned = httpx.Client(timeout=120.0)
            try:
                return _embed_batch_resilient(
                    owned,
                    embed_url=embed_url,
                    model=model,
                    texts=texts,
                    max_chars=max_chars,
                )
            finally:
                owned.close()
        except Exception as exc:
            last_err = exc
    raise RuntimeError(f"embed batch failed after {retries + 1} attempts: {last_err}") from last_err

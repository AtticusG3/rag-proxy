"""Pure retrieval request/response helpers shared by sync and async clients."""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger("rag-proxy")

EMBED_MODEL = "nomic-embed-text-v1.5"


def prepare_embed_text(text: str, max_chars: int) -> str:
    """Trim to a safe size for llama-server embed batch (-ub defaults to 512 tokens)."""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    log.warning(f"Embed input truncated (tail) {len(text)} -> {max_chars} chars")
    return text[-max_chars:]


def embed_input_too_large(response_text: str) -> bool:
    return "too large to process" in response_text


def embed_payload(text: str, *, model: str = EMBED_MODEL) -> dict[str, Any]:
    return {"model": model, "input": text}


def dense_search_payload(
    vector: list[float],
    limit: int,
    score_threshold: float | None,
    *,
    omit_zero_threshold: bool = False,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "vector": vector,
        "limit": limit,
        "with_payload": True,
    }
    if score_threshold is not None:
        if not omit_zero_threshold or score_threshold > 0:
            body["score_threshold"] = score_threshold
    return body


def sparse_search_payload(query: str, limit: int, collection: str) -> dict[str, Any]:
    return {"query": query, "limit": limit, "collection": collection}


def parse_embedding(response_json: dict[str, Any]) -> list[float] | None:
    data = response_json.get("data")
    if not data:
        return None
    embedding = data[0].get("embedding")
    if not isinstance(embedding, list):
        return None
    return embedding


def parse_dense_hits(response_json: dict[str, Any]) -> list[dict]:
    result = response_json.get("result")
    if not isinstance(result, list):
        return []
    return result


def parse_sparse_hits(response_json: dict[str, Any]) -> list[dict]:
    results = response_json.get("results")
    if not isinstance(results, list):
        return []
    return results

"""Pure retrieval request/response helpers shared by sync and async clients."""

from __future__ import annotations

import logging
from typing import Any

from rag_proxy.config import settings

log = logging.getLogger("rag-proxy")

# Default model id when EMBED_MODEL is unset (legacy nomic until llama-swap cutover).
DEFAULT_EMBED_MODEL = "nomic-embed-text-v1.5"
# Back-compat alias; prefer settings.embed_model / embed_payload() at call time.
EMBED_MODEL = DEFAULT_EMBED_MODEL


def prepare_embed_text(text: str, max_chars: int) -> str:
    """Trim to a safe size for llama-server embed batch (-ub defaults to 512 tokens)."""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    log.warning(f"Embed input truncated (tail) {len(text)} -> {max_chars} chars")
    return text[-max_chars:]


def embed_input_too_large(response_text: str) -> bool:
    """True when the embed server rejected the input for context size.

    llama-server reports this as HTTP 400 ``exceed_context_size`` / ``max context
    size``. Older nomic builds say ``too large to process``. Callers shrink and
    retry; treating only one phrasing as fatal drops RAG for the turn.
    """
    lower = response_text.lower()
    return (
        "too large to process" in lower
        or "exceed_context_size" in lower
        or "max context size" in lower
    )


def canonical_point_id(raw: object) -> str:
    """Fold Qdrant dashed UUIDs onto TurboVec's 32-char hex point ids.

    Ingest dual-writes bare MD5 hex. Qdrant scroll/retrieve often echoes the
    same point as a dashed UUID. Fusion and payload merge must treat them as one.
    Non-hex ids (integers, sparse labels) are returned unchanged.
    """
    text = str(raw).strip()
    compact = text.replace("-", "").lower()
    if len(compact) == 32 and all(c in "0123456789abcdef" for c in compact):
        return compact
    return text


def hit_doc_id(hit: dict) -> str:
    """Canonical document id for a dense or sparse hit."""
    raw = hit.get("id", hit.get("point_id", ""))
    if raw is None or raw == "":
        return ""
    return canonical_point_id(raw)


def embed_payload(text: str, *, model: str | None = None) -> dict[str, Any]:
    """Build OpenAI-compatible embeddings body; model defaults to settings.embed_model."""
    return {"model": model if model is not None else settings.embed_model, "input": text}


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


def turbovec_search_payload(
    vector: list[float],
    limit: int,
    score_threshold: float | None,
) -> dict[str, Any]:
    body: dict[str, Any] = {"vector": vector, "limit": limit}
    if score_threshold is not None:
        body["score_threshold"] = score_threshold
    return body


def parse_turbovec_hits(response_json: dict[str, Any]) -> list[dict]:
    results = response_json.get("results")
    if not isinstance(results, list):
        return []
    hits: list[dict] = []
    for row in results:
        if not isinstance(row, dict):
            continue
        hit_id = row.get("id")
        if hit_id is None:
            continue
        hits.append({"id": str(hit_id), "score": float(row.get("score", 0.0))})
    return hits


def merge_dense_scores_with_payloads(
    scored: list[dict],
    payload_points: list[dict],
) -> list[dict]:
    """Attach Qdrant payloads to turbovec id/score rows; preserve score order."""
    by_id: dict[str, dict] = {}
    for point in payload_points:
        point_id = point.get("id")
        if point_id is None:
            continue
        by_id[canonical_point_id(point_id)] = point.get("payload") or {}
    merged: list[dict] = []
    for row in scored:
        hit_id = canonical_point_id(row.get("id", ""))
        if not hit_id:
            continue
        merged.append(
            {
                "id": hit_id,
                "score": float(row.get("score", 0.0)),
                "payload": by_id.get(hit_id, {}),
            }
        )
    return merged


def qdrant_retrieve_payload(
    ids: list[str],
    *,
    with_payload: bool = True,
    with_vector: bool = False,
) -> dict[str, Any]:
    return {
        "ids": ids,
        "with_payload": with_payload,
        "with_vector": with_vector,
    }


def parse_qdrant_retrieve(response_json: dict[str, Any]) -> list[dict]:
    result = response_json.get("result")
    if not isinstance(result, list):
        return []
    return result


def turbovec_hits_from_scored(
    scored: list[dict],
    retrieve_json: dict[str, Any],
) -> list[dict]:
    """Merge pre-parsed TurboVec id/score rows with Qdrant retrieve JSON."""
    if not scored:
        return []
    payloads = parse_qdrant_retrieve(retrieve_json)
    return merge_dense_scores_with_payloads(scored, payloads)


def turbovec_hits_from_responses(
    search_json: dict[str, Any],
    retrieve_json: dict[str, Any],
) -> list[dict]:
    """Parse TurboVec /search and Qdrant retrieve JSON into merged hit dicts."""
    scored = parse_turbovec_hits(search_json)
    return turbovec_hits_from_scored(scored, retrieve_json)

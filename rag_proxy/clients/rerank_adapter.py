"""Map sidecar pairs/indices rerank contract to OpenAI-style /v1/rerank.

RERANK_API=sidecar (default): POST {RERANKER_URL}/rerank with pairs + top_k,
response indices.

RERANK_API=openai: POST {RERANKER_URL}/v1/rerank with model/query/documents/top_n
(llama-swap / llama-server). Response results[].index mapped back to indices.
"""

from __future__ import annotations

from typing import Any

from rag_proxy.config import settings


def use_openai_rerank() -> bool:
    return settings.rerank_api == "openai"


def rerank_request_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if use_openai_rerank():
        return f"{base}/v1/rerank"
    return f"{base}/rerank"


def pairs_to_openai_documents(pairs: list[dict[str, str]]) -> tuple[str, list[str]]:
    """Extract query + documents from sidecar-style pairs (shared query assumed)."""
    if not pairs:
        return "", []
    query = str(pairs[0].get("query", ""))
    documents = [str(p.get("document", "")) for p in pairs]
    return query, documents


def rerank_request_payload(
    pairs: list[dict[str, str]],
    top_k: int,
) -> dict[str, Any]:
    if use_openai_rerank():
        query, documents = pairs_to_openai_documents(pairs)
        return {
            "model": settings.rerank_model,
            "query": query,
            "documents": documents,
            "top_n": top_k,
        }
    return {"pairs": pairs, "top_k": top_k}


def parse_rerank_indices(
    response_json: dict[str, Any],
    *,
    n_pairs: int,
    top_k: int,
) -> list[int]:
    """Normalize sidecar or OpenAI-style rerank JSON to a list of pair indices."""
    if use_openai_rerank():
        results = response_json.get("results")
        if not isinstance(results, list):
            return []
        out: list[int] = []
        seen: set[int] = set()
        for row in results:
            if not isinstance(row, dict):
                continue
            idx = row.get("index")
            if not isinstance(idx, int) or idx < 0 or idx >= n_pairs or idx in seen:
                continue
            seen.add(idx)
            out.append(idx)
            if len(out) >= top_k:
                break
        return out

    order = response_json.get("indices", [])
    if not isinstance(order, list):
        return []
    return [i for i in order if isinstance(i, int) and 0 <= i < n_pairs][:top_k]

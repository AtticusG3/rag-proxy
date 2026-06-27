"""Service health probes for the settings page."""

from __future__ import annotations

import os
from typing import Any

import httpx


async def probe_url(url: str, path: str = "/health") -> dict[str, Any]:
    target = f"{url.rstrip('/')}{path}"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(target)
            body: Any = None
            if response.headers.get("content-type", "").startswith("application/json"):
                body = response.json()
            return {
                "url": target,
                "status_code": response.status_code,
                "ok": response.status_code < 400,
                "body": body,
            }
    except Exception as exc:
        return {"url": target, "status_code": 0, "ok": False, "error": str(exc)}


async def service_status(
    *,
    qdrant_url: str,
    collection: str,
    sparse_index_url: str,
    embed_url: str,
    rag_proxy_url: str,
    reranker_url: str,
    memgraph_db_path: str,
) -> dict[str, Any]:
    qdrant_points = 0
    qdrant_ok = False
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            response = await client.get(
                f"{qdrant_url.rstrip('/')}/collections/{collection}"
            )
            if response.status_code == 200:
                qdrant_ok = True
                qdrant_points = int(response.json()["result"]["points_count"])
    except Exception:
        pass

    sparse = await probe_url(sparse_index_url) if sparse_index_url else {"ok": False}
    embed = await probe_url(embed_url, "/health")
    if not embed.get("ok"):
        embed = await probe_url(embed_url, "/v1/models")
    proxy = await probe_url(rag_proxy_url)
    reranker = await probe_url(reranker_url) if reranker_url else {"ok": False}

    memgraph_db_exists = os.path.isfile(memgraph_db_path)
    memgraph_db_bytes = os.path.getsize(memgraph_db_path) if memgraph_db_exists else 0

    return {
        "qdrant": {"ok": qdrant_ok, "points": qdrant_points, "collection": collection},
        "sparse": sparse,
        "embed": embed,
        "proxy": proxy,
        "reranker": reranker,
        "memgraph_db": {
            "path": memgraph_db_path,
            "exists": memgraph_db_exists,
            "bytes": memgraph_db_bytes,
        },
    }

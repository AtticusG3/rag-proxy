"""Qdrant upsert and delete helpers for ingest."""

from __future__ import annotations

import hashlib
from typing import Any

import httpx

DEFAULT_VECTOR_SIZE = 768


def make_point_id(text: str, source: str, chunk_idx: int) -> str:
    digest = hashlib.md5(f"{source}:{chunk_idx}:{text[:100]}".encode()).hexdigest()
    return digest


def ensure_collection(
    qdrant_url: str,
    collection: str,
    vector_size: int = DEFAULT_VECTOR_SIZE,
    *,
    client: httpx.Client | None = None,
) -> None:
    base = qdrant_url.rstrip("/")
    if client is not None:
        _ensure_collection_with_client(client, base, collection, vector_size)
        return
    with httpx.Client(timeout=30.0) as owned:
        _ensure_collection_with_client(owned, base, collection, vector_size)


def _ensure_collection_with_client(
    client: httpx.Client,
    base: str,
    collection: str,
    vector_size: int,
) -> None:
    exists = client.get(f"{base}/collections/{collection}")
    if exists.status_code == 200:
        return
    response = client.put(
        f"{base}/collections/{collection}",
        json={"vectors": {"size": vector_size, "distance": "Cosine"}},
    )
    response.raise_for_status()


def get_collection_count(qdrant_url: str, collection: str) -> int:
    with httpx.Client(timeout=30.0) as client:
        response = client.get(f"{qdrant_url.rstrip('/')}/collections/{collection}")
        response.raise_for_status()
        return int(response.json()["result"]["points_count"])


def upsert_points(
    qdrant_url: str,
    collection: str,
    points: list[dict[str, Any]],
    *,
    client: httpx.Client | None = None,
) -> None:
    if not points:
        return
    url = f"{qdrant_url.rstrip('/')}/collections/{collection}/points"
    payload = {"points": points}
    if client is not None:
        response = client.put(url, json=payload)
        response.raise_for_status()
        return
    with httpx.Client(timeout=120.0) as owned:
        response = owned.put(url, json=payload)
        response.raise_for_status()


def delete_by_source(qdrant_url: str, collection: str, source: str) -> None:
    """Remove all points whose payload.source matches."""
    with httpx.Client(timeout=120.0) as client:
        response = client.post(
            f"{qdrant_url.rstrip('/')}/collections/{collection}/points/delete",
            json={
                "filter": {
                    "must": [
                        {"key": "source", "match": {"value": source}},
                    ]
                }
            },
        )
        response.raise_for_status()


def clear_collection(
    qdrant_url: str,
    collection: str,
    vector_size: int = DEFAULT_VECTOR_SIZE,
) -> int:
    """Drop and recreate a collection; returns the prior point count (0 if missing)."""
    base = qdrant_url.rstrip("/")
    with httpx.Client(timeout=120.0) as client:
        prior = 0
        info = client.get(f"{base}/collections/{collection}")
        if info.status_code == 200:
            prior = int(info.json()["result"]["points_count"])
        elif info.status_code != 404:
            info.raise_for_status()
        delete = client.delete(f"{base}/collections/{collection}")
        if delete.status_code not in (200, 404):
            delete.raise_for_status()
        response = client.put(
            f"{base}/collections/{collection}",
            json={"vectors": {"size": vector_size, "distance": "Cosine"}},
        )
        response.raise_for_status()
        return prior


def build_point(
    *,
    text: str,
    source: str,
    title: str,
    chunk_idx: int,
    embedding: list[float],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "text": text[:2000],
        "source": source,
        "title": title,
        "chunk_idx": chunk_idx,
        "chunk_size": len(text),
    }
    if extra:
        payload.update(extra)
    return {
        "id": make_point_id(text, source, chunk_idx),
        "vector": embedding,
        "payload": payload,
    }

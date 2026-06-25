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
) -> None:
    base = qdrant_url.rstrip("/")
    with httpx.Client(timeout=30.0) as client:
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
) -> None:
    if not points:
        return
    with httpx.Client(timeout=120.0) as client:
        response = client.put(
            f"{qdrant_url.rstrip('/')}/collections/{collection}/points",
            json={"points": points},
        )
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

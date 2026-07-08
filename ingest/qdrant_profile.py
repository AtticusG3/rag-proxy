"""Probe Qdrant collection stats for ingest capacity planning."""

from __future__ import annotations

from dataclasses import dataclass

import httpx


@dataclass(frozen=True)
class QdrantCollectionProfile:
    points_count: int
    indexed_vectors_count: int
    segment_count: int
    status: str
    optimizer_status: str


def probe_qdrant_collection(
    qdrant_url: str,
    collection: str,
    *,
    timeout_sec: float = 10.0,
) -> QdrantCollectionProfile | None:
    """Return collection stats or None when Qdrant is unreachable."""
    url = f"{qdrant_url.rstrip('/')}/collections/{collection}"
    try:
        with httpx.Client(timeout=timeout_sec) as client:
            response = client.get(url)
            response.raise_for_status()
    except Exception:
        return None
    result = response.json().get("result") or {}
    return QdrantCollectionProfile(
        points_count=int(result.get("points_count") or 0),
        indexed_vectors_count=int(result.get("indexed_vectors_count") or 0),
        segment_count=int(result.get("segments_count") or 0),
        status=str(result.get("status") or ""),
        optimizer_status=str(result.get("optimizer_status") or ""),
    )

"""Qdrant upsert and delete helpers for ingest."""

from __future__ import annotations

import hashlib
import logging
import os
import time
from typing import Any

import httpx

log = logging.getLogger("ingest.qdrant")

DEFAULT_VECTOR_SIZE = 768
DEFAULT_UPSERT_TIMEOUT_SEC = 180.0
DEFAULT_UPSERT_RETRIES = 4
DEFAULT_UPSERT_BACKOFF_SEC = 2.0

_RETRYABLE_HTTP = frozenset({408, 429, 500, 502, 503, 504})


def qdrant_upsert_timeout_sec() -> float:
    return float(os.getenv("QDRANT_UPSERT_TIMEOUT_SEC", str(DEFAULT_UPSERT_TIMEOUT_SEC)))


def qdrant_upsert_retries() -> int:
    return max(0, int(os.getenv("QDRANT_UPSERT_RETRIES", str(DEFAULT_UPSERT_RETRIES))))


def qdrant_upsert_backoff_sec() -> float:
    return float(os.getenv("QDRANT_UPSERT_BACKOFF_SEC", str(DEFAULT_UPSERT_BACKOFF_SEC)))


def _is_retryable_qdrant_error(exc: Exception) -> bool:
    if isinstance(
        exc,
        (
            httpx.ConnectError,
            httpx.ReadTimeout,
            httpx.WriteTimeout,
            httpx.RemoteProtocolError,
            httpx.ReadError,
            httpx.WriteError,
            httpx.PoolTimeout,
        ),
    ):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _RETRYABLE_HTTP
    return False


def _put_points_once(
    client: httpx.Client,
    url: str,
    points: list[dict[str, Any]],
) -> None:
    response = client.put(url, json={"points": points})
    response.raise_for_status()


def _upsert_points_resilient(
    client: httpx.Client,
    url: str,
    points: list[dict[str, Any]],
    *,
    retries: int,
    backoff_sec: float,
) -> None:
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        if attempt:
            delay = backoff_sec * (2 ** (attempt - 1))
            log.warning(
                "qdrant upsert retry %d/%d for %d points after %.1fs (%s)",
                attempt,
                retries,
                len(points),
                delay,
                last_err,
            )
            time.sleep(delay)
        try:
            _put_points_once(client, url, points)
            return
        except Exception as exc:
            last_err = exc
            if not _is_retryable_qdrant_error(exc):
                raise
    if len(points) > 1:
        mid = len(points) // 2
        log.warning(
            "qdrant upsert bisecting batch of %d points after retries exhausted",
            len(points),
        )
        _upsert_points_resilient(
            client, url, points[:mid], retries=retries, backoff_sec=backoff_sec
        )
        _upsert_points_resilient(
            client, url, points[mid:], retries=retries, backoff_sec=backoff_sec
        )
        return
    raise RuntimeError(
        f"qdrant upsert failed after {retries + 1} attempts: {last_err}"
    ) from last_err


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
    retries: int | None = None,
    backoff_sec: float | None = None,
) -> None:
    if not points:
        return
    url = f"{qdrant_url.rstrip('/')}/collections/{collection}/points"
    max_retries = qdrant_upsert_retries() if retries is None else max(0, retries)
    backoff = qdrant_upsert_backoff_sec() if backoff_sec is None else backoff_sec
    if client is not None:
        _upsert_points_resilient(
            client, url, points, retries=max_retries, backoff_sec=backoff
        )
        return
    with httpx.Client(timeout=qdrant_upsert_timeout_sec()) as owned:
        _upsert_points_resilient(
            owned, url, points, retries=max_retries, backoff_sec=backoff
        )


def _source_filter(source: str) -> dict[str, Any]:
    return {
        "must": [
            {"key": "source", "match": {"value": source}},
        ]
    }


def list_point_ids_by_source(
    qdrant_url: str,
    collection: str,
    source: str,
    *,
    page_size: int = 256,
) -> list[str]:
    """Return point ids whose payload.source matches (for MemGraphRAG scrub)."""
    base = qdrant_url.rstrip("/")
    ids: list[str] = []
    offset: Any = None
    with httpx.Client(timeout=120.0) as client:
        while True:
            body: dict[str, Any] = {
                "filter": _source_filter(source),
                "limit": page_size,
                "with_payload": False,
                "with_vector": False,
            }
            if offset is not None:
                body["offset"] = offset
            response = client.post(
                f"{base}/collections/{collection}/points/scroll",
                json=body,
            )
            response.raise_for_status()
            result = response.json().get("result") or {}
            for point in result.get("points") or []:
                point_id = point.get("id")
                if point_id is not None:
                    ids.append(str(point_id))
            offset = result.get("next_page_offset")
            if offset is None:
                break
    return ids


def delete_by_source(qdrant_url: str, collection: str, source: str) -> None:
    """Remove all points whose payload.source matches."""
    with httpx.Client(timeout=120.0) as client:
        response = client.post(
            f"{qdrant_url.rstrip('/')}/collections/{collection}/points/delete",
            json={"filter": _source_filter(source)},
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

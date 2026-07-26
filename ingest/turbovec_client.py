"""Best-effort TurboVec dual-write / remove helpers for ingest."""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

log = logging.getLogger("ingest.turbovec")


def turbovec_url_from_env(explicit: str | None = None) -> str:
    if explicit is not None:
        return explicit.strip()
    return os.getenv("TURBOVEC_URL", "").strip()


def add_points(
    turbovec_url: str,
    points: list[dict[str, Any]],
    *,
    client: httpx.Client | None = None,
) -> int | None:
    """POST /add with ids+vectors from Qdrant-shaped points. Fail-open."""
    url = turbovec_url_from_env(turbovec_url)
    if not url or not points:
        return None
    ids: list[str] = []
    vectors: list[list[float]] = []
    for point in points:
        point_id = point.get("id")
        vector = point.get("vector")
        if point_id is None or not isinstance(vector, list) or not vector:
            continue
        ids.append(str(point_id))
        vectors.append([float(x) for x in vector])
    if not ids:
        return None
    payload = {"ids": ids, "vectors": vectors}
    try:
        if client is not None:
            response = client.post(f"{url.rstrip('/')}/add", json=payload)
            response.raise_for_status()
            return int(response.json().get("added", len(ids)))
        with httpx.Client(timeout=60.0) as owned:
            response = owned.post(f"{url.rstrip('/')}/add", json=payload)
            response.raise_for_status()
            return int(response.json().get("added", len(ids)))
    except Exception as exc:
        log.warning("turbovec add failed (%d points): %s", len(ids), exc)
        return None


def remove_ids(
    turbovec_url: str,
    ids: list[str],
    *,
    client: httpx.Client | None = None,
) -> int | None:
    """POST /remove. Fail-open."""
    url = turbovec_url_from_env(turbovec_url)
    if not url or not ids:
        return None
    payload = {"ids": [str(i) for i in ids]}
    try:
        if client is not None:
            response = client.post(f"{url.rstrip('/')}/remove", json=payload)
            response.raise_for_status()
            return int(response.json().get("removed", 0))
        with httpx.Client(timeout=60.0) as owned:
            response = owned.post(f"{url.rstrip('/')}/remove", json=payload)
            response.raise_for_status()
            return int(response.json().get("removed", 0))
    except Exception as exc:
        log.warning("turbovec remove failed (%d ids): %s", len(ids), exc)
        return None


def trigger_reindex(
    turbovec_url: str,
    collection: str,
) -> int | None:
    url = turbovec_url_from_env(turbovec_url)
    if not url:
        return None
    try:
        with httpx.Client(timeout=600.0) as client:
            response = client.post(
                f"{url.rstrip('/')}/reindex",
                json={"collection": collection},
            )
            response.raise_for_status()
            return int(response.json().get("docs", 0))
    except Exception as exc:
        log.warning("turbovec reindex failed: %s", exc)
        return None

"""One-time Qdrant → TurboVec / BM25 sidecar migration helpers."""

from __future__ import annotations

from typing import Any


def sparse_target_docs(qdrant_points: int, *, max_points: int | None) -> int:
    """How many BM25 docs we expect after a full reindex from Qdrant."""
    points = max(0, int(qdrant_points))
    if max_points is None:
        return points
    cap = int(max_points)
    if cap <= 0:
        return points
    return min(points, cap)


def needs_sidecar_rebuild(current: int, target: int) -> bool:
    """True when the sidecar count disagrees with Qdrant.

    Behind means missing docs. Ahead means ghost entries left by a collection
    clear or partial wipe, which a rebuild flushes. Both need a full reindex.
    """
    return int(current) != int(target)


def health_count(body: dict[str, Any] | None, *keys: str) -> int:
    if not isinstance(body, dict):
        return 0
    for key in keys:
        raw = body.get(key)
        if raw is None:
            continue
        try:
            return int(raw)
        except (TypeError, ValueError):
            continue
    return 0


def health_max_points(body: dict[str, Any] | None) -> int | None:
    if not isinstance(body, dict):
        return None
    raw = body.get("max_points")
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None

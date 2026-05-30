"""Rerank scoring helpers (no FastAPI dependency)."""

from __future__ import annotations


def rank_indices(scores: list[float], top_k: int) -> list[int]:
    """Return document indices sorted by descending score."""
    if not scores:
        return []
    ranked = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)
    limit = max(1, min(top_k, len(ranked)))
    return ranked[:limit]

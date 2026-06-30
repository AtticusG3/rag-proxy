"""Shared environment value parsing helpers."""

from __future__ import annotations


def parse_bool(raw: str | None, default: bool) -> bool:
    """Parse common truthy/falsey string forms; None or blank returns default."""
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}

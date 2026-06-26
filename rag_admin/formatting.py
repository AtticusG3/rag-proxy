"""Display formatting helpers for admin UI."""

from __future__ import annotations

from datetime import datetime


def format_datetime(value: str | None) -> str:
    """Format ISO-8601 timestamps as DD-MMM-YYYY hh:mm:ss."""
    if not value:
        return ""
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value
    return dt.strftime("%d-%b-%Y %H:%M:%S")

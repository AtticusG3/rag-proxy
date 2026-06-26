"""Detect ingest jobs that stopped making progress."""

from __future__ import annotations

from datetime import datetime, timezone


def parse_utc_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def seconds_since_update(updated_at: str | None) -> float | None:
    if not updated_at:
        return None
    delta = datetime.now(timezone.utc) - parse_utc_timestamp(updated_at)
    return max(0.0, delta.total_seconds())


def is_stalled(updated_at: str | None, stall_seconds: int) -> bool:
    if stall_seconds <= 0:
        return False
    age = seconds_since_update(updated_at)
    if age is None:
        return True
    return age > stall_seconds


def stall_error_message(*, stall_seconds: int, chunks_embedded: int) -> str:
    minutes = max(1, stall_seconds // 60)
    return (
        f"stalled: no progress for {minutes}+ minutes "
        f"(stopped at {chunks_embedded} chunks)"
    )


def interrupt_error_message(chunks_embedded: int) -> str:
    if chunks_embedded:
        return (
            f"ingest interrupted (worker restarted at {chunks_embedded} chunks)"
        )
    return "ingest interrupted (worker restarted)"

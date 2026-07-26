"""Server-side embed throughput samples for admin ingest velocity."""

from __future__ import annotations

import time
from collections import deque

MIN_ELAPSED_S = 4.8
WINDOW_5_S = 5 * 60
WINDOW_15_S = 15 * 60
# 5m rate once samples span ~half the window (~2.5 min); 15m kept for API/tests.
MIN_WINDOW_FRACTION_5M = 0.5
MIN_WINDOW_FRACTION_15M = 0.9

# Filled on spaced /api/ingest/status polls (~8s while active). ~120 samples ≈ 16 minutes.
_samples: deque[tuple[float, int]] = deque(maxlen=120)


def reset_embed_throughput() -> None:
    """Clear samples (tests only)."""
    _samples.clear()


def record_embed_progress(total_chunks: int, *, now: float | None = None) -> None:
    """Record corpus chunk total; ignore bursts closer than MIN_ELAPSED_S.

    Dashboard SSR and Jobs live poll can hit ingest_queue_stats within the same
    second; accepting both would leave "now" as None forever (elapsed < 4.8s).
    """
    ts = now if now is not None else time.time()
    if _samples and (ts - _samples[-1][0]) < MIN_ELAPSED_S:
        return
    _samples.append((ts, total_chunks))


def _rate_between(
    baseline: tuple[float, int],
    current: tuple[float, int],
) -> int | None:
    elapsed = current[0] - baseline[0]
    if elapsed < MIN_ELAPSED_S:
        return None
    delta = current[1] - baseline[1]
    if delta < 0:
        return None
    return round(delta / (elapsed / 60.0))


def _baseline_for_window(window_s: float) -> tuple[float, int] | None:
    if not _samples:
        return None
    current = _samples[-1]
    target = current[0] - window_s
    baseline = _samples[0]
    for sample in _samples:
        if sample[0] <= target:
            baseline = sample
        else:
            break
    return baseline


def _rate_over_window(window_s: float, *, min_fraction: float) -> int | None:
    if len(_samples) < 2:
        return None
    current = _samples[-1]
    baseline = _baseline_for_window(window_s)
    if baseline is None or baseline[0] == current[0]:
        return None
    actual_span = current[0] - baseline[0]
    if actual_span < window_s * min_fraction:
        return None
    return _rate_between(baseline, current)


def embed_throughput_rates() -> dict[str, int | None]:
    """Return now / 5m / 15m chunks-per-minute from recorded samples."""
    if len(_samples) < 2:
        return {"embed_rate_now": None, "embed_rate_5m": None, "embed_rate_15m": None}
    current = _samples[-1]
    previous = _samples[-2]
    return {
        "embed_rate_now": _rate_between(previous, current),
        "embed_rate_5m": _rate_over_window(
            WINDOW_5_S, min_fraction=MIN_WINDOW_FRACTION_5M
        ),
        "embed_rate_15m": _rate_over_window(
            WINDOW_15_S, min_fraction=MIN_WINDOW_FRACTION_15M
        ),
    }


def format_primary_rate(
    rate: int | None,
    *,
    running: int,
    pending: int,
) -> str | None:
    """Primary throughput clause for the velocity line, or None to omit.

    Returns phrases like ``420 chunks/min``, ``measuring…``, or ``waiting``.
    """
    if rate is not None and rate > 0:
        return f"{rate:,} chunks/min"
    if running > 0:
        # Flat counter mid-batch (GPU busy) or zero delta — not idle.
        return "measuring…"
    if pending > 0:
        return "waiting"
    return None


def format_embed_rate(
    rate: int | None,
    *,
    running: int,
    pending: int,
    window: str,
) -> str:
    """Legacy helper; prefer format_primary_rate for the Jobs line."""
    if window == "now":
        primary = format_primary_rate(rate, running=running, pending=pending)
        return primary if primary is not None else "—"
    if rate is not None:
        return f"{rate:,} chunks/min"
    return "—"

"""Server-side embed throughput samples for admin ingest velocity."""

from __future__ import annotations

import time
from collections import deque

MIN_ELAPSED_S = 4.8
WINDOW_5_S = 5 * 60
WINDOW_15_S = 15 * 60
# Do not label a window rate until samples span at least this fraction of the window.
MIN_WINDOW_FRACTION = 0.9

# Filled on each /api/ingest/status poll (~8s while active). ~120 samples ≈ 16 minutes.
_samples: deque[tuple[float, int]] = deque(maxlen=120)


def reset_embed_throughput() -> None:
    """Clear samples (tests only)."""
    _samples.clear()


def record_embed_progress(total_chunks: int, *, now: float | None = None) -> None:
    """Record corpus chunk total; monotonic while ingest embeds new chunks."""
    ts = now if now is not None else time.time()
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


def _rate_over_window(window_s: float) -> int | None:
    if len(_samples) < 2:
        return None
    current = _samples[-1]
    baseline = _baseline_for_window(window_s)
    if baseline is None or baseline[0] == current[0]:
        return None
    actual_span = current[0] - baseline[0]
    if actual_span < window_s * MIN_WINDOW_FRACTION:
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
        "embed_rate_5m": _rate_over_window(WINDOW_5_S),
        "embed_rate_15m": _rate_over_window(WINDOW_15_S),
    }


def format_embed_rate(
    rate: int | None,
    *,
    running: int,
    pending: int,
    window: str,
) -> str:
    if rate is not None:
        return f"{rate:,} chunks/min"
    if window == "now" and running == 0 and pending > 0:
        return "waiting"
    return "—"

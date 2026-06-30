"""Per-IP login failure rate limiting for rag-admin."""

from __future__ import annotations

import time
from collections import defaultdict
from threading import Lock


class LoginRateLimiter:
    """Track failed login attempts per client IP within a sliding window."""

    def __init__(self, *, max_attempts: int, lockout_minutes: int) -> None:
        self._max_attempts = max_attempts
        self._window_sec = lockout_minutes * 60
        self._failures: dict[str, list[float]] = defaultdict(list)
        self._lock = Lock()

    def _prune(self, ip: str, now: float) -> list[float]:
        cutoff = now - self._window_sec
        recent = [ts for ts in self._failures[ip] if ts > cutoff]
        if recent:
            self._failures[ip] = recent
        elif ip in self._failures:
            del self._failures[ip]
        return recent

    def is_locked(self, ip: str, *, now: float | None = None) -> bool:
        current = time.time() if now is None else now
        with self._lock:
            return len(self._prune(ip, current)) >= self._max_attempts

    def record_failure(self, ip: str, *, now: float | None = None) -> None:
        current = time.time() if now is None else now
        with self._lock:
            self._prune(ip, current)
            self._failures[ip].append(current)

    def clear(self, ip: str) -> None:
        with self._lock:
            self._failures.pop(ip, None)

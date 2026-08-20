"""Probe ingest-critical HTTP endpoints.

An unhealthy embed or Qdrant server is infrastructure, not a ruined file.
Callers should wait and resume; they must not delete already-upserted chunks.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import httpx

from ingest.embed_lifecycle import probe_embed_url

CACHE_TTL_SEC = 5.0
PROBE_TIMEOUT_SEC = 2.0

_CACHE_LOCK = threading.Lock()
_CACHE: tuple[str, float, "EndpointHealth"] | None = None


@dataclass(frozen=True)
class ProbeResult:
    name: str
    url: str
    ok: bool
    detail: str = ""


@dataclass(frozen=True)
class EndpointHealth:
    results: tuple[ProbeResult, ...]

    @property
    def ok(self) -> bool:
        return all(result.ok for result in self.results)

    @property
    def waiting_on(self) -> str:
        return ", ".join(result.name for result in self.results if not result.ok)

    def as_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "waiting_on": self.waiting_on,
            "results": [
                {
                    "name": result.name,
                    "url": result.url,
                    "ok": result.ok,
                    "detail": result.detail,
                }
                for result in self.results
            ],
        }


def probe_qdrant(url: str, *, timeout: float = PROBE_TIMEOUT_SEC) -> ProbeResult:
    target = url.rstrip("/")
    if not target:
        return ProbeResult("qdrant", "", False, "no Qdrant URL configured")
    last_status = 0
    try:
        with httpx.Client(timeout=timeout) as client:
            for path in ("/readyz", "/healthz"):
                response = client.get(f"{target}{path}")
                last_status = response.status_code
                if response.status_code < 400:
                    return ProbeResult("qdrant", target, True)
    except Exception as exc:
        return ProbeResult("qdrant", target, False, str(exc))
    return ProbeResult("qdrant", target, False, f"HTTP {last_status}")


def probe_embeds(
    urls: list[str], *, timeout: float = PROBE_TIMEOUT_SEC
) -> ProbeResult:
    normalized = [url.rstrip("/") for url in urls if url.strip()]
    if not normalized:
        return ProbeResult("embed", "", False, "no embed URL configured")
    for url in normalized:
        if probe_embed_url(url, timeout=timeout):
            return ProbeResult("embed", url, True)
    return ProbeResult(
        "embed",
        normalized[0],
        False,
        f"none of {len(normalized)} embed URL(s) accepted embeddings",
    )


def _cache_key(embed_urls: list[str], qdrant_url: str) -> str:
    urls = [url.rstrip("/") for url in embed_urls if url.strip()]
    return "|".join((qdrant_url.rstrip("/"), *urls))


def check_ingest_endpoints(
    *,
    embed_urls: list[str],
    qdrant_url: str,
    use_cache: bool = True,
) -> EndpointHealth:
    """At least one embed URL must produce a vector; Qdrant must be ready."""
    key = _cache_key(embed_urls, qdrant_url)
    now = time.monotonic()
    global _CACHE
    if use_cache:
        with _CACHE_LOCK:
            cached = _CACHE
        if cached is not None and cached[0] == key and now - cached[1] < CACHE_TTL_SEC:
            return cached[2]
    health = EndpointHealth((probe_embeds(embed_urls), probe_qdrant(qdrant_url)))
    with _CACHE_LOCK:
        _CACHE = (key, now, health)
    return health

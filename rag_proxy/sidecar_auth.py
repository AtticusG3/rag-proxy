"""Shared-token auth for sidecar mutating routes (SIDECAR_TOKEN).

The rerank, sparse and TurboVec sidecars gate their work-doing routes on the
X-Internal-Token header when SIDECAR_TOKEN is set. Clients call this helper to
build the request headers; an empty token yields no header at all, so the
default (unset) behaves exactly as before the gate existed.
"""

from __future__ import annotations

SIDECAR_TOKEN_HEADER = "X-Internal-Token"


def sidecar_auth_headers(token: str) -> dict[str, str]:
    token = (token or "").strip()
    if not token:
        return {}
    return {SIDECAR_TOKEN_HEADER: token}

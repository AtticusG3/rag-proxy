"""Shared Qdrant payload text extraction (dense + sparse sidecar)."""

from __future__ import annotations

from typing import Any

# Field order for chunk bodies in Qdrant payloads (nomad + homelab collections).
PAYLOAD_TEXT_KEYS = ("text", "content", "chunk", "document", "page_content")


def extract_chunk_text(hit: dict[str, Any]) -> str:
    """Pull text from a Qdrant hit or sparse sidecar result row."""
    payload = hit.get("payload", {})
    for key in PAYLOAD_TEXT_KEYS:
        value = payload.get(key)
        if value:
            return str(value)
    return str(payload) if payload else ""

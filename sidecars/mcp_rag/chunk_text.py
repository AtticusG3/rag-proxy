"""Qdrant payload text extraction (shared with sparse sidecar)."""

from __future__ import annotations

from typing import Any

PAYLOAD_TEXT_KEYS = ("text", "content", "chunk", "document", "page_content")


def extract_chunk_text(hit: dict[str, Any]) -> str:
    payload = hit.get("payload") or {}
    for key in PAYLOAD_TEXT_KEYS:
        value = payload.get(key)
        if value:
            return str(value)
    return str(payload) if payload else ""

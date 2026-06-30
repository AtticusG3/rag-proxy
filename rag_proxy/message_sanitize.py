"""Sanitize chat messages before transcript capture or dataset export."""

from __future__ import annotations

import copy
from typing import Any

from rag_proxy.legacy_rag import is_embeddable_user_query
from rag_proxy.rag_context import RAG_CONTEXT_PREFIX

ROLLING_MEMORY_PREFIX = "Operational memory (session):\n"


def _strip_prefixed_block(content: str, prefix: str) -> str:
    if not content.startswith(prefix):
        return content
    remainder = content[len(prefix):]
    marker = "\n\n"
    if marker not in remainder:
        return ""
    return remainder.split(marker, 1)[1]


def _strip_first_system_prefix(
    messages: list[dict[str, Any]],
    prefix: str,
) -> list[dict[str, Any]]:
    if not messages:
        return messages
    first = messages[0]
    if first.get("role") == "system" and isinstance(first.get("content"), str):
        first["content"] = _strip_prefixed_block(first["content"], prefix).strip()
    return messages


def _strip_proxy_artefacts(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = copy.deepcopy(messages)
    for prefix in (RAG_CONTEXT_PREFIX, ROLLING_MEMORY_PREFIX):
        _strip_first_system_prefix(out, prefix)
    return out


def _drop_empty_system_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        msg
        for msg in messages
        if not (msg.get("role") == "system" and str(msg.get("content", "")).strip() == "")
    ]


def sanitize_client_messages(
    messages: list[dict[str, Any]],
    *,
    strip_proxy_artefacts: bool = True,
) -> list[dict[str, Any]]:
    """Return client messages safe for transcript capture."""
    out = _strip_proxy_artefacts(messages) if strip_proxy_artefacts else copy.deepcopy(messages)
    return _drop_empty_system_messages(out)


def is_exportable_turn(query_text: str | None, assistant_text: str | None) -> bool:
    """True when a turn has user intent and assistant output worth training on."""
    if not query_text or not query_text.strip():
        return False
    if not assistant_text or not assistant_text.strip():
        return False
    return is_embeddable_user_query(query_text)

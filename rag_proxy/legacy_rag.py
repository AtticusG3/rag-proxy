"""Core RAG helpers (message extraction, injection) and retrieval re-exports."""

from __future__ import annotations

from rag_proxy.clients.retrieval_async import get_embedding, search_qdrant_dense
from rag_proxy.clients.retrieval_core import prepare_embed_text
from rag_proxy.rag_context import RAG_CONTEXT_PREFIX

__all__ = [
    "extract_query_text",
    "get_embedding",
    "inject_context",
    "is_embeddable_user_query",
    "prepare_embed_text",
    "search_qdrant_dense",
    "user_message_text",
]


def user_message_text(msg: dict) -> str:
    content = msg.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return " ".join(
            block.get("text", "")
            for block in content
            if block.get("type") == "text"
        ).strip()
    return ""


def is_embeddable_user_query(text: str) -> bool:
    """Skip UI meta-prompts (e.g. Open WebUI follow-up suggestion tasks)."""
    head = text.lstrip()[:400].lower()
    if head.startswith("### task:"):
        return False
    if "follow-up" in head and "suggest" in head:
        return False
    if "relevant follow-up questions" in head:
        return False
    return True


def extract_query_text(messages: list[dict]) -> str | None:
    """Last real user turn — skip automated ### Task: / follow-up prompts."""
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        text = user_message_text(msg)
        if text and is_embeddable_user_query(text):
            return text
    return None


def inject_context(messages: list[dict], chunks: list[str]) -> list[dict]:
    """Prepend retrieved context as a system message (or prefix an existing one)."""
    context_block = "\n\n---\n\n".join(chunks)
    rag_prefix = f"{RAG_CONTEXT_PREFIX}{context_block}"
    messages = list(messages)
    if messages and messages[0]["role"] == "system":
        messages[0] = {**messages[0], "content": rag_prefix + "\n\n" + messages[0]["content"]}
    else:
        messages.insert(0, {"role": "system", "content": rag_prefix})
    return messages

"""Shared RAG injection text used by retrieval and transcript sanitization."""

from __future__ import annotations

RAG_CONTEXT_PREFIX = (
    "The following context was retrieved from the local knowledge base. "
    "Use it to inform your response where relevant. "
    "Do not mention the knowledge base unless the user asks.\n\n"
)
ROLLING_MEMORY_PREFIX = "Operational memory (session):\n"

# Marker lines make an injected block self-delimiting, so chunks may hold blank lines.
RAG_BLOCK_START = "<rag-context-start>"
RAG_BLOCK_END = "<rag-context-end>"


def wrap_injected_block(prefix: str, body: str) -> str:
    """Prefix, then body between marker lines."""
    return f"{prefix}{RAG_BLOCK_START}\n{body}\n{RAG_BLOCK_END}"

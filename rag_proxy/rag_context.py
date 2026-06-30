"""Shared RAG injection text used by retrieval and transcript sanitization."""

from __future__ import annotations

RAG_CONTEXT_PREFIX = (
    "The following context was retrieved from the local knowledge base. "
    "Use it to inform your response where relevant. "
    "Do not mention the knowledge base unless the user asks.\n\n"
)

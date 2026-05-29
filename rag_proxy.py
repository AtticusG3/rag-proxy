#!/usr/bin/env python3
"""
rag_proxy.py -- shim entrypoint and backward-compatible exports.

Run: python3 rag_proxy.py
Or:  python3 -m rag_proxy
"""

from __future__ import annotations

from rag_proxy.app import app, main
from rag_proxy.config import (
    CHAT_PATHS,
    EMBED_MAX_CHARS,
    EMBED_RETRIES,
    EMBED_URL,
    LLAMA_SWAP_URL,
    PROXY_HOST,
    PROXY_PORT,
    QDRANT_COLLECTION,
    QDRANT_URL,
    SIMILARITY_THRESHOLD,
    TOP_K,
)
from rag_proxy.legacy_rag import (
    extract_query_text,
    get_embedding,
    inject_context,
    is_embeddable_user_query,
    prepare_embed_text,
    search_qdrant_dense as search_qdrant,
    user_message_text,
)

__all__ = [
    "app",
    "main",
    "CHAT_PATHS",
    "extract_query_text",
    "inject_context",
    "get_embedding",
    "search_qdrant",
    "prepare_embed_text",
    "is_embeddable_user_query",
    "user_message_text",
]

if __name__ == "__main__":
    main()

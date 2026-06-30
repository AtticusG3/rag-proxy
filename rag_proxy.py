#!/usr/bin/env python3
"""
rag_proxy.py -- shim entrypoint and backward-compatible exports.

Run: python3 rag_proxy.py
Or:  python3 -m rag_proxy
"""

from __future__ import annotations

from rag_proxy.app import app, main
from rag_proxy.config import CHAT_PATHS, settings
from rag_proxy.legacy_rag import (
    extract_query_text,
    get_embedding,
    inject_context,
    is_embeddable_user_query,
    prepare_embed_text,
    search_qdrant_dense as search_qdrant,
    user_message_text,
)

# Backward-compatible module-level aliases for scripts and tests.
LLAMA_SWAP_URL = settings.llama_swap_url
EMBED_URL = settings.embed_url
QDRANT_URL = settings.qdrant_url
QDRANT_COLLECTION = settings.qdrant_collection
TOP_K = settings.top_k
SIMILARITY_THRESHOLD = settings.similarity_threshold
PROXY_HOST = settings.proxy_host
PROXY_PORT = settings.proxy_port
EMBED_MAX_CHARS = settings.embed_max_chars
EMBED_RETRIES = settings.embed_retries

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
    "LLAMA_SWAP_URL",
    "EMBED_URL",
    "QDRANT_URL",
    "QDRANT_COLLECTION",
    "TOP_K",
    "SIMILARITY_THRESHOLD",
    "PROXY_HOST",
    "PROXY_PORT",
    "EMBED_MAX_CHARS",
    "EMBED_RETRIES",
]

if __name__ == "__main__":
    main()

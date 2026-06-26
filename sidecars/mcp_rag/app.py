#!/usr/bin/env python3
"""MCP server exposing hybrid RAG retrieval as agent tools."""

from __future__ import annotations

import json
import logging
import os

from mcp.server.fastmcp import FastMCP

from retrieve import (
    RetrieveSettings,
    fetch_index_status,
    format_chunks_for_agent,
    search_knowledge_base as run_retrieval,
)

log = logging.getLogger("mcp-rag-context")

_HOST = os.getenv("MCP_HOST", "127.0.0.1")
_PORT = int(os.getenv("MCP_PORT", "9001"))

mcp = FastMCP(
    "RAG Knowledge Base",
    instructions=(
        "Search the local offline knowledge base (ZIM archives, PDFs, and text files). "
        "Use search_knowledge_base before answering questions that "
        "may depend on ingested documentation."
    ),
    host=_HOST,
    port=_PORT,
    streamable_http_path="/mcp",
)


@mcp.tool(name="search_knowledge_base")
def search_knowledge_base_tool(
    query: str,
    top_k: int = 5,
    min_score: float = 0.0,
) -> str:
    """Retrieve relevant passages from the indexed knowledge base.

    Args:
        query: Natural-language search query.
        top_k: Number of passages to return (1-20).
        min_score: Optional minimum dense similarity threshold (0 disables).
    """
    limit = max(1, min(int(top_k), 20))
    threshold = float(min_score) if min_score > 0 else None
    chunks = run_retrieval(
        query.strip(),
        top_k=limit,
        score_threshold=threshold,
    )
    return format_chunks_for_agent(chunks)


@mcp.tool()
def knowledge_base_status() -> str:
    """Report Qdrant vector count and sparse BM25 index health."""
    status = fetch_index_status()
    return json.dumps(status, indent=2)


def main() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = RetrieveSettings.from_env()
    transport = os.getenv("MCP_TRANSPORT", "streamable-http").strip().lower()
    log.info(
        "MCP RAG context starting transport=%s %s:%s collection=%s",
        transport,
        _HOST,
        _PORT,
        cfg.qdrant_collection,
    )

    if transport == "stdio":
        mcp.run(transport="stdio")
        return

    if transport in ("streamable-http", "http"):
        mcp.run(transport="streamable-http")
        return

    if transport == "sse":
        mcp.run(transport="sse")
        return

    raise SystemExit(f"Unsupported MCP_TRANSPORT: {transport}")


if __name__ == "__main__":
    main()

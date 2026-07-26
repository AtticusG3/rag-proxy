#!/usr/bin/env python3
"""MCP server exposing hybrid RAG retrieval and personal memory as agent tools."""

from __future__ import annotations

import json
import logging
import os

from mcp.server.fastmcp import FastMCP

from personal_store import (
    default_store_path,
    forget_note,
    format_notes_for_agent,
    recall_notes,
    store_note,
)
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
        "Search the local offline knowledge base (ZIM archives, PDFs, and text files) "
        "with search_knowledge_base. Use mode=facts for MemGraphRAG entity triples when "
        "the graph index is available. Personal notes use memory_store / memory_recall "
        "(separate from the curated knowledge base — never write into Qdrant via these tools)."
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
    mode: str = "passages",
) -> str:
    """Retrieve relevant passages (or MemGraph facts) from the indexed knowledge base.

    Args:
        query: Natural-language search query.
        top_k: Number of passages/facts to return (1-20).
        min_score: Minimum dense cosine similarity for the dense retrieval leg only
            (0 disables). Does not filter sparse BM25 or MemGraph fact scores.
        mode: Retrieval mode — passages (default hybrid/dense policy), dense, sparse,
            or facts (MemGraphRAG triples; empty if index unavailable).
    """
    limit = max(1, min(int(top_k), 20))
    threshold = float(min_score) if min_score > 0 else None
    chunks = run_retrieval(
        query.strip(),
        top_k=limit,
        score_threshold=threshold,
        mode=mode,
    )
    return format_chunks_for_agent(chunks)


@mcp.tool()
def knowledge_base_status() -> str:
    """Report Qdrant, sparse, embed, and MemGraphRAG index health."""
    status = fetch_index_status()
    return json.dumps(status, indent=2)


@mcp.tool()
def memory_store(key: str, text: str, tags: str = "") -> str:
    """Upsert a personal agent note (Hermes/OpenClaw local_store). Not the curated KB.

    Args:
        key: Stable note id (e.g. project preference name).
        text: Note body to remember.
        tags: Optional space-separated tags.
    """
    try:
        note = store_note(key, text, tags=tags)
    except ValueError as exc:
        return f"memory_store error: {exc}"
    return json.dumps(
        {
            "ok": True,
            "key": note.key,
            "updated_at": note.updated_at,
            "store": str(default_store_path()),
        },
        indent=2,
    )


@mcp.tool()
def memory_recall(query: str, top_k: int = 5) -> str:
    """Recall personal notes by lexical match. Does not search the curated knowledge base.

    Args:
        query: Search string (empty returns newest notes).
        top_k: Max notes to return (1-50).
    """
    notes = recall_notes(query, top_k=top_k)
    return format_notes_for_agent(notes)


@mcp.tool()
def memory_forget(key: str) -> str:
    """Delete a personal note by key. Does not modify the curated knowledge base.

    Args:
        key: Note id previously stored via memory_store.
    """
    deleted = forget_note(key)
    return json.dumps({"ok": deleted, "key": key.strip()}, indent=2)


def main() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = RetrieveSettings.from_env()
    transport = os.getenv("MCP_TRANSPORT", "streamable-http").strip().lower()
    log.info(
        "MCP RAG context starting transport=%s %s:%s collection=%s personal_store=%s",
        transport,
        _HOST,
        _PORT,
        cfg.qdrant_collection,
        default_store_path(),
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

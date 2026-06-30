"""Shared ingest queue presentation for admin UI and API."""

from __future__ import annotations

import os
from typing import Any

from ingest.stall import is_stalled


def enrich_file_rows(
    rows: list[dict[str, Any]],
    *,
    stall_seconds: int,
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        stalled = item.get("status") == "running" and is_stalled(
            item.get("updated_at"), stall_seconds
        )
        item["is_stalled"] = stalled
        item["display_status"] = "stalled" if stalled else item.get("status", "")
        if not item.get("file_name"):
            path = str(item.get("file_path", ""))
            item["file_name"] = path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        path = str(item.get("file_path", ""))
        item["file_missing"] = bool(path) and not os.path.isfile(path)
        enriched.append(item)
    return enriched


def ingest_queue_stats(files: list[dict[str, Any]]) -> dict[str, int]:
    pending = 0
    running = 0
    stalled = 0
    indexed = 0
    total_chunks = 0
    missing = 0
    for row in files:
        status = row.get("status", "")
        display = row.get("display_status", status)
        chunks = int(row.get("chunks_embedded") or 0)
        total_chunks += chunks
        if row.get("file_missing"):
            missing += 1
        if status in ("pending", "queued"):
            pending += 1
        elif status == "running":
            running += 1
            if display == "stalled":
                stalled += 1
        elif status == "indexed":
            indexed += 1
        elif status == "failed":
            pass
    return {
        "pending": pending,
        "running": running,
        "stalled": stalled,
        "indexed": indexed,
        "missing": missing,
        "active": pending + running,
        "total_chunks": total_chunks,
    }


def ingest_config_snapshot(worker: Any) -> dict[str, Any]:
    config = worker.config
    pool_urls = config.embed_urls or []
    chunk = config.chunk_config
    return {
        "batch_size": config.batch_size,
        "embed_concurrency": config.embed_concurrency,
        "file_concurrency": config.file_concurrency,
        "embed_max_chars": config.embed_max_chars,
        "embed_url": config.embed_url,
        "embed_pool_count": len(pool_urls) if pool_urls else 1,
        "sparse_reindex_mode": config.sparse_reindex_mode,
        "stall_minutes": config.stall_seconds // 60,
        "qdrant_collection": config.qdrant_collection,
        "chunk_size_tokens": chunk.chunk_size,
        "chunk_overlap_tokens": chunk.chunk_overlap,
        "chunk_semantic": "on" if chunk.semantic_enabled else "off",
        "paused": worker.paused,
    }

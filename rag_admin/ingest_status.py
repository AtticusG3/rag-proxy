"""Shared ingest queue presentation for admin UI and API."""

from __future__ import annotations

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
        enriched.append(item)
    return enriched


def ingest_queue_stats(files: list[dict[str, Any]]) -> dict[str, int]:
    pending = 0
    running = 0
    stalled = 0
    indexed = 0
    total_chunks = 0
    for row in files:
        status = row.get("status", "")
        display = row.get("display_status", status)
        chunks = int(row.get("chunks_embedded") or 0)
        total_chunks += chunks
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
        "active": pending + running,
        "total_chunks": total_chunks,
    }


def ingest_config_snapshot(worker: Any) -> dict[str, Any]:
    config = worker.config
    return {
        "batch_size": config.batch_size,
        "embed_concurrency": config.embed_concurrency,
        "embed_max_chars": config.embed_max_chars,
        "embed_url": config.embed_url,
        "sparse_reindex_mode": config.sparse_reindex_mode,
        "stall_minutes": config.stall_seconds // 60,
        "qdrant_collection": config.qdrant_collection,
        "paused": worker.paused,
    }

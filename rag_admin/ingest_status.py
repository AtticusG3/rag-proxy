"""Shared ingest queue presentation for admin UI and API."""

from __future__ import annotations

import os
from typing import Any

from ingest.db import DEFAULT_PRIORITY
from ingest.stall import is_stalled

from rag_admin.embed_throughput import (
    embed_throughput_rates,
    format_embed_rate,
    record_embed_progress,
)

# Sort keys accepted by the /jobs UI. Keep in sync with sortable headers in jobs.html.
SORT_KEYS = ("name", "priority", "status", "size", "updated")
DEFAULT_SORT = "updated"
DEFAULT_SORT_DIR = "desc"

# Lower rank = higher priority. Keep in sync with ingest.db._PRIORITY_ORDER_SQL.
_PRIORITY_RANK = {"high": 0, "mid": 1, "low": 2}


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
        item["priority"] = item.get("priority") or DEFAULT_PRIORITY
        item["file_size"] = _file_size(path) if not item["file_missing"] else None
        enriched.append(item)
    return enriched


def resolve_sort(sort: str | None, direction: str | None) -> tuple[str, str]:
    """Normalize sort key/direction from query params to a valid pair."""
    key = sort if sort in SORT_KEYS else DEFAULT_SORT
    order = direction if direction in ("asc", "desc") else DEFAULT_SORT_DIR
    return key, order


def _file_size(path: str) -> int | None:
    if not path:
        return None
    try:
        return os.path.getsize(path)
    except OSError:
        return None


def sort_file_rows(
    rows: list[dict[str, Any]],
    *,
    sort: str = DEFAULT_SORT,
    direction: str = DEFAULT_SORT_DIR,
) -> list[dict[str, Any]]:
    """Order enriched file rows for display. Falls back to defaults on bad input."""
    if sort not in SORT_KEYS:
        sort = DEFAULT_SORT
    reverse = direction != "asc"

    def key(row: dict[str, Any]) -> Any:
        if sort == "name":
            return (row.get("file_name") or "").lower()
        if sort == "priority":
            return _PRIORITY_RANK.get(row.get("priority") or DEFAULT_PRIORITY, 1)
        if sort == "status":
            return str(row.get("display_status") or row.get("status") or "")
        if sort == "size":
            return row.get("file_size") or 0
        return str(row.get("updated_at") or "")

    return sorted(rows, key=key, reverse=reverse)


def ingest_queue_stats(files: list[dict[str, Any]]) -> dict[str, int | None]:
    pending = 0
    running = 0
    stalled = 0
    indexed = 0
    total_chunks = 0
    queue_chunks = 0
    missing = 0
    for row in files:
        status = row.get("status", "")
        display = row.get("display_status", status)
        chunks = int(row.get("chunks_embedded") or 0)
        total_chunks += chunks
        if status in ("pending", "queued", "running"):
            queue_chunks += chunks
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
    record_embed_progress(total_chunks)
    stats = {
        "pending": pending,
        "running": running,
        "stalled": stalled,
        "indexed": indexed,
        "missing": missing,
        "active": pending + running,
        "total_chunks": total_chunks,
        "queue_chunks": queue_chunks,
    }
    stats.update(embed_throughput_rates())
    stats["velocity_text"] = ingest_velocity_text(stats)
    return stats


def ingest_velocity_text(stats: dict[str, int | None]) -> str:
    """Single-line ingest velocity summary for Jobs page."""
    if int(stats.get("active") or 0) <= 0:
        indexed = int(stats.get("indexed") or 0)
        total = int(stats.get("total_chunks") or 0)
        return f"{indexed:,} indexed · {total:,} corpus chunks"

    active = int(stats["active"])
    total = int(stats.get("total_chunks") or 0)
    running = int(stats.get("running") or 0)
    pending = int(stats.get("pending") or 0)
    parts = [f"{active} in queue", f"{total:,} corpus chunks"]
    if running:
        parts.append(f"{running} embedding")
    parts.extend(
        [
            "now " + format_embed_rate(stats.get("embed_rate_now"), running=running, pending=pending, window="now"),
            "5m " + format_embed_rate(stats.get("embed_rate_5m"), running=running, pending=pending, window="5m"),
            "15m " + format_embed_rate(stats.get("embed_rate_15m"), running=running, pending=pending, window="15m"),
        ]
    )
    return " · ".join(parts)


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

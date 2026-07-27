"""Shared ingest queue presentation for admin UI and API."""

from __future__ import annotations

import os
from typing import Any

from ingest.db import DEFAULT_PRIORITY
from ingest.queue_order import order_queue_rows
from ingest.stall import is_stalled, seconds_since_update

from rag_admin.embed_throughput import (
    embed_throughput_rates,
    format_primary_rate,
    record_embed_progress,
)

# Sort keys accepted by the /jobs UI. Keep in sync with sortable headers in jobs.html.
SORT_KEYS = ("name", "priority", "status", "size", "updated")
DEFAULT_SORT = "updated"
DEFAULT_SORT_DIR = "desc"

def truthy_query_flag(value: str | None) -> bool:
    return (value or "").strip().lower() in ("1", "true", "yes", "on")


def filter_visible_file_rows(
    rows: list[dict[str, Any]],
    *,
    hide_indexed_seconds: int,
    show_indexed: bool = False,
) -> tuple[list[dict[str, Any]], int]:
    """Hide indexed rows older than hide_indexed_seconds from the Jobs table.

    Stats should be computed on the unfiltered list first. Returns
    (visible_rows, hidden_indexed_count). hide_indexed_seconds <= 0 disables
    filtering. show_indexed=True shows every indexed row.
    """
    if show_indexed or hide_indexed_seconds <= 0:
        return list(rows), 0

    visible: list[dict[str, Any]] = []
    hidden = 0
    for row in rows:
        if row.get("status") == "indexed":
            age = seconds_since_update(row.get("updated_at"))
            if age is None or age > hide_indexed_seconds:
                hidden += 1
                continue
        visible.append(row)
    return visible, hidden


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
    if direction not in ("asc", "desc"):
        direction = DEFAULT_SORT_DIR
    return order_queue_rows(rows, sort=sort, direction=direction)


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
        "turbovec_reindex_mode": getattr(config, "turbovec_reindex_mode", "idle"),
        "turbovec_configured": bool(getattr(config, "turbovec_url", "").strip()),
        "stall_minutes": config.stall_seconds // 60,
        "qdrant_collection": config.qdrant_collection,
        "chunk_size_tokens": chunk.chunk_size,
        "chunk_overlap_tokens": chunk.chunk_overlap,
        "chunk_semantic": "on" if chunk.semantic_enabled else "off",
        "paused": worker.paused,
    }


def _health_count(body: Any, *keys: str) -> int | None:
    if not isinstance(body, dict):
        return None
    for key in keys:
        if key in body and body[key] is not None:
            try:
                return int(body[key])
            except (TypeError, ValueError):
                continue
    return None


def derive_sidecar_phase(
    *,
    kind: str,
    configured: bool,
    mode: str,
    ok: bool | None,
    dirty: bool,
    reindexing: bool,
    queue_active: bool,
    dual_write: bool = False,
    on_demand: bool = False,
) -> str:
    """Human phase label for Jobs BM25 / TurboVec panels."""
    if not configured:
        return "unconfigured"
    if reindexing:
        return "reindexing"
    if kind == "turbovec" and dual_write and queue_active:
        # Dual-write keeps TurboVec current during ingest; a failed per-file
        # full reindex (each mode at corpus scale) should not mask that.
        return "dual_write"
    if dirty:
        return "pending_reindex"
    if ok is False:
        return "down"
    if mode == "off":
        return "off"
    return "idle"


def build_sidecars_payload(
    worker: Any,
    *,
    queue_active: int,
    sparse_health: dict[str, Any] | None = None,
    turbovec_health: dict[str, Any] | None = None,
    on_demand: bool | None = None,
) -> dict[str, Any]:
    """Merge worker scheduler status with optional /health probe results."""
    from ingest.sidecar_lifecycle import sidecar_on_demand_enabled

    if on_demand is None:
        on_demand = sidecar_on_demand_enabled()

    raw = worker.sidecar_status() if hasattr(worker, "sidecar_status") else {
        "sparse": {"configured": False, "mode": "off", "dirty": False, "reindexing": False},
        "turbovec": {
            "configured": False,
            "mode": "off",
            "dirty": False,
            "reindexing": False,
            "dual_write": False,
        },
    }
    active = int(queue_active or 0) > 0

    sparse_h = sparse_health or {}
    sparse_body = sparse_h.get("body") if isinstance(sparse_h, dict) else None
    sparse_ok = sparse_h.get("ok") if sparse_h else None
    sparse = dict(raw.get("sparse") or {})
    sparse_configured = bool(sparse.get("configured"))
    sparse["ok"] = bool(sparse_ok) if sparse_configured and sparse_h else None
    sparse["docs"] = _health_count(sparse_body, "docs")
    sparse["last_sync"] = (
        sparse_body.get("last_sync") if isinstance(sparse_body, dict) else None
    )
    sparse["phase"] = derive_sidecar_phase(
        kind="sparse",
        configured=sparse_configured,
        mode=str(sparse.get("mode") or "idle"),
        ok=sparse["ok"],
        dirty=bool(sparse.get("dirty")),
        reindexing=bool(sparse.get("reindexing")),
        queue_active=active,
        on_demand=bool(on_demand),
    )

    tv_h = turbovec_health or {}
    tv_body = tv_h.get("body") if isinstance(tv_h, dict) else None
    tv_ok = tv_h.get("ok") if tv_h else None
    turbovec = dict(raw.get("turbovec") or {})
    tv_configured = bool(turbovec.get("configured"))
    turbovec["ok"] = bool(tv_ok) if tv_configured and tv_h else None
    turbovec["vectors"] = _health_count(tv_body, "vectors", "docs")
    turbovec["phase"] = derive_sidecar_phase(
        kind="turbovec",
        configured=tv_configured,
        mode=str(turbovec.get("mode") or "idle"),
        ok=turbovec["ok"],
        dirty=bool(turbovec.get("dirty")),
        reindexing=bool(turbovec.get("reindexing")),
        queue_active=active,
        dual_write=bool(turbovec.get("dual_write")),
        on_demand=bool(on_demand),
    )

    return {"bm25": sparse, "turbovec": turbovec}


async def load_sidecars_payload(worker: Any, *, queue_active: int) -> dict[str, Any]:
    """Probe BM25 / TurboVec /health and merge with worker scheduler state."""
    from rag_admin.service_status import probe_url

    sparse_url = (getattr(worker.config, "sparse_index_url", "") or "").strip()
    tv_url = (getattr(worker.config, "turbovec_url", "") or "").strip()
    sparse_health = await probe_url(sparse_url) if sparse_url else None
    turbovec_health = await probe_url(tv_url) if tv_url else None
    return build_sidecars_payload(
        worker,
        queue_active=queue_active,
        sparse_health=sparse_health,
        turbovec_health=turbovec_health,
    )


_PHASE_VELOCITY = {
    "reindexing": "reindexing",
    "pending_reindex": "pending",
    "dual_write": "dual-write",
}


def sidecars_velocity_clause(sidecars: dict[str, Any]) -> str:
    """Compact BM25/TurboVec clause for the Jobs velocity line."""
    parts: list[str] = []
    bm25 = sidecars.get("bm25") or {}
    tv = sidecars.get("turbovec") or {}
    bm25_label = _PHASE_VELOCITY.get(str(bm25.get("phase") or ""))
    if bm25_label:
        parts.append(f"BM25 {bm25_label}")
    tv_label = _PHASE_VELOCITY.get(str(tv.get("phase") or ""))
    if tv_label:
        parts.append(f"TurboVec {tv_label}")
    return " · ".join(parts)


def sidecars_activity_active(sidecars: dict[str, Any]) -> bool:
    """True when Live badge should stay on for sidecar work."""
    for key in ("bm25", "turbovec"):
        row = sidecars.get(key) or {}
        if row.get("reindexing") or row.get("dirty"):
            return True
        if row.get("phase") in ("reindexing", "pending_reindex", "dual_write"):
            return True
    return False


def ingest_velocity_text(
    stats: dict[str, int | None],
    *,
    sidecars: dict[str, Any] | None = None,
) -> str:
    """Single-line ingest velocity summary for Jobs page."""
    if int(stats.get("active") or 0) <= 0:
        indexed = int(stats.get("indexed") or 0)
        total = int(stats.get("total_chunks") or 0)
        base = f"{indexed:,} indexed · {total:,} corpus chunks"
    else:
        active = int(stats["active"])
        total = int(stats.get("total_chunks") or 0)
        running = int(stats.get("running") or 0)
        pending = int(stats.get("pending") or 0)
        parts = [f"{active} in queue", f"{total:,} corpus chunks"]
        if running:
            parts.append(f"{running} embedding")
        primary = format_primary_rate(
            stats.get("embed_rate_now"),
            running=running,
            pending=pending,
        )
        if primary:
            parts.append(primary)
        rate_5m = stats.get("embed_rate_5m")
        if rate_5m is not None and int(rate_5m) > 0:
            parts.append(f"5m avg {int(rate_5m):,} chunks/min")
        base = " · ".join(parts)

    if sidecars:
        clause = sidecars_velocity_clause(sidecars)
        if clause:
            return f"{base} · {clause}"
    return base


def ingest_queue_stats(
    files: list[dict[str, Any]],
    *,
    sidecars: dict[str, Any] | None = None,
) -> dict[str, int | None]:
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
    stats: dict[str, Any] = {
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
    stats["velocity_text"] = ingest_velocity_text(stats, sidecars=sidecars)
    return stats

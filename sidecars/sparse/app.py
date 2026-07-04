#!/usr/bin/env python3
"""BM25 sparse index sidecar; syncs payloads from Qdrant (POST /search)."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from contextlib import asynccontextmanager
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from core import DEFAULT_COLLECTION as _DEFAULT_COLLECTION
from core import IndexRegistry, SparseIndex

try:
    from rag_proxy.chunk_text import PAYLOAD_TEXT_KEYS
except ImportError:
    from chunk_text import PAYLOAD_TEXT_KEYS  # noqa: F401 — Docker flat layout

log = logging.getLogger("sparse-sidecar")

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333").rstrip("/")
DEFAULT_COLLECTION = os.getenv("QDRANT_COLLECTION", _DEFAULT_COLLECTION)
REFRESH_SEC = float(os.getenv("SPARSE_REFRESH_SEC", "3600"))
SCROLL_BATCH = int(os.getenv("SPARSE_SCROLL_BATCH", "256"))
HOST = os.getenv("SPARSE_HOST", "0.0.0.0")
PORT = int(os.getenv("SPARSE_PORT", "8096"))
# 0 = index everything. Otherwise stop after N points (BM25 sampling for large collections).
MAX_POINTS = int(os.getenv("SPARSE_MAX_POINTS", "0"))

# Payload keys fetched from Qdrant (text + recency metadata only).
_SCROLL_PAYLOAD_KEYS = tuple(
    dict.fromkeys(
        (*PAYLOAD_TEXT_KEYS, "updated_at", "mtime", "timestamp"),
    )
)

registry = IndexRegistry()


async def _scroll_page(
    client: httpx.AsyncClient,
    collection: str,
    offset: str | int | None,
    limit: int,
) -> tuple[list[dict[str, Any]], str | int | None]:
    body: dict[str, Any] = {
        "limit": limit,
        "with_payload": list(_SCROLL_PAYLOAD_KEYS),
        "with_vector": False,
    }
    if offset is not None:
        body["offset"] = offset
    response = await client.post(
        f"{QDRANT_URL}/collections/{collection}/points/scroll",
        json=body,
    )
    response.raise_for_status()
    result = response.json().get("result", {})
    batch = result.get("points", [])
    return batch, result.get("next_page_offset")


async def sync_collection(collection: str) -> int:
    index = SparseIndex()
    indexed = 0
    truncated = False
    offset: str | int | None = None

    async with httpx.AsyncClient(timeout=60.0) as client:
        while True:
            remaining = SCROLL_BATCH
            if MAX_POINTS > 0:
                remaining = min(SCROLL_BATCH, MAX_POINTS - indexed)
                if remaining <= 0:
                    truncated = True
                    break

            batch, offset = await _scroll_page(client, collection, offset, remaining)
            if batch:
                await asyncio.to_thread(index.add_points, batch)
                indexed += len(batch)

            if offset is None:
                break
            if MAX_POINTS > 0 and indexed >= MAX_POINTS:
                truncated = True
                break

    count = await asyncio.to_thread(_finalize_and_install, collection, index)
    if truncated:
        log.info(
            "Sparse index synced collection=%s docs=%d (truncated at MAX_POINTS=%d)",
            collection,
            count,
            MAX_POINTS,
        )
    else:
        log.info("Sparse index synced collection=%s docs=%d", collection, count)
    return count


def _finalize_and_install(collection: str, index: SparseIndex) -> int:
    index.finalize(collection)
    return registry.install(collection, index)


class SearchRequest(BaseModel):
    query: str
    limit: int = 20
    collection: str = Field(default_factory=lambda: DEFAULT_COLLECTION)


class ReindexRequest(BaseModel):
    collection: str = Field(default_factory=lambda: DEFAULT_COLLECTION)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    initial_sync_failed = False
    try:
        await sync_collection(DEFAULT_COLLECTION)
    except Exception as exc:
        initial_sync_failed = True
        log.warning("Initial sparse sync failed (will retry): %s", exc)

    refresh_task: asyncio.Task | None = None
    if REFRESH_SEC > 0:
        initial_delay = 5.0 if initial_sync_failed else REFRESH_SEC
        refresh_task = asyncio.create_task(_refresh_loop(initial_delay))

    yield

    if refresh_task is not None:
        refresh_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await refresh_task


app = FastAPI(title="RAG Sparse Index Sidecar", docs_url=None, redoc_url=None, lifespan=lifespan)


async def _refresh_loop(initial_delay: float) -> None:
    delay = initial_delay
    while True:
        await asyncio.sleep(delay)
        delay = REFRESH_SEC
        collection = registry.loaded_collection() or DEFAULT_COLLECTION
        try:
            await sync_collection(collection)
        except Exception as exc:
            log.warning("Periodic sparse sync failed: %s", exc)


@app.get("/health")
def health() -> dict[str, Any]:
    collection = registry.loaded_collection() or DEFAULT_COLLECTION
    return {
        "status": "ok",
        "collection": collection,
        "docs": registry.doc_count(collection),
        "last_sync": registry.last_sync(collection),
        "max_points": MAX_POINTS,
        "truncated": bool(MAX_POINTS) and registry.doc_count(collection) >= MAX_POINTS,
    }


@app.post("/search")
def search(body: SearchRequest) -> dict[str, list[dict[str, Any]]]:
    limit = max(1, min(body.limit, 100))
    return {"results": registry.search(body.collection, body.query, limit)}


@app.post("/reindex")
async def reindex(body: ReindexRequest) -> dict[str, Any]:
    try:
        count = await sync_collection(body.collection)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Qdrant sync failed: {exc}") from exc
    return {"collection": body.collection, "docs": count}


def main() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    log.info("Sparse sidecar listening on %s:%s qdrant=%s", HOST, PORT, QDRANT_URL)
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")


if __name__ == "__main__":
    main()

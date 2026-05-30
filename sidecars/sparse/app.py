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
from core import IndexRegistry

log = logging.getLogger("sparse-sidecar")

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333").rstrip("/")
DEFAULT_COLLECTION = os.getenv("QDRANT_COLLECTION", _DEFAULT_COLLECTION)
REFRESH_SEC = float(os.getenv("SPARSE_REFRESH_SEC", "3600"))
SCROLL_BATCH = int(os.getenv("SPARSE_SCROLL_BATCH", "256"))
HOST = os.getenv("SPARSE_HOST", "0.0.0.0")
PORT = int(os.getenv("SPARSE_PORT", "8096"))

registry = IndexRegistry()


async def fetch_qdrant_points(collection: str) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    offset: str | int | None = None
    async with httpx.AsyncClient(timeout=60.0) as client:
        while True:
            body: dict[str, Any] = {
                "limit": SCROLL_BATCH,
                "with_payload": True,
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
            points.extend(batch)
            offset = result.get("next_page_offset")
            if offset is None:
                break
    return points


async def sync_collection(collection: str) -> int:
    points = await fetch_qdrant_points(collection)
    count = await asyncio.to_thread(registry.rebuild, collection, points)
    log.info("Sparse index synced collection=%s docs=%d", collection, count)
    return count


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

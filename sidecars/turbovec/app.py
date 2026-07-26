#!/usr/bin/env python3
"""TurboQuant dense ANN sidecar; dual-write from ingest, search for proxy."""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, List, Optional

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from core import DEFAULT_BIT_WIDTH, DEFAULT_DIM, HAS_TURBOVEC, TurboIndex

log = logging.getLogger("turbovec-sidecar")

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333").rstrip("/")
DEFAULT_COLLECTION = os.getenv("QDRANT_COLLECTION", "nomad_knowledge_base")
HOST = os.getenv("TURBOVEC_HOST", "0.0.0.0")
PORT = int(os.getenv("TURBOVEC_PORT", "8097"))
DIM = int(os.getenv("TURBOVEC_DIM", str(DEFAULT_DIM)))
BIT_WIDTH = int(os.getenv("TURBOVEC_BIT_WIDTH", str(DEFAULT_BIT_WIDTH)))
INDEX_PATH = Path(
    os.getenv("TURBOVEC_INDEX_PATH", "/var/lib/rag_proxy/turbovec/index.tvim")
)
SCROLL_BATCH = int(os.getenv("TURBOVEC_SCROLL_BATCH", "256"))
# When true, persist .tvim after each /add, /remove, /reindex, and on shutdown.
# Bulk ingest dual-writes many /add calls; set TURBOVEC_AUTO_SAVE=false and POST /save once.
AUTO_SAVE = os.getenv("TURBOVEC_AUTO_SAVE", "true").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)

index: Optional[TurboIndex] = None


def _require_index() -> TurboIndex:
    if index is None:
        raise HTTPException(status_code=503, detail="turbovec index not ready")
    return index


class AddRequest(BaseModel):
    ids: List[str]
    vectors: List[List[float]]


class SearchRequest(BaseModel):
    vector: List[float]
    limit: int = 20
    score_threshold: Optional[float] = None
    allowlist: Optional[List[str]] = None


class RemoveRequest(BaseModel):
    ids: List[str]


class ReindexRequest(BaseModel):
    collection: str = Field(default_factory=lambda: DEFAULT_COLLECTION)


async def _scroll_page(
    client: httpx.AsyncClient,
    collection: str,
    offset: str | int | None,
    limit: int,
) -> tuple[list[dict[str, Any]], str | int | None]:
    body: dict[str, Any] = {
        "limit": limit,
        "with_payload": False,
        "with_vector": True,
    }
    if offset is not None:
        body["offset"] = offset
    response = await client.post(
        f"{QDRANT_URL}/collections/{collection}/points/scroll",
        json=body,
    )
    response.raise_for_status()
    result = response.json().get("result", {})
    return result.get("points", []), result.get("next_page_offset")


def _point_vector(point: dict[str, Any]) -> list[float] | None:
    raw = point.get("vector")
    if isinstance(raw, list) and raw and isinstance(raw[0], (int, float)):
        return [float(x) for x in raw]
    if isinstance(raw, dict):
        # Named vectors — take first list value.
        for value in raw.values():
            if isinstance(value, list) and value and isinstance(value[0], (int, float)):
                return [float(x) for x in value]
    return None


async def sync_from_qdrant(collection: str) -> int:
    idx = _require_index()
    await asyncio.to_thread(idx.reset)
    added = 0
    offset: str | int | None = None
    async with httpx.AsyncClient(timeout=120.0) as client:
        while True:
            batch, offset = await _scroll_page(client, collection, offset, SCROLL_BATCH)
            if batch:
                ids: list[str] = []
                vectors: list[list[float]] = []
                for point in batch:
                    point_id = point.get("id")
                    vector = _point_vector(point)
                    if point_id is None or vector is None:
                        continue
                    if len(vector) != idx.dim:
                        continue
                    ids.append(str(point_id))
                    vectors.append(vector)
                if ids:
                    added += await asyncio.to_thread(idx.add, ids, vectors)
            if offset is None:
                break
    if AUTO_SAVE:
        await asyncio.to_thread(idx.save)
    log.info("turbovec reindex collection=%s vectors=%d", collection, added)
    return added


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global index
    if not HAS_TURBOVEC:
        raise RuntimeError("turbovec package is required; pip install turbovec")
    index = TurboIndex(dim=DIM, bit_width=BIT_WIDTH, index_path=INDEX_PATH)
    try:
        await asyncio.to_thread(index.load)
    except Exception as exc:
        log.warning("failed to load existing index (starting empty): %s", exc)
        index.reset()
    yield
    if index is not None:
        if AUTO_SAVE:
            try:
                await asyncio.to_thread(index.save)
            except Exception as exc:
                log.warning("shutdown save failed: %s", exc)
        index.close()
        index = None


app = FastAPI(
    title="RAG TurboVec Dense Sidecar",
    docs_url=None,
    redoc_url=None,
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, Any]:
    if index is None:
        return {"status": "starting"}
    stats = index.stats()
    return {"status": "ok", **stats}


@app.post("/add")
async def add(body: AddRequest) -> dict[str, Any]:
    idx = _require_index()
    try:
        count = await asyncio.to_thread(idx.add, body.ids, body.vectors)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if AUTO_SAVE:
        await asyncio.to_thread(idx.save)
    return {"added": count, "vectors": len(idx)}


@app.post("/search")
async def search(body: SearchRequest) -> dict[str, list[dict[str, Any]]]:
    idx = _require_index()
    limit = max(1, min(body.limit, 200))
    try:
        results = await asyncio.to_thread(
            idx.search,
            body.vector,
            limit=limit,
            score_threshold=body.score_threshold,
            allowlist=body.allowlist,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"results": results}


@app.post("/remove")
async def remove(body: RemoveRequest) -> dict[str, Any]:
    idx = _require_index()
    removed = await asyncio.to_thread(idx.remove, body.ids)
    if AUTO_SAVE and removed:
        await asyncio.to_thread(idx.save)
    return {"removed": removed, "vectors": len(idx)}


@app.post("/reindex")
async def reindex(body: ReindexRequest) -> dict[str, Any]:
    try:
        count = await sync_from_qdrant(body.collection)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Qdrant sync failed: {exc}") from exc
    return {"collection": body.collection, "docs": count, "vectors": count}


@app.post("/save")
async def save() -> dict[str, Any]:
    idx = _require_index()
    await asyncio.to_thread(idx.save)
    return {"saved": True, **idx.stats()}


def main() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    log.info(
        "TurboVec sidecar listening on %s:%s dim=%s bit_width=%s index=%s",
        HOST,
        PORT,
        DIM,
        BIT_WIDTH,
        INDEX_PATH,
    )
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")


if __name__ == "__main__":
    main()

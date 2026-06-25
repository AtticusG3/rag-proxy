"""Dashboard stats and health."""

from __future__ import annotations

import os
import shutil
from typing import Any

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from rag_admin.config import settings

router = APIRouter()
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "..", "templates"))


def _dir_size(path: str) -> int:
    total = 0
    if not os.path.isdir(path):
        return 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


async def fetch_stats(db: Any) -> dict[str, Any]:
    qdrant_points = 0
    sparse_docs = 0
    sparse_status = "unknown"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(
                f"{settings.qdrant_url.rstrip('/')}/collections/{settings.qdrant_collection}"
            )
            if r.status_code == 200:
                qdrant_points = int(r.json()["result"]["points_count"])
    except Exception:
        pass
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(f"{settings.sparse_index_url.rstrip('/')}/health")
            if r.status_code == 200:
                body = r.json()
                sparse_docs = int(body.get("docs", 0))
                sparse_status = str(body.get("status", "ok"))
    except Exception:
        sparse_status = "down"

    files = db.list_file_states()
    pending = sum(1 for f in files if f["status"] in ("pending", "queued", "running"))
    indexed = sum(1 for f in files if f["status"] == "indexed")
    zim_bytes = _dir_size(settings.zim_dir)
    upload_bytes = _dir_size(settings.upload_dir)
    disk = shutil.disk_usage(settings.zim_dir if os.path.isdir(settings.zim_dir) else "/")

    return {
        "qdrant_points": qdrant_points,
        "sparse_docs": sparse_docs,
        "sparse_status": sparse_status,
        "collection": settings.qdrant_collection,
        "pending_files": pending,
        "indexed_files": indexed,
        "zim_bytes": zim_bytes,
        "upload_bytes": upload_bytes,
        "disk_free_gb": round(disk.free / (1024**3), 1),
        "disk_used_pct": round(100 * disk.used / disk.total, 1) if disk.total else 0,
    }


@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request) -> HTMLResponse:
    db = request.app.state.db
    stats = await fetch_stats(db)
    jobs = db.list_jobs(limit=10)
    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {"stats": stats, "jobs": jobs},
    )


@router.get("/jobs", response_class=HTMLResponse)
async def jobs_page(request: Request) -> HTMLResponse:
    db = request.app.state.db
    files = db.list_file_states()
    jobs = db.list_jobs(limit=100)
    return templates.TemplateResponse(
        request,
        "jobs.html",
        {"files": files, "jobs": jobs},
    )

"""Ingest API routes."""

from __future__ import annotations

from urllib.parse import urlencode

from fastapi import APIRouter, Form, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ingest.db import VALID_PRIORITIES

from rag_admin.config import settings
from rag_admin.helpers import flash_redirect, validated_ingest_file_path
from rag_admin.ingest_status import (
    enrich_file_rows,
    ingest_config_snapshot,
    ingest_queue_stats,
    resolve_sort,
    sort_file_rows,
)

router = APIRouter(prefix="/api/ingest")


class SyncResponse(BaseModel):
    job_id: str
    status: str


@router.post("/sync", response_model=SyncResponse)
async def sync_storage(request: Request) -> SyncResponse:
    worker = request.app.state.worker
    job_id = worker.enqueue_sync()
    return SyncResponse(job_id=job_id, status="queued")


@router.post("/file")
async def ingest_file(request: Request, file_path: str) -> JSONResponse:
    file_path = validated_ingest_file_path(file_path)
    worker = request.app.state.worker
    job_id = worker.enqueue_file(file_path)
    return JSONResponse({"job_id": job_id, "status": "queued"})


@router.get("/status")
async def ingest_status(request: Request) -> JSONResponse:
    db = request.app.state.db
    sort, sort_dir = resolve_sort(
        request.query_params.get("sort"), request.query_params.get("dir")
    )
    files = enrich_file_rows(
        db.ingest.list_file_states(order="updated_desc"),
        stall_seconds=settings.stall_seconds,
    )
    stats = ingest_queue_stats(files)
    files = sort_file_rows(files, sort=sort, direction=sort_dir)
    return JSONResponse(
        {
            "files": files,
            "jobs": db.ingest.list_jobs(limit=20),
            "stats": stats,
            "config": ingest_config_snapshot(request.app.state.worker),
        }
    )


@router.post("/retry")
async def retry_file(request: Request, file_path: str) -> JSONResponse:
    file_path = validated_ingest_file_path(file_path)
    worker = request.app.state.worker
    job_id = worker.retry_file(file_path)
    return JSONResponse({"job_id": job_id, "status": "queued"})


@router.post("/retry-form")
async def retry_file_form(
    request: Request,
    file_path: str = Form(...),
):
    file_path = validated_ingest_file_path(file_path)
    worker = request.app.state.worker
    worker.retry_file(file_path)
    name = file_path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    return flash_redirect(f"/jobs", f"Retry queued for {name}.")


@router.post("/priority-form")
async def set_priority_form(
    request: Request,
    file_path: str = Form(...),
    priority: str = Form(...),
    sort: str = Form(default=""),
    dir: str = Form(default=""),
):
    if priority not in VALID_PRIORITIES:
        return flash_redirect("/jobs", f"Invalid priority: {priority}", level="error")
    file_path = validated_ingest_file_path(file_path)
    db = request.app.state.db
    updated = db.ingest.set_file_priority(file_path, priority)
    name = file_path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    target = "/jobs"
    if sort or dir:
        query = urlencode({"sort": sort, "dir": dir})
        target = f"/jobs?{query}"
    if not updated:
        return flash_redirect(target, f"{name} is not in the ingest queue.", level="error")
    return flash_redirect(target, f"Priority for {name} set to {priority}.")


@router.post("/retry-failed-form")
async def retry_failed_form(request: Request):
    worker = request.app.state.worker
    worker.retry_all_failed()
    return flash_redirect("/jobs", "All failed files re-queued for ingest.")


@router.post("/restart-stalled-form")
async def restart_stalled_form(request: Request):
    worker = request.app.state.worker
    worker.restart_stalled_files()
    return flash_redirect("/jobs", "Stalled files restarted.")


@router.post("/sync-form")
async def sync_form(request: Request):
    worker = request.app.state.worker
    worker.enqueue_sync()
    return flash_redirect("/jobs", "Storage scan complete. New and failed files were queued.")


@router.post("/dismiss-form")
async def dismiss_file_form(
    request: Request,
    file_path: str = Form(...),
):
    file_path = validated_ingest_file_path(file_path)
    worker = request.app.state.worker
    worker.remove_file_from_index(file_path)
    name = file_path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    return flash_redirect("/jobs", f"Removed {name} from ingest queue and index.")


@router.post("/dismiss-missing-form")
async def dismiss_missing_form(request: Request):
    worker = request.app.state.worker
    removed = worker.dismiss_all_missing_files()
    if not removed:
        return flash_redirect("/jobs", "No missing files in ingest queue.")
    return flash_redirect(
        "/jobs",
        f"Removed {len(removed)} missing file(s) from ingest queue.",
    )

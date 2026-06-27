"""Ingest API routes."""

from __future__ import annotations

from fastapi import APIRouter, Form, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from rag_admin.config import settings
from rag_admin.flash import flash_redirect
from rag_admin.ingest_status import (
    enrich_file_rows,
    ingest_config_snapshot,
    ingest_queue_stats,
)
from rag_admin.paths import validated_ingest_file_path

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
    files = enrich_file_rows(
        db.ingest.list_file_states(order="updated_desc"),
        stall_seconds=settings.stall_seconds,
    )
    return JSONResponse(
        {
            "files": files,
            "jobs": db.ingest.list_jobs(limit=20),
            "stats": ingest_queue_stats(files),
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

"""Ingest API routes."""

from __future__ import annotations

from fastapi import APIRouter, Form, Request
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel

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
    worker = request.app.state.worker
    job_id = worker.enqueue_file(file_path)
    return JSONResponse({"job_id": job_id, "status": "queued"})


@router.get("/status")
async def ingest_status(request: Request) -> JSONResponse:
    db = request.app.state.db
    return JSONResponse(
        {
            "files": db.list_file_states(),
            "jobs": db.list_jobs(limit=20),
        }
    )


@router.post("/retry")
async def retry_file(request: Request, file_path: str) -> JSONResponse:
    worker = request.app.state.worker
    job_id = worker.retry_file(file_path)
    return JSONResponse({"job_id": job_id, "status": "queued"})


@router.post("/retry-form")
async def retry_file_form(
    request: Request,
    file_path: str = Form(...),
) -> RedirectResponse:
    worker = request.app.state.worker
    worker.retry_file(file_path)
    return RedirectResponse(url="/jobs", status_code=303)


@router.post("/retry-failed-form")
async def retry_failed_form(request: Request) -> RedirectResponse:
    worker = request.app.state.worker
    worker.retry_all_failed()
    return RedirectResponse(url="/jobs", status_code=303)


@router.post("/restart-stalled-form")
async def restart_stalled_form(request: Request) -> RedirectResponse:
    worker = request.app.state.worker
    worker.restart_stalled_files()
    return RedirectResponse(url="/jobs", status_code=303)


@router.post("/sync-form")
async def sync_form(request: Request) -> RedirectResponse:
    worker = request.app.state.worker
    worker.enqueue_sync()
    return RedirectResponse(url="/jobs", status_code=303)

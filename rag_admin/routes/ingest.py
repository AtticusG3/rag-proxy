"""Ingest API routes."""

from __future__ import annotations

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel

from rag_admin.config import resolve_ingest_path, settings

router = APIRouter(prefix="/api/ingest")


def _validated_file_path(file_path: str) -> str:
    try:
        return str(
            resolve_ingest_path(
                file_path,
                zim_dir=settings.zim_dir,
                upload_dir=settings.upload_dir,
            )
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


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
    file_path = _validated_file_path(file_path)
    worker = request.app.state.worker
    job_id = worker.enqueue_file(file_path)
    return JSONResponse({"job_id": job_id, "status": "queued"})


@router.get("/status")
async def ingest_status(request: Request) -> JSONResponse:
    db = request.app.state.db
    return JSONResponse(
        {
            "files": db.ingest.list_file_states(),
            "jobs": db.ingest.list_jobs(limit=20),
        }
    )


@router.post("/retry")
async def retry_file(request: Request, file_path: str) -> JSONResponse:
    file_path = _validated_file_path(file_path)
    worker = request.app.state.worker
    job_id = worker.retry_file(file_path)
    return JSONResponse({"job_id": job_id, "status": "queued"})


@router.post("/retry-form")
async def retry_file_form(
    request: Request,
    file_path: str = Form(...),
) -> RedirectResponse:
    file_path = _validated_file_path(file_path)
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

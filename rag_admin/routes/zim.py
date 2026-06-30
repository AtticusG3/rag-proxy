"""ZIM content manager routes."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse

from ingest.types import determine_file_type
from rag_admin.config import settings
from rag_admin.helpers import flash_redirect, validated_ingest_file_path
from rag_admin.templates_env import templates

router = APIRouter()


def _ensure_dirs() -> None:
    os.makedirs(settings.zim_dir, exist_ok=True)
    os.makedirs(settings.upload_dir, exist_ok=True)


def _list_zim_files() -> list[dict]:
    _ensure_dirs()
    out: list[dict] = []
    for directory in (settings.zim_dir, settings.upload_dir):
        if not os.path.isdir(directory):
            continue
        for name in sorted(os.listdir(directory)):
            path = os.path.join(directory, name)
            if not os.path.isfile(path):
                continue
            ftype = determine_file_type(path)
            if ftype == "unknown":
                continue
            out.append(
                {
                    "path": path,
                    "name": name,
                    "type": ftype,
                    "size": os.path.getsize(path),
                    "dir": directory,
                }
            )
    return out


@router.get("/zim", response_class=HTMLResponse)
async def zim_list(request: Request) -> HTMLResponse:
    db = request.app.state.db
    files = _list_zim_files()
    states = {row["file_path"]: row for row in db.ingest.list_file_states()}
    for item in files:
        path = item["path"]
        if path not in states:
            db.ingest.upsert_file_state(path, status="pending", file_type=item["type"])
    states = {row["file_path"]: row for row in db.ingest.list_file_states()}
    for item in files:
        item["state"] = states.get(item["path"], {})
    return templates.TemplateResponse(request, "zim.html", {"files": files})


@router.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "upload.html",
        {
            "zim_dir": settings.zim_dir,
            "upload_dir": settings.upload_dir,
        },
    )


@router.post("/upload")
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    target: str = Form("zim"),
):
    _ensure_dirs()
    dest_dir = settings.zim_dir if target == "zim" else settings.upload_dir
    filename = Path(file.filename or "upload.bin").name
    dest = os.path.join(dest_dir, filename)
    with open(dest, "wb") as handle:
        shutil.copyfileobj(file.file, handle)
    db = request.app.state.db
    db.ingest.upsert_file_state(
        dest,
        status="pending",
        file_type=determine_file_type(dest),
    )
    return flash_redirect("/zim", f"Uploaded {filename} and queued for ingest.")


@router.post("/zim/delete")
async def delete_zim(request: Request, file_path: str = Form(...)) -> RedirectResponse:
    file_path = validated_ingest_file_path(file_path)
    worker = request.app.state.worker
    worker.remove_file_from_index(file_path)
    return RedirectResponse(url="/zim", status_code=303)

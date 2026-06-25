"""ZIM content manager routes."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from ingest.types import determine_file_type
from rag_admin.config import settings

router = APIRouter()
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "..", "templates"))


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
    states = {row["file_path"]: row for row in db.list_file_states()}
    for item in files:
        path = item["path"]
        if path not in states:
            db.upsert_file_state(path, status="pending", file_type=item["type"])
    states = {row["file_path"]: row for row in db.list_file_states()}
    for item in files:
        item["state"] = states.get(item["path"], {})
    return templates.TemplateResponse(request, "zim.html", {"files": files})


@router.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "upload.html", {})


@router.post("/upload")
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    target: str = Form("zim"),
) -> RedirectResponse:
    _ensure_dirs()
    dest_dir = settings.zim_dir if target == "zim" else settings.upload_dir
    filename = Path(file.filename or "upload.bin").name
    dest = os.path.join(dest_dir, filename)
    with open(dest, "wb") as handle:
        shutil.copyfileobj(file.file, handle)
    db = request.app.state.db
    db.upsert_file_state(
        dest,
        status="pending",
        file_type=determine_file_type(dest),
    )
    return RedirectResponse(url="/zim", status_code=303)


@router.post("/zim/delete")
async def delete_zim(request: Request, file_path: str = Form(...)) -> RedirectResponse:
    worker = request.app.state.worker
    worker.remove_file_from_index(file_path)
    return RedirectResponse(url="/zim", status_code=303)

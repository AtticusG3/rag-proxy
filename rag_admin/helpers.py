"""Small shared helpers for rag-admin routes and templates."""

from __future__ import annotations

from datetime import datetime
from urllib.parse import urlencode

from fastapi import HTTPException
from fastapi.responses import RedirectResponse

from rag_admin.config import resolve_ingest_path, settings


def flash_redirect(url: str, message: str, *, level: str = "info") -> RedirectResponse:
    query = urlencode({"flash": message, "flash_level": level})
    separator = "&" if "?" in url else "?"
    return RedirectResponse(url=f"{url}{separator}{query}", status_code=303)


def validated_ingest_file_path(file_path: str) -> str:
    """Resolve file_path under zim_dir or upload_dir; HTTP 400 if invalid."""
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


def format_datetime(value: str | None) -> str:
    """Format ISO-8601 timestamps as DD-MMM-YYYY hh:mm:ss."""
    if not value:
        return ""
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value
    return dt.strftime("%d-%b-%Y %H:%M:%S")

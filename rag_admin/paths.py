"""Shared path validation for rag-admin routes."""

from __future__ import annotations

from fastapi import HTTPException

from rag_admin.config import resolve_ingest_path, settings


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

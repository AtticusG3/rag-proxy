"""RAG admin web UI for ZIM ingest and knowledge base management."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

import uvicorn
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from ingest.worker import IngestConfig, IngestWorker
from rag_admin.auth import (
    AuthMiddleware,
    clear_session,
    set_session,
    verify_password,
)
from rag_admin.config import settings, validate_settings
from rag_admin.db import AdminDatabase
from rag_admin.catalog import CatalogDownloadManager
from rag_admin.routes import dashboard, explorer, ingest, zim
from rag_admin.templates_env import templates

log = logging.getLogger("rag-admin")

_BASE = os.path.dirname(__file__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    validate_settings(settings)
    os.makedirs(settings.zim_dir, exist_ok=True)
    os.makedirs(settings.upload_dir, exist_ok=True)
    os.makedirs(os.path.dirname(settings.db_path), exist_ok=True)

    db = AdminDatabase(settings.db_path)
    config = IngestConfig(
        zim_dir=settings.zim_dir,
        upload_dir=settings.upload_dir,
        embed_url=settings.embed_url,
        qdrant_url=settings.qdrant_url,
        qdrant_collection=settings.qdrant_collection,
        sparse_index_url=settings.sparse_index_url,
        batch_size=settings.batch_size,
        max_articles=settings.max_articles,
        embed_max_chars=settings.embed_max_chars,
        sparse_reindex_mode=settings.sparse_reindex_mode,
        stall_seconds=settings.stall_seconds,
    )
    worker = IngestWorker(config, db.ingest)
    worker.start()
    catalog_manager = CatalogDownloadManager(
        db, settings.zim_dir, settings.upload_dir, worker
    )
    catalog_manager.start()

    app.state.db = db
    app.state.worker = worker
    app.state.catalog_manager = catalog_manager
    log.info("rag-admin started zim_dir=%s port=%s", settings.zim_dir, settings.port)
    yield
    catalog_manager.stop()
    worker.stop()


def create_app() -> FastAPI:
    app = FastAPI(title="RAG Admin", docs_url=None, redoc_url=None, lifespan=lifespan)
    app.add_middleware(AuthMiddleware)
    static_dir = os.path.join(_BASE, "static")
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    app.include_router(dashboard.router)
    app.include_router(explorer.router)
    app.include_router(zim.router)
    app.include_router(ingest.router)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/login", response_class=HTMLResponse)
    async def login_page(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(request, "login.html", {"error": None})

    @app.post("/login", response_model=None)
    async def login_submit(
        request: Request,
        password: str = Form(...),
    ):
        if not verify_password(password):
            return templates.TemplateResponse(
                request,
                "login.html",
                {"error": "Invalid password"},
                status_code=401,
            )
        response = RedirectResponse(url="/", status_code=303)
        set_session(response)
        return response

    @app.post("/logout")
    async def logout() -> RedirectResponse:
        response = RedirectResponse(url="/login", status_code=303)
        clear_session(response)
        return response

    return app


app = create_app()


def main() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    uvicorn.run(
        "rag_admin.app:app",
        host=settings.host,
        port=settings.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()

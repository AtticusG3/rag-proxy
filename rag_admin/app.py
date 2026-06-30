"""RAG admin web UI for ZIM ingest and knowledge base management."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

import uvicorn
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from ingest.worker import IngestWorker
from ingest.chunking import warmup_chunking
from rag_admin.auth import (
    AuthMiddleware,
    clear_session,
    is_authenticated,
    set_session,
    verify_password,
)
from rag_admin.config import settings, validate_settings
from rag_admin.db import AdminDatabase
from rag_admin.rate_limit import LoginRateLimiter
from rag_admin.catalog import CatalogDownloadManager
from rag_admin.job_runner import BackgroundJobRunner
from rag_admin.routes import dashboard, explorer, ingest, settings as settings_routes, zim
from rag_admin.settings_store import SettingsStore
from rag_admin.helpers import client_ip, templates

log = logging.getLogger("rag-admin")

_BASE = os.path.dirname(__file__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    validate_settings(settings)
    os.makedirs(settings.zim_dir, exist_ok=True)
    os.makedirs(settings.upload_dir, exist_ok=True)
    os.makedirs(os.path.dirname(settings.db_path), exist_ok=True)

    db = AdminDatabase(settings.db_path)
    settings_store = SettingsStore(
        db,
        admin_env_path=settings.admin_env_path,
        proxy_env_path=settings.proxy_env_path,
        pool_scale_env_path=settings.pool_scale_env_path,
        pool_env_path=settings.pool_env_path,
    )
    config = settings_store.build_ingest_config(
        zim_dir=settings.zim_dir,
        upload_dir=settings.upload_dir,
    )
    worker = IngestWorker(config, db.ingest)
    settings_store.apply_to_worker(
        worker,
        zim_dir=settings.zim_dir,
        upload_dir=settings.upload_dir,
    )
    worker.start()
    warmup_chunking()
    job_runner = BackgroundJobRunner(
        db,
        repo_root=settings.repo_root,
        log_dir=settings.job_log_dir,
    )
    catalog_manager = CatalogDownloadManager(
        db, settings.zim_dir, settings.upload_dir, worker
    )
    catalog_manager.start()

    app.state.db = db
    app.state.worker = worker
    app.state.settings_store = settings_store
    app.state.job_runner = job_runner
    app.state.catalog_manager = catalog_manager
    app.state.login_rate_limiter = LoginRateLimiter(
        max_attempts=settings.login_max_attempts,
        lockout_minutes=settings.login_lockout_minutes,
    )
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
    app.include_router(settings_routes.router)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> Response:
        return Response(status_code=204)

    @app.get("/login", response_class=HTMLResponse)
    async def login_page(request: Request) -> HTMLResponse:
        if is_authenticated(request, request.app.state.db):
            return RedirectResponse(url="/", status_code=303)
        return templates.TemplateResponse(request, "login.html", {"error": None})

    @app.post("/login", response_model=None)
    async def login_submit(
        request: Request,
        password: str = Form(...),
    ):
        db: AdminDatabase = request.app.state.db
        limiter: LoginRateLimiter = request.app.state.login_rate_limiter
        db.prune_expired_admin_sessions()
        client_addr = client_ip(request)
        if limiter.is_locked(client_addr):
            return templates.TemplateResponse(
                request,
                "login.html",
                {"error": "Too many failed login attempts. Try again later."},
                status_code=429,
            )
        if not verify_password(password):
            limiter.record_failure(client_addr)
            return templates.TemplateResponse(
                request,
                "login.html",
                {"error": "Invalid password"},
                status_code=401,
            )
        limiter.clear(client_addr)
        response = RedirectResponse(url="/", status_code=303)
        set_session(response, db, client_ip=client_addr)
        return response

    @app.post("/logout")
    async def logout(request: Request) -> RedirectResponse:
        response = RedirectResponse(url="/login", status_code=303)
        clear_session(response, request, request.app.state.db)
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

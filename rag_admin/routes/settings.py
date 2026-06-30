"""Admin settings: persistent env overrides and job controls."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse

from ingest.worker import trigger_sparse_reindex
from rag_admin.job_runner import JOB_EMBED_POOL_SCALE, JOB_MEMGRAPH_BUILD
from rag_admin.config import settings
from rag_admin.helpers import flash_redirect
from rag_admin.service_restart import schedule_restart
from rag_admin.service_status import service_status
from rag_admin.settings_guides import GROUP_TUNING, field_placeholder
from rag_admin.settings_schema import GROUP_LABELS, SETTING_FIELDS, SETTING_GROUPS
from rag_admin.settings_store import SettingsStore
from rag_admin.helpers import templates

router = APIRouter()
log = logging.getLogger("rag-admin")


def _store(request: Request) -> SettingsStore:
    return request.app.state.settings_store


def _fields_for_group(store: SettingsStore, group: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for field in SETTING_FIELDS:
        if field.group != group:
            continue
        effective = store.get_value(field.key, field.default)
        rows.append(
            {
                "key": field.key,
                "label": field.label,
                "field_type": field.field_type,
                "options": field.options,
                "help_text": field.help_text,
                "default": field.default,
                "placeholder": field_placeholder(field),
                "display_value": store.get_override_value(field.key, target=field.target) or "",
                "effective_value": effective,
                "has_override": store.has_override(field),
                "hot": field.hot,
            }
        )
    return rows


@router.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request, tab: str = "ingest") -> HTMLResponse:
    store = _store(request)
    worker = request.app.state.worker
    job_runner = request.app.state.job_runner
    active_tab = tab if tab in SETTING_GROUPS else "ingest"
    values = store.get_group_values(active_tab)
    services = await service_status(
        qdrant_url=store.get_value("QDRANT_URL", settings.qdrant_url),
        collection=store.get_value("QDRANT_COLLECTION", settings.qdrant_collection),
        sparse_index_url=store.get_value("SPARSE_INDEX_URL", settings.sparse_index_url),
        embed_url=store.get_value("EMBED_URL", settings.embed_url),
        rag_proxy_url=store.get_value("RAG_PROXY_URL", settings.rag_proxy_url),
        reranker_url=store.get_value("RERANKER_URL", ""),
        memgraph_db_path=store.get_value(
            "MEMGRAPHRAG_DB_PATH",
            "/var/lib/rag_proxy/memgraphrag.sqlite",
        ),
    )
    build_job = job_runner.active_job(JOB_MEMGRAPH_BUILD)
    pool_scale_job = job_runner.active_job(JOB_EMBED_POOL_SCALE)
    build_history = request.app.state.db.list_background_jobs(JOB_MEMGRAPH_BUILD, limit=5)
    pool_scale_history = request.app.state.db.list_background_jobs(JOB_EMBED_POOL_SCALE, limit=5)
    log_tail = ""
    if build_job:
        log_tail = job_runner.tail_log(build_job["id"])
    pool_scale_log_tail = ""
    if pool_scale_job:
        pool_scale_log_tail = job_runner.tail_log(pool_scale_job["id"])

    return templates.TemplateResponse(
        request,
        "settings.html",
        {
            "tabs": SETTING_GROUPS,
            "tab_labels": GROUP_LABELS,
            "active_tab": active_tab,
            "fields": _fields_for_group(store, active_tab),
            "values": values,
            "group_tuning": GROUP_TUNING.get(active_tab, ()),
            "services": services,
            "ingest_paused": worker.paused,
            "ingest_config": worker.config,
            "build_job": build_job,
            "build_history": build_history,
            "log_tail": log_tail,
            "pool_scale_job": pool_scale_job,
            "pool_scale_history": pool_scale_history,
            "pool_scale_log_tail": pool_scale_log_tail,
            "admin_env_path": store.admin_env_path,
            "proxy_env_path": store.proxy_env_path,
            "pool_env": store.pool_env_snapshot(),
            "can_restart_proxy": bool(settings.proxy_restart_cmd.strip()),
            "can_restart_admin": bool(settings.admin_restart_cmd.strip()),
        },
    )


@router.post("/settings/save/{group}")
async def settings_save(request: Request, group: str):
    if group not in SETTING_GROUPS:
        return flash_redirect("/settings", "Unknown settings group.", level="error")
    form = await request.form()
    store = _store(request)
    try:
        result = store.save_group(group, {k: str(v) for k, v in form.items() if k != "tab"})
    except (ValueError, TypeError) as exc:
        return flash_redirect(f"/settings?tab={group}", str(exc), level="error")

    worker = request.app.state.worker
    if group == "ingest":
        store.apply_to_worker(
            worker,
            zim_dir=settings.zim_dir,
            upload_dir=settings.upload_dir,
        )

    message = f"Saved {len(result.updated)} setting(s)."
    if result.pool_scale_updated:
        message += " Use Scale embed pool on the ingest tab, or save and click Scale embed pool."
    if result.restart_proxy:
        message += " Restart rag-proxy to apply proxy env changes."
    if result.restart_admin:
        message += " Restart rag-admin to apply admin env changes."
    return flash_redirect(f"/settings?tab={group}", message)


@router.post("/settings/ingest/pause")
async def ingest_pause(request: Request):
    store = _store(request)
    store.set_ingest_paused(True)
    request.app.state.worker.set_paused(True)
    return flash_redirect("/settings?tab=ingest", "Dense ingest paused.")


@router.post("/settings/ingest/resume")
async def ingest_resume(request: Request):
    store = _store(request)
    store.set_ingest_paused(False)
    request.app.state.worker.set_paused(False)
    return flash_redirect("/settings?tab=ingest", "Dense ingest resumed.")


@router.post("/settings/sparse/reindex")
async def sparse_reindex_now(request: Request):
    worker = request.app.state.worker
    docs = trigger_sparse_reindex(worker.config)
    if docs is None:
        return flash_redirect(
            "/settings?tab=ingest",
            "BM25 reindex failed or sparse sidecar not configured.",
            level="error",
        )
    return flash_redirect("/settings?tab=ingest", f"BM25 reindex complete ({docs} docs).")


@router.post("/settings/embed-pool/scale")
async def embed_pool_scale_start(request: Request):
    store = _store(request)
    job_runner = request.app.state.job_runner
    worker = request.app.state.worker

    def on_success() -> None:
        synced = store.sync_pool_ingest_from_pool_env()
        store.apply_to_worker(
            worker,
            zim_dir=settings.zim_dir,
            upload_dir=settings.upload_dir,
        )
        if synced:
            log.info("embed pool scale synced ingest keys: %s", ", ".join(synced))

    try:
        job_id = job_runner.start_embed_pool_scale(
            store.embed_pool_scale_params(),
            on_success=on_success,
        )
    except RuntimeError as exc:
        return flash_redirect("/settings?tab=ingest", str(exc), level="error")
    return flash_redirect(
        "/settings?tab=ingest",
        f"Embed pool scale started (job {job_id[:8]}). This may restart nomic-embed units.",
    )


@router.post("/settings/embed-pool/stop")
async def embed_pool_scale_stop(request: Request):
    stopped = request.app.state.job_runner.stop_active(JOB_EMBED_POOL_SCALE)
    if not stopped:
        return flash_redirect("/settings?tab=ingest", "No running pool scale job to stop.", level="error")
    return flash_redirect("/settings?tab=ingest", "Embed pool scale stopped.")


@router.post("/settings/memgraph/build")
async def memgraph_build_start(request: Request):
    store = _store(request)
    job_runner = request.app.state.job_runner
    params = store.memgraph_build_params()
    if not params.get("llm_model"):
        return flash_redirect(
            "/settings?tab=memgraph_build",
            "Set Build LLM model before starting.",
            level="error",
        )
    try:
        job_id = job_runner.start_memgraph_build(params)
    except RuntimeError as exc:
        return flash_redirect("/settings?tab=memgraph_build", str(exc), level="error")
    return flash_redirect(
        "/settings?tab=memgraph_build",
        f"MemGraphRAG build started (job {job_id[:8]}).",
    )


@router.post("/settings/memgraph/stop")
async def memgraph_build_stop(request: Request):
    stopped = request.app.state.job_runner.stop_active(JOB_MEMGRAPH_BUILD)
    if not stopped:
        return flash_redirect("/settings?tab=memgraph_build", "No running build to stop.", level="error")
    return flash_redirect("/settings?tab=memgraph_build", "MemGraphRAG build stopped.")


@router.get("/api/settings/status")
async def settings_status_api(request: Request) -> JSONResponse:
    store = _store(request)
    worker = request.app.state.worker
    job_runner = request.app.state.job_runner
    build_job = job_runner.active_job(JOB_MEMGRAPH_BUILD)
    pool_scale_job = job_runner.active_job(JOB_EMBED_POOL_SCALE)
    payload: dict[str, Any] = {
        "ingest_paused": worker.paused,
        "ingest_config": {
            "batch_size": worker.config.batch_size,
            "embed_concurrency": worker.config.embed_concurrency,
            "file_concurrency": worker.config.file_concurrency,
            "sparse_reindex_mode": worker.config.sparse_reindex_mode,
        },
        "build_job": build_job,
        "pool_scale_job": pool_scale_job,
        "log_tail": job_runner.tail_log(build_job["id"]) if build_job else "",
        "pool_scale_log_tail": job_runner.tail_log(pool_scale_job["id"]) if pool_scale_job else "",
        "pool_env": store.pool_env_snapshot(),
    }
    services = await service_status(
        qdrant_url=store.get_value("QDRANT_URL", settings.qdrant_url),
        collection=store.get_value("QDRANT_COLLECTION", settings.qdrant_collection),
        sparse_index_url=store.get_value("SPARSE_INDEX_URL", settings.sparse_index_url),
        embed_url=store.get_value("EMBED_URL", settings.embed_url),
        rag_proxy_url=store.get_value("RAG_PROXY_URL", settings.rag_proxy_url),
        reranker_url=store.get_value("RERANKER_URL", ""),
        memgraph_db_path=store.get_value(
            "MEMGRAPHRAG_DB_PATH",
            "/var/lib/rag_proxy/memgraphrag.sqlite",
        ),
    )
    payload["services"] = services
    return JSONResponse(payload)


@router.post("/settings/restart/proxy")
async def restart_proxy_service(request: Request):
    ok, msg = schedule_restart(settings.proxy_restart_cmd)
    if not ok:
        return flash_redirect("/settings", msg, level="error")
    return flash_redirect("/settings", f"rag-proxy restart scheduled. {msg}")


@router.post("/settings/restart/admin")
async def restart_admin_service(request: Request):
    ok, msg = schedule_restart(settings.admin_restart_cmd)
    if not ok:
        return flash_redirect("/settings", msg, level="error")
    return flash_redirect(
        "/settings",
        f"rag-admin restart scheduled; refresh this page in a few seconds. {msg}",
    )

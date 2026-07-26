"""Admin settings: persistent env overrides and job controls."""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse

from ingest.worker import trigger_sparse_reindex
from ingest.embed_lifecycle import ensure_embed_urls, on_demand_enabled
from ingest.embed_urls import parse_ingest_embed_urls
from rag_admin.job_runner import (
    JOB_EMBED_POOL_SCALE,
    JOB_MEMGRAPH_BUILD,
    JOB_SIDECAR_MIGRATE,
)
from rag_admin.config import settings
from rag_admin.helpers import flash_redirect
from rag_admin.service_restart import schedule_restart
from rag_admin.service_status import service_status
from rag_admin.settings_guides import GROUP_TUNING, field_placeholder
from rag_admin.settings_schema import GROUP_LABELS, SETTING_FIELDS, SETTING_GROUPS
from rag_admin.settings_store import SettingsStore
from rag_admin.settings_ui import (
    ADVANCED_KEYS,
    COGNITIVE_STAGE_MAP,
    FIELD_SECTION,
    GROUP_NAV_HINTS,
    GROUP_NAV_LABELS,
    GROUP_SECTIONS,
    SETTING_REQUIRES,
    SETTING_REQUIRES_NONEMPTY,
    option_labels_for,
    settings_control_plane,
    settings_query,
    workflow_steps_for,
)
from rag_admin.helpers import templates

router = APIRouter()
log = logging.getLogger("rag-admin")


def _store(request: Request) -> SettingsStore:
    return request.app.state.settings_store


def _parse_mode(raw: str | None) -> str:
    mode = (raw or "basic").strip().lower()
    return mode if mode in ("basic", "advanced") else "basic"


def _request_mode(request: Request) -> str:
    return _parse_mode(request.query_params.get("mode"))


def _settings_flash(
    tab: str,
    message: str,
    *,
    mode: str = "basic",
    level: str = "info",
):
    return flash_redirect(settings_query(tab, mode), message, level=level)


def _fields_for_group(
    store: SettingsStore, group: str, *, mode: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for field in SETTING_FIELDS:
        if field.group != group:
            continue
        advanced = field.key in ADVANCED_KEYS
        if mode == "basic" and advanced:
            continue
        effective = store.get_value(field.key, field.default)
        bool_on = effective.lower() in ("true", "1", "yes", "on")
        rows.append(
            {
                "key": field.key,
                "label": field.label,
                "field_type": field.field_type,
                "options": field.options,
                "option_choices": option_labels_for(field.key, field.options),
                "help_text": field.help_text,
                "default": field.default,
                "placeholder": field_placeholder(field),
                "display_value": store.get_override_value(field.key, target=field.target) or "",
                "effective_value": effective,
                "bool_on": bool_on,
                "has_override": store.has_override(field),
                "hot": field.hot,
                "advanced": advanced,
                "section": FIELD_SECTION.get(field.key, "general"),
                "requires": SETTING_REQUIRES.get(field.key, ()),
                "requires_nonempty": SETTING_REQUIRES_NONEMPTY.get(field.key, ()),
            }
        )
    return rows


def _field_sections(group: str, fields: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_section: dict[str, list[dict[str, Any]]] = {}
    for field in fields:
        by_section.setdefault(field["section"], []).append(field)
    sections: list[dict[str, Any]] = []
    for section_id, label in GROUP_SECTIONS.get(group, (("general", "Settings"),)):
        section_fields = by_section.get(section_id)
        if not section_fields:
            continue
        sections.append(
            {
                "id": section_id,
                "label": label,
                "fields": section_fields,
            }
        )
    # Any keys missing from FIELD_SECTION land in a leftover bucket.
    known = {sid for sid, _ in GROUP_SECTIONS.get(group, ())}
    leftovers = [
        field
        for sid, group_fields in by_section.items()
        if sid not in known
        for field in group_fields
    ]
    if leftovers:
        sections.append({"id": "other", "label": "Other", "fields": leftovers})
    return sections


@router.get("/settings", response_class=HTMLResponse)
async def settings_page(
    request: Request, tab: str = "ingest", mode: str = "basic"
) -> HTMLResponse:
    store = _store(request)
    worker = request.app.state.worker
    job_runner = request.app.state.job_runner
    active_tab = tab if tab in SETTING_GROUPS else "ingest"
    ui_mode = _parse_mode(mode)
    values = store.get_group_values(active_tab)
    fields = _fields_for_group(store, active_tab, mode=ui_mode)
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
    migrate_job = job_runner.active_job(JOB_SIDECAR_MIGRATE)
    pool_scale_starting = job_runner.scale_starting()
    build_history = request.app.state.db.list_background_jobs(JOB_MEMGRAPH_BUILD, limit=5)
    pool_scale_history = request.app.state.db.list_background_jobs(JOB_EMBED_POOL_SCALE, limit=5)
    migrate_history = request.app.state.db.list_background_jobs(JOB_SIDECAR_MIGRATE, limit=5)
    log_tail = ""
    if build_job:
        log_tail = job_runner.tail_log(build_job["id"])
    pool_scale_log_tail = ""
    if pool_scale_job:
        pool_scale_log_tail = job_runner.tail_log(pool_scale_job["id"])
    migrate_log_tail = ""
    if migrate_job:
        migrate_log_tail = job_runner.tail_log(migrate_job["id"])

    pool_env = store.pool_env_snapshot()
    # Cross-tab effective values for prerequisite checks (switches may depend on other tabs).
    all_values = {
        field.key: store.get_value(field.key, field.default) for field in SETTING_FIELDS
    }
    control_plane = settings_control_plane(all_values)
    return templates.TemplateResponse(
        request,
        "settings.html",
        {
            "tabs": SETTING_GROUPS,
            "tab_labels": GROUP_LABELS,
            "nav_labels": GROUP_NAV_LABELS,
            "nav_hints": GROUP_NAV_HINTS,
            "active_tab": active_tab,
            "mode": ui_mode,
            "settings_href": settings_query,
            "fields": fields,
            "field_sections": _field_sections(active_tab, fields),
            "values": values,
            "control_plane": control_plane,
            "config_warnings": control_plane["warnings"],
            "group_tuning": GROUP_TUNING.get(active_tab, ()),
            "cognitive_stages": COGNITIVE_STAGE_MAP,
            "workflow_steps": workflow_steps_for(
                active_tab,
                pool_env_exists=bool(pool_env.get("exists")),
                pool_scale_job=bool(pool_scale_job),
                pool_scale_starting=bool(pool_scale_starting),
                build_job=bool(build_job),
                ingest_paused=bool(worker.paused),
            ),
            "services": services,
            "ingest_paused": worker.paused,
            "ingest_config": worker.config,
            "build_job": build_job,
            "build_history": build_history,
            "log_tail": log_tail,
            "pool_scale_job": pool_scale_job,
            "pool_scale_starting": pool_scale_starting,
            "pool_scale_history": pool_scale_history,
            "pool_scale_log_tail": pool_scale_log_tail,
            "migrate_job": migrate_job,
            "migrate_history": migrate_history,
            "migrate_log_tail": migrate_log_tail,
            "admin_env_path": store.admin_env_path,
            "proxy_env_path": store.proxy_env_path,
            "pool_env": pool_env,
            "can_restart_proxy": bool(settings.proxy_restart_cmd.strip()),
            "can_restart_admin": bool(settings.admin_restart_cmd.strip()),
            "can_restart_embed_pool": bool(settings.embed_pool_restart_cmd.strip()),
            "advanced_count": sum(
                1
                for field in SETTING_FIELDS
                if field.group == active_tab and field.key in ADVANCED_KEYS
            ),
        },
    )


@router.post("/settings/save/{group}")
async def settings_save(request: Request, group: str):
    if group not in SETTING_GROUPS:
        return flash_redirect("/settings", "Unknown settings group.", level="error")
    form = await request.form()
    ui_mode = _parse_mode(str(form.get("mode") or "basic"))
    store = _store(request)
    try:
        result = store.save_group(
            group,
            {
                k: str(v)
                for k, v in form.items()
                if k not in ("tab", "mode")
            },
        )
    except (ValueError, TypeError) as exc:
        return flash_redirect(settings_query(group, ui_mode), str(exc), level="error")

    worker = request.app.state.worker
    if group == "ingest":
        store.apply_to_worker(
            worker,
            zim_dir=settings.zim_dir,
            upload_dir=settings.upload_dir,
        )

    message = f"Saved {len(result.updated)} setting(s)."
    if result.pool_scale_updated:
        message += " Click Scale ingest capacity to apply pool changes."
    if result.restart_proxy:
        message += " Restart rag-proxy to apply proxy env changes."
    if result.restart_admin:
        message += " Restart rag-admin to apply admin env changes."
    return flash_redirect(settings_query(group, ui_mode), message)


@router.post("/settings/ingest/pause")
async def ingest_pause(request: Request):
    store = _store(request)
    store.set_ingest_paused(True)
    request.app.state.worker.set_paused(True)
    return _settings_flash("ingest", "Dense ingest paused.", mode=_request_mode(request))


@router.post("/settings/ingest/resume")
async def ingest_resume(request: Request):
    store = _store(request)
    worker = request.app.state.worker
    if on_demand_enabled():
        config = store.build_ingest_config(
            zim_dir=settings.zim_dir,
            upload_dir=settings.upload_dir,
        )
        urls = config.embed_urls or parse_ingest_embed_urls(embed_url=config.embed_url)
        ensure_embed_urls(urls)
    store.set_ingest_paused(False)
    worker.set_paused(False)
    return _settings_flash("ingest", "Dense ingest resumed.", mode=_request_mode(request))


@router.post("/settings/sparse/reindex")
async def sparse_reindex_now(request: Request):
    worker = request.app.state.worker
    mode = _request_mode(request)
    docs = trigger_sparse_reindex(worker.config)
    if docs is None:
        return _settings_flash(
            "ingest",
            "BM25 reindex failed or sparse sidecar not configured.",
            mode=mode,
            level="error",
        )
    return _settings_flash(
        "ingest", f"BM25 reindex complete ({docs} docs).", mode=mode
    )


@router.post("/settings/sidecars/migrate")
async def sidecar_migrate_start(request: Request):
    """One-time Qdrant → TurboVec/BM25 rebuild; safe no-op when already synced."""
    store = _store(request)
    job_runner = request.app.state.job_runner
    worker = request.app.state.worker
    mode = _request_mode(request)
    params = {
        "qdrant_url": store.get_value("QDRANT_URL", settings.qdrant_url),
        "collection": store.get_value("QDRANT_COLLECTION", settings.qdrant_collection),
        "turbovec_url": (getattr(worker.config, "turbovec_url", "") or "").strip()
        or store.get_value("TURBOVEC_URL", ""),
        "sparse_url": (getattr(worker.config, "sparse_index_url", "") or "").strip()
        or store.get_value("SPARSE_INDEX_URL", settings.sparse_index_url),
    }
    try:
        job_id = job_runner.start_sidecar_migrate(params)
    except RuntimeError as exc:
        return _settings_flash("ingest", str(exc), mode=mode, level="error")
    return _settings_flash(
        "ingest",
        f"Sidecar migration started (job {job_id[:8]}). "
        "Watch the step log below — already-synced sidecars are skipped.",
        mode=mode,
    )


@router.post("/settings/sidecars/migrate/stop")
async def sidecar_migrate_stop(request: Request):
    mode = _request_mode(request)
    stopped = request.app.state.job_runner.stop_active(JOB_SIDECAR_MIGRATE)
    if not stopped:
        return _settings_flash(
            "ingest",
            "No running sidecar migration to stop.",
            mode=mode,
            level="error",
        )
    return _settings_flash("ingest", "Sidecar migration stopped.", mode=mode)


@router.post("/settings/embed-pool/scale")
async def embed_pool_scale_start(request: Request):
    store = _store(request)
    job_runner = request.app.state.job_runner
    worker = request.app.state.worker
    mode = _request_mode(request)
    semantic_before = store.get_value("INGEST_CHUNK_SEMANTIC", "true").lower()
    was_paused = store.ingest_paused() or worker.paused

    # Pause now (also aborts mid-file), redirect immediately, then start the job.
    # Do not wait for multi-week ZIM embeds to finish — yield them back to pending.
    if not job_runner.try_begin_scale_prep():
        return _settings_flash(
            "ingest",
            "Capacity scale already starting or running — use Stop scale first.",
            mode=mode,
            level="error",
        )

    # Brief window for cooperative abort of the current embed batch only.
    drain_timeout_s = float(os.getenv("INGEST_SCALE_DRAIN_TIMEOUT_SEC", "60"))
    store.set_ingest_paused(True)
    worker.set_paused(True)
    preempted = worker.preempt_running()
    running = worker.running_file_count()

    def restore_pause_state() -> None:
        store.set_ingest_paused(was_paused)
        worker.set_paused(was_paused)

    def resume_after_scale() -> None:
        # Flash promises ingest resumes when the job completes; do not leave the
        # queue paused just because an earlier failed click had already paused it.
        store.set_ingest_paused(False)
        worker.set_paused(False)

    def on_success() -> None:
        synced = store.sync_pool_ingest_from_pool_env()
        store.apply_to_worker(
            worker,
            zim_dir=settings.zim_dir,
            upload_dir=settings.upload_dir,
        )
        if synced:
            log.info("capacity scale synced ingest keys: %s", ", ".join(synced))
        semantic_after = store.get_value("INGEST_CHUNK_SEMANTIC", "true").lower()
        if semantic_after != semantic_before:
            job_id = worker.requeue_all_files()
            log.warning(
                "capacity scale changed INGEST_CHUNK_SEMANTIC %s -> %s; "
                "requeued all ingest files (job %s)",
                semantic_before,
                semantic_after,
                job_id[:8],
            )
        resume_after_scale()

    def on_failure() -> None:
        restore_pause_state()

    def prepare_and_start() -> None:
        try:
            drained = worker.drain_active_files(timeout_s=drain_timeout_s)
            if not drained:
                forced = worker.force_requeue_running("paused for capacity scale")
                log.warning(
                    "capacity scale: force-requeued %s file(s) still marked running",
                    forced,
                )
            job_runner.start_embed_pool_scale(
                store.embed_pool_scale_params(),
                on_success=on_success,
                on_failure=on_failure,
            )
        except Exception:
            log.exception("capacity scale prep failed")
            restore_pause_state()
        finally:
            job_runner.end_scale_prep()

    threading.Thread(
        target=prepare_and_start,
        daemon=True,
        name="embed-pool-scale-prep",
    ).start()

    if preempted or running:
        msg = (
            f"Ingest paused; yielding {preempted or running} active file(s) back to the queue, "
            "then capacity scale starts. Refresh this page for the job log."
        )
    else:
        msg = (
            "Ingest paused; capacity scale starting. "
            "Refresh this page for the job log. Ingest resumes when the job completes."
        )
    return _settings_flash("ingest", msg, mode=mode)


@router.post("/settings/embed-pool/stop")
async def embed_pool_scale_stop(request: Request):
    mode = _request_mode(request)
    stopped = request.app.state.job_runner.stop_active(JOB_EMBED_POOL_SCALE)
    if not stopped:
        return _settings_flash(
            "ingest",
            "No running pool scale job to stop.",
            mode=mode,
            level="error",
        )
    return _settings_flash("ingest", "Embed pool scale stopped.", mode=mode)


@router.post("/settings/memgraph/build")
async def memgraph_build_start(request: Request):
    store = _store(request)
    job_runner = request.app.state.job_runner
    mode = _request_mode(request)
    params = store.memgraph_build_params()
    if not params.get("llm_model"):
        return _settings_flash(
            "memgraph_build",
            "Set Build LLM model before starting.",
            mode=mode,
            level="error",
        )
    try:
        job_id = job_runner.start_memgraph_build(params)
    except RuntimeError as exc:
        return _settings_flash("memgraph_build", str(exc), mode=mode, level="error")
    return _settings_flash(
        "memgraph_build",
        f"MemGraphRAG build started (job {job_id[:8]}).",
        mode=mode,
    )


@router.post("/settings/memgraph/stop")
async def memgraph_build_stop(request: Request):
    mode = _request_mode(request)
    stopped = request.app.state.job_runner.stop_active(JOB_MEMGRAPH_BUILD)
    if not stopped:
        return _settings_flash(
            "memgraph_build",
            "No running build to stop.",
            mode=mode,
            level="error",
        )
    return _settings_flash("memgraph_build", "MemGraphRAG build stopped.", mode=mode)


@router.get("/api/settings/status")
async def settings_status_api(request: Request) -> JSONResponse:
    store = _store(request)
    worker = request.app.state.worker
    job_runner = request.app.state.job_runner
    build_job = job_runner.active_job(JOB_MEMGRAPH_BUILD)
    pool_scale_job = job_runner.active_job(JOB_EMBED_POOL_SCALE)
    migrate_job = job_runner.active_job(JOB_SIDECAR_MIGRATE)
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
        "pool_scale_starting": job_runner.scale_starting(),
        "migrate_job": migrate_job,
        "log_tail": job_runner.tail_log(build_job["id"]) if build_job else "",
        "pool_scale_log_tail": job_runner.tail_log(pool_scale_job["id"]) if pool_scale_job else "",
        "migrate_log_tail": job_runner.tail_log(migrate_job["id"]) if migrate_job else "",
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
    mode = _request_mode(request)
    tab = request.query_params.get("tab") or "ingest"
    if tab not in SETTING_GROUPS:
        tab = "ingest"
    ok, msg = schedule_restart(settings.proxy_restart_cmd)
    if not ok:
        return _settings_flash(tab, msg, mode=mode, level="error")
    return _settings_flash(tab, f"rag-proxy restart scheduled. {msg}", mode=mode)


@router.post("/settings/restart/admin")
async def restart_admin_service(request: Request):
    mode = _request_mode(request)
    tab = request.query_params.get("tab") or "ingest"
    if tab not in SETTING_GROUPS:
        tab = "ingest"
    ok, msg = schedule_restart(settings.admin_restart_cmd)
    if not ok:
        return _settings_flash(tab, msg, mode=mode, level="error")
    return _settings_flash(
        tab,
        f"rag-admin restart scheduled; refresh this page in a few seconds. {msg}",
        mode=mode,
    )


@router.post("/settings/restart/embed-pool")
async def restart_embed_pool_service(request: Request):
    mode = _request_mode(request)
    ok, msg = schedule_restart(settings.embed_pool_restart_cmd)
    if not ok:
        return _settings_flash("ingest", msg, mode=mode, level="error")
    return _settings_flash(
        "ingest",
        f"Embed pool restart scheduled (re-applies pool env). {msg}",
        mode=mode,
    )

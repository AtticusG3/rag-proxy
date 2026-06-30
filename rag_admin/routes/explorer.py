"""Content Explorer: browse remote ZIM catalogs and manage subscriptions."""

from __future__ import annotations

import os
from urllib.parse import urlencode

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from rag_admin.catalog.item_context import SOURCE_META, describe_item
from rag_admin.catalog.listing_parser import infer_subscribable, is_internal_href
from rag_admin.catalog.providers import SOURCES, browse_source
from rag_admin.helpers import templates

router = APIRouter()


def _format_bytes(size: int | None) -> str:
    if size is None:
        return "-"
    if size < 1024:
        return f"{size} B"
    if size < 1024**2:
        return f"{size / 1024:.1f} KiB"
    if size < 1024**3:
        return f"{size / 1024**2:.1f} MiB"
    return f"{size / 1024**3:.2f} GiB"


def _explorer_url(source: str, path: str = "") -> str:
    params = {"source": source}
    if path:
        params["path"] = path
    return f"/explorer?{urlencode(params)}"


def _child_browse_path(path: str, item: object) -> str:
    href = getattr(item, "href", "")
    if not getattr(item, "is_directory", False):
        return ""
    if href.startswith(
        ("collection/", "item/", "search/", "cat/", "prefix/", "paper/")
    ):
        return href.rstrip("/")
    if not is_internal_href(href):
        return ""
    child = f"{path}/{href}" if path else href
    return child.rstrip("/")


def _path_title(source_id: str, path: str) -> str:
    if not path:
        return "Library root"
    leaf = path.rstrip("/").split("/")[-1]
    if source_id == "archive" and path.startswith("collection/"):
        return f"Collection: {leaf}"
    if source_id == "arxiv" and path.startswith("cat/"):
        return f"Category: {leaf}"
    return leaf.replace("_", " ").replace("%20", " ")


@router.get("/explorer", response_class=HTMLResponse)
async def explorer_page(
    request: Request,
    source: str = "dotsrc",
    path: str = "",
) -> HTMLResponse:
    db = request.app.state.db
    result = browse_source(source, path)
    all_subs = db.list_subscriptions()
    subscribed = {row["remote_url"] for row in all_subs}
    subscribed_packages = {
        (row["source_id"], row.get("package_key"), row.get("catalog_path") or "")
        for row in all_subs
        if row.get("package_key")
    }
    items = []
    browse_count = 0
    subscribe_count = 0
    for item in result.get("items", []):
        child_path = _child_browse_path(path, item)
        is_dir = item.is_directory and bool(child_path)
        subscribable = item.subscribable or infer_subscribable(
            item.name, is_directory=is_dir
        )
        context = describe_item(
            source,
            name=item.name,
            href=item.href,
            path=path,
            is_directory=is_dir,
            subscribable=subscribable,
            external_url=getattr(item, "external_url", None),
            modified=item.modified or "",
            package_key=getattr(item, "package_key", None),
            version_stamp=getattr(item, "version_stamp", None),
            hidden_older_versions=getattr(item, "hidden_older_versions", 0),
        )
        if context["kind"] != "pagination":
            browse_count += 1
        if subscribable:
            subscribe_count += 1
        items.append(
            {
                "name": item.name,
                "href": item.href,
                "browse_path": child_path,
                "browse_href": _explorer_url(source, child_path) if child_path else "",
                "url": item.url,
                "is_directory": is_dir,
                "external_url": getattr(item, "external_url", None),
                "size_label": _format_bytes(item.size_bytes),
                "modified": item.modified or "",
                "subscribed": (
                    (item.url in subscribed if item.url else False)
                    or (
                        getattr(item, "package_key", None)
                        and (source, item.package_key, path) in subscribed_packages
                    )
                ),
                "subscribable": subscribable,
                "package_key": getattr(item, "package_key", None),
                "version_stamp": getattr(item, "version_stamp", None),
                "hidden_older_versions": getattr(item, "hidden_older_versions", 0),
                **context,
            }
        )
    crumbs = []
    if path and "://" not in path:
        parts = path.strip("/").split("/")
        accum = ""
        for part in parts:
            accum = f"{accum}/{part}" if accum else part
            crumbs.append(
                {
                    "label": part.replace("_", " "),
                    "path": accum,
                    "href": _explorer_url(source, accum),
                }
            )
    active = SOURCES.get(source)
    source_meta = SOURCE_META.get(
        source,
        {"label": active.name if active else source, "tagline": "", "accent": "default"},
    )
    return templates.TemplateResponse(
        request,
        "explorer.html",
        {
            "sources": SOURCES,
            "active_source": source,
            "source_meta": source_meta,
            "path": path,
            "path_title": _path_title(source, path),
            "root_href": _explorer_url(source),
            "crumbs": crumbs,
            "items": items,
            "browse_stats": {
                "total": len(items),
                "visible": browse_count,
                "subscribable": subscribe_count,
            },
            "dedupe_hidden": result.get("dedupe_hidden", 0),
            "subscriptions_count": len(all_subs),
            "error": result.get("error"),
            "browse_url": result.get("browse_url"),
        },
    )


@router.get("/subscriptions", response_class=HTMLResponse)
async def subscriptions_page(request: Request) -> HTMLResponse:
    db = request.app.state.db
    rows = db.list_subscriptions()
    for row in rows:
        row["size_label"] = _format_bytes(row.get("remote_size"))
        row["auto_update"] = bool(row.get("auto_update"))
    return templates.TemplateResponse(
        request,
        "subscriptions.html",
        {"subscriptions": rows},
    )


@router.post("/explorer/subscribe")
async def subscribe_package(
    request: Request,
    source: str = Form(...),
    remote_url: str = Form(...),
    path: str = Form(""),
    package_key: str = Form(""),
    auto_update: str = Form("1"),
) -> RedirectResponse:
    catalog = request.app.state.catalog_manager
    catalog.subscribe(
        source,
        remote_url,
        auto_update=auto_update == "1",
        package_key=package_key or None,
        catalog_path=path,
    )
    return RedirectResponse(
        url=_explorer_url(source, path),
        status_code=303,
    )


@router.post("/subscriptions/check-updates")
async def check_updates(request: Request) -> RedirectResponse:
    catalog = request.app.state.catalog_manager
    queued = catalog.check_updates()
    db = request.app.state.db
    db.ingest.create_job(
        "catalog-update",
        job_type="catalog_update",
        message=f"queued {len(queued)} updates",
    )
    return RedirectResponse(url="/subscriptions", status_code=303)


@router.post("/subscriptions/toggle-auto")
async def toggle_auto(
    request: Request,
    sub_id: int = Form(...),
    enabled: str = Form("0"),
) -> RedirectResponse:
    db = request.app.state.db
    db.set_subscription_auto_update(sub_id, enabled == "1")
    return RedirectResponse(url="/subscriptions", status_code=303)


@router.post("/subscriptions/retry")
async def retry_subscription(
    request: Request,
    sub_id: int = Form(...),
) -> RedirectResponse:
    catalog = request.app.state.catalog_manager
    catalog.retry_subscription(sub_id)
    return RedirectResponse(url="/subscriptions", status_code=303)


@router.post("/subscriptions/delete")
async def delete_subscription(
    request: Request,
    sub_id: int = Form(...),
) -> RedirectResponse:
    db = request.app.state.db
    worker = request.app.state.worker
    row = db.delete_subscription(sub_id)
    if row and row.get("local_path") and os.path.isfile(row["local_path"]):
        worker.remove_file_from_index(row["local_path"])
    return RedirectResponse(url="/subscriptions", status_code=303)


@router.get("/api/catalog/browse")
async def api_browse(source: str = "dotsrc", path: str = "") -> JSONResponse:
    result = browse_source(source, path)
    items = [
        {
            "name": i.name,
            "url": i.url,
            "is_directory": i.is_directory,
            "size_bytes": i.size_bytes,
            "modified": i.modified,
            "subscribable": i.subscribable,
        }
        for i in result.get("items", [])
    ]
    return JSONResponse(
        {
            "source": source,
            "path": path,
            "items": items,
            "error": result.get("error"),
        }
    )

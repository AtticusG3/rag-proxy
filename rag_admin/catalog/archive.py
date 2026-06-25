"""Internet Archive catalog browsing via Advanced Search + Metadata APIs."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import quote, unquote

import httpx

from rag_admin.catalog.listing_parser import CatalogItem

IA_SEARCH_URL = "https://archive.org/advancedsearch.php"
IA_METADATA_URL = "https://archive.org/metadata"
IA_DOWNLOAD_URL = "https://archive.org/download"

USER_AGENT = "rag-admin/1.0 (local-ai-infra; +https://archive.org)"

INGESTABLE_SUFFIXES = (".zim", ".pdf", ".txt", ".md")
INGESTABLE_FORMAT_HINTS = frozenset(
    {
        "zim",
        "text pdf",
        "pdf",
        "txt",
        "djvutxt",
    }
)

PAGE_SIZE = 50

FEATURED_COLLECTIONS: tuple[tuple[str, str], ...] = (
    ("opensource", "Community texts (public domain)"),
    ("gutenberg", "Project Gutenberg mirrors"),
    ("internetarchivebooks", "Internet Archive Books"),
    ("usnationalarchives", "US National Archives"),
    ("nasa", "NASA media and documents"),
    ("prelinger", "Prelinger Archives"),
    ("folkscanomy", "Folkscanomy community uploads"),
    ("software", "Historical software library"),
)

CURATED_SEARCHES: tuple[tuple[str, str], ...] = (
    ("title%3Azim+AND+mediatype%3Atexts", "Offline Wikipedia / ZIM files"),
    ("subject%3Apython+AND+mediatype%3Atexts", "Texts about Python"),
)


def _client() -> httpx.Client:
    return httpx.Client(
        timeout=60.0,
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT},
    )


def _parse_path(subpath: str) -> tuple[str, dict[str, str]]:
    """Return (kind, params) from explorer subpath."""
    subpath = subpath.strip("/")
    if not subpath:
        return "root", {}
    parts = subpath.split("/")
    if parts[0] == "collection" and len(parts) >= 2:
        params: dict[str, str] = {"collection": parts[1]}
        if len(parts) >= 4 and parts[2] == "page":
            params["page"] = parts[3]
        return "collection", params
    if parts[0] == "search" and len(parts) >= 2:
        params = {"query": unquote(parts[1])}
        if len(parts) >= 4 and parts[2] == "page":
            params["page"] = parts[3]
        return "search", params
    if parts[0] == "item" and len(parts) >= 2:
        return "item", {"identifier": parts[1]}
    return "unknown", {}


def _safe_title(title: str, fallback: str) -> str:
    cleaned = re.sub(r"\s+", " ", title or "").strip()
    return cleaned[:120] if cleaned else fallback


def _browse_root() -> list[CatalogItem]:
    items: list[CatalogItem] = []
    for coll_id, label in FEATURED_COLLECTIONS:
        items.append(
            CatalogItem(
                name=label,
                href=f"collection/{coll_id}",
                url="",
                is_directory=True,
                size_bytes=None,
                modified=None,
            )
        )
    for query, label in CURATED_SEARCHES:
        items.append(
            CatalogItem(
                name=label,
                href=f"search/{query}",
                url="",
                is_directory=True,
                size_bytes=None,
                modified=None,
            )
        )
    return items


def _search_items(
    query: str,
    *,
    page: int = 1,
    href_prefix: str,
) -> list[CatalogItem]:
    with _client() as client:
        response = client.get(
            IA_SEARCH_URL,
            params={
                "q": query,
                "fl[]": ["identifier", "title", "date", "item_size"],
                "rows": PAGE_SIZE,
                "page": page,
                "output": "json",
            },
        )
        response.raise_for_status()
        payload = response.json()

    response_body = payload.get("response", {})
    docs = response_body.get("docs", [])
    num_found = int(response_body.get("numFound", 0))
    items: list[CatalogItem] = []
    for row in docs:
        identifier = row.get("identifier") or ""
        if not identifier:
            continue
        title = _safe_title(row.get("title", ""), identifier)
        size_raw = row.get("item_size")
        size_bytes = int(size_raw) if size_raw and str(size_raw).isdigit() else None
        items.append(
            CatalogItem(
                name=title,
                href=f"item/{identifier}",
                url="",
                is_directory=True,
                size_bytes=size_bytes,
                modified=row.get("date"),
            )
        )
    if page * PAGE_SIZE < num_found:
        items.append(
            CatalogItem(
                name="[Next page]",
                href=f"{href_prefix}/page/{page + 1}",
                url="",
                is_directory=True,
                size_bytes=None,
                modified=None,
            )
        )
    return items


def _is_ingestable_file(name: str, fmt: str) -> bool:
    lower = name.lower()
    if lower.endswith(INGESTABLE_SUFFIXES):
        return True
    fmt_lower = (fmt or "").lower()
    return any(hint in fmt_lower for hint in INGESTABLE_FORMAT_HINTS)


def _browse_item(identifier: str) -> list[CatalogItem]:
    with _client() as client:
        response = client.get(f"{IA_METADATA_URL}/{quote(identifier)}")
        response.raise_for_status()
        payload = response.json()

    items: list[CatalogItem] = []
    for file_row in payload.get("files", []):
        name = file_row.get("name") or ""
        if not name or name.endswith(".xml") or name.startswith("__"):
            continue
        fmt = file_row.get("format") or ""
        if not _is_ingestable_file(name, fmt):
            continue
        if name.lower().endswith(".epub"):
            continue
        size_raw = file_row.get("size")
        size_bytes = int(size_raw) if size_raw and str(size_raw).isdigit() else None
        mtime = file_row.get("mtime")
        modified = None
        if mtime and str(mtime).isdigit():
            from datetime import datetime, timezone

            modified = datetime.fromtimestamp(int(mtime), tz=timezone.utc).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        download_url = f"{IA_DOWNLOAD_URL}/{quote(identifier)}/{quote(name)}"
        items.append(
            CatalogItem(
                name=name,
                href=name,
                url=download_url,
                is_directory=False,
                size_bytes=size_bytes,
                modified=modified,
                subscribable=True,
            )
        )
    return sorted(items, key=lambda i: i.name.lower())


def browse_archive(subpath: str = "") -> dict[str, Any]:
    kind, params = _parse_path(subpath)
    browse_url = IA_SEARCH_URL
    try:
        if kind == "root":
            items = _browse_root()
        elif kind == "collection":
            coll = params["collection"]
            page = int(params.get("page", "1"))
            query = f"collection:{coll}"
            href_prefix = f"collection/{coll}"
            items = _search_items(query, page=page, href_prefix=href_prefix)
            browse_url = f"{IA_SEARCH_URL}?q={quote(query)}"
        elif kind == "search":
            query = params["query"]
            page = int(params.get("page", "1"))
            href_prefix = f"search/{quote(query, safe='')}"
            items = _search_items(query, page=page, href_prefix=href_prefix)
            browse_url = f"{IA_SEARCH_URL}?q={quote(query)}"
        elif kind == "item":
            items = _browse_item(params["identifier"])
            browse_url = f"{IA_METADATA_URL}/{params['identifier']}"
        else:
            return {
                "path": subpath,
                "items": [],
                "error": f"Unknown archive path: {subpath}",
                "browse_url": "https://archive.org/",
            }
    except Exception as exc:
        return {
            "path": subpath,
            "items": [],
            "error": str(exc),
            "browse_url": browse_url,
        }
    return {
        "path": subpath,
        "items": items,
        "error": None,
        "browse_url": browse_url,
    }

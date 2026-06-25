"""Catalog source definitions (tabs in Content Explorer)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import httpx

from rag_admin.catalog.archive import browse_archive
from rag_admin.catalog.arxiv import browse_arxiv
from rag_admin.catalog.listing_parser import (
    CatalogItem,
    is_directory_listing,
    is_internal_href,
    parse_directory_listing,
    same_origin,
)
from rag_admin.catalog.zim_versions import KIWIX_SOURCES, dedupe_kiwix_items

USER_AGENT = "rag-admin/1.0 (local-ai-infra)"


@dataclass(frozen=True)
class CatalogSource:
    id: str
    name: str
    base_url: str
    enabled: bool
    description: str = ""


SOURCES: dict[str, CatalogSource] = {
    "dotsrc": CatalogSource(
        id="dotsrc",
        name="Kiwix (dotsrc)",
        base_url="https://mirrors.dotsrc.org/kiwix/zim/",
        enabled=True,
        description="Danish dotsrc mirror of the Kiwix ZIM library",
    ),
    "kiwix": CatalogSource(
        id="kiwix",
        name="Kiwix (official)",
        base_url="https://lb.download.kiwix.org/zim/",
        enabled=True,
        description="Official Kiwix ZIM mirror (lb.download.kiwix.org)",
    ),
    "archive": CatalogSource(
        id="archive",
        name="Internet Archive",
        base_url="https://archive.org/",
        enabled=True,
        description="Browse IA collections and subscribe to ZIM, PDF, and text files",
    ),
    "arxiv": CatalogSource(
        id="arxiv",
        name="arXiv",
        base_url="https://arxiv.org/",
        enabled=True,
        description="Browse arXiv categories and subscribe to paper PDFs for ingest",
    ),
    "gutenberg": CatalogSource(
        id="gutenberg",
        name="Project Gutenberg",
        base_url="",
        enabled=False,
        description="Planned: direct .txt bulk ingest (not ZIM)",
    ),
    "openlibrary": CatalogSource(
        id="openlibrary",
        name="Open Library",
        base_url="",
        enabled=False,
        description="Planned: OL dumps and borrowable texts",
    ),
    "wikimedia": CatalogSource(
        id="wikimedia",
        name="Wikimedia Dumps",
        base_url="",
        enabled=False,
        description="Planned: XML/sql dumps (heavy; prefer ZIM for RAG)",
    ),
}


def get_source(source_id: str) -> CatalogSource:
    if source_id not in SOURCES:
        raise KeyError(f"unknown catalog source: {source_id}")
    return SOURCES[source_id]


def _normalize_mirror_subpath(subpath: str) -> str:
    cleaned = subpath.strip().lstrip("/")
    if not cleaned:
        return ""
    if cleaned.startswith(("http://", "https://", "mailto:", "ftp://")):
        raise ValueError(
            "Invalid browse path (external URL). Return to root and pick a folder."
        )
    if "://" in cleaned:
        raise ValueError("Invalid browse path. Return to root and pick a folder.")
    return cleaned


def browse_source(source_id: str, subpath: str = "") -> dict[str, Any]:
    source = get_source(source_id)
    if not source.enabled:
        return {
            "source": source,
            "path": subpath,
            "items": [],
            "error": "Source not enabled yet",
        }

    if source_id == "archive":
        result = browse_archive(subpath)
        return {"source": source, **result}

    if source_id == "arxiv":
        result = browse_arxiv(subpath)
        return {"source": source, **result}

    if not source.base_url:
        return {
            "source": source,
            "path": subpath,
            "items": [],
            "error": "Source not enabled yet",
        }

    try:
        subpath = _normalize_mirror_subpath(subpath)
    except ValueError as exc:
        return {
            "source": source,
            "path": "",
            "items": [],
            "error": str(exc),
            "browse_url": source.base_url,
        }

    url = source.base_url + subpath
    if not url.endswith("/"):
        url += "/"
    try:
        with httpx.Client(
            timeout=60.0,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
        ) as client:
            response = client.get(url)
            response.raise_for_status()
            final_url = str(response.url)
            if not same_origin(final_url, source.base_url):
                return {
                    "source": source,
                    "path": subpath,
                    "items": [],
                    "error": (
                        f"Mirror redirected to {final_url}. "
                        "Use dotsrc or lb.download.kiwix.org paths only."
                    ),
                    "browse_url": final_url,
                }
            if not is_directory_listing(response.text):
                return {
                    "source": source,
                    "path": subpath,
                    "items": [],
                    "error": (
                        "Page is not a directory listing. "
                        "The official Kiwix hub moved; this tab uses lb.download.kiwix.org."
                    ),
                    "browse_url": final_url,
                }
            items = parse_directory_listing(response.text, final_url)
            dedupe_hidden = 0
            if source_id in KIWIX_SOURCES:
                zim_before = sum(
                    1 for item in items if item.name.lower().endswith(".zim")
                )
                items = dedupe_kiwix_items(items)
                zim_after = sum(
                    1 for item in items if item.name.lower().endswith(".zim")
                )
                dedupe_hidden = max(0, zim_before - zim_after)
    except Exception as exc:
        return {
            "source": source,
            "path": subpath,
            "items": [],
            "error": str(exc),
            "browse_url": url,
            "dedupe_hidden": 0,
        }
    return {
        "source": source,
        "path": subpath,
        "items": items,
        "error": None,
        "browse_url": url,
        "dedupe_hidden": dedupe_hidden if source_id in KIWIX_SOURCES else 0,
    }


def fetch_remote_meta(url: str) -> dict[str, Any]:
    """HEAD request for size and last-modified."""
    with httpx.Client(
        timeout=60.0,
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT},
    ) as client:
        response = client.head(url)
        if response.status_code >= 400:
            response = client.get(url, headers={"Range": "bytes=0-0"})
        response.raise_for_status()
        size = response.headers.get("Content-Length")
        modified = response.headers.get("Last-Modified")
        return {
            "size_bytes": int(size) if size and size.isdigit() else None,
            "modified": modified,
        }

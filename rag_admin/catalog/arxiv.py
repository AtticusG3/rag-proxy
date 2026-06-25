"""arXiv catalog browsing via the export API (Atom)."""

from __future__ import annotations

import os
import re
import threading
import time
import xml.etree.ElementTree as ET
from typing import Any
from urllib.parse import quote, unquote

import httpx

from rag_admin.catalog.listing_parser import CatalogItem

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ARXIV_ABS_URL = "https://arxiv.org/abs"
ARXIV_PDF_URL = "https://arxiv.org/pdf"

# arXiv asks for >=3s between requests and a contact mailto in User-Agent.
ARXIV_MIN_INTERVAL_S = 3.1
ARXIV_USER_AGENT = os.getenv(
    "ARXIV_USER_AGENT",
    "rag-admin-buster/1.0 (mailto:admin@localhost)",
)

ATOM_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}

PAGE_SIZE = 25

_ARXIV_LOCK = threading.Lock()
_ARXIV_LAST_REQUEST_MONO = 0.0

ATOM_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}

PAGE_SIZE = 25

TOP_PREFIXES: tuple[tuple[str, str], ...] = (
    ("cs", "Computer Science"),
    ("econ", "Economics"),
    ("eess", "Electrical Engineering and Systems Science"),
    ("math", "Mathematics"),
    ("physics", "Physics"),
    ("q-bio", "Quantitative Biology"),
    ("q-fin", "Quantitative Finance"),
    ("stat", "Statistics"),
)

CS_CATEGORIES: tuple[tuple[str, str], ...] = (
    ("cs.AI", "Artificial Intelligence"),
    ("cs.LG", "Machine Learning"),
    ("cs.CL", "Computation and Language"),
    ("cs.CV", "Computer Vision"),
    ("cs.NE", "Neural and Evolutionary Computing"),
    ("cs.SE", "Software Engineering"),
    ("cs.DS", "Data Structures and Algorithms"),
    ("cs.IR", "Information Retrieval"),
    ("cs.CR", "Cryptography and Security"),
    ("cs.RO", "Robotics"),
    ("cs.HC", "Human-Computer Interaction"),
    ("cs.PL", "Programming Languages"),
)

POPULAR_CATEGORIES: tuple[tuple[str, str], ...] = (
    ("stat.ML", "Machine Learning (stat)"),
    ("math.PR", "Probability"),
    ("physics.comp-ph", "Computational Physics"),
)


def _arxiv_get(params: dict[str, Any]) -> httpx.Response:
    """Rate-limited arXiv API GET with 429 retry."""
    global _ARXIV_LAST_REQUEST_MONO
    last_error: Exception | None = None
    with httpx.Client(
        timeout=60.0,
        follow_redirects=True,
        headers={"User-Agent": ARXIV_USER_AGENT},
    ) as client:
        for attempt in range(5):
            with _ARXIV_LOCK:
                wait_s = ARXIV_MIN_INTERVAL_S - (
                    time.monotonic() - _ARXIV_LAST_REQUEST_MONO
                )
                if wait_s > 0:
                    time.sleep(wait_s)
                response = client.get(ARXIV_API_URL, params=params)
                _ARXIV_LAST_REQUEST_MONO = time.monotonic()
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                delay = ARXIV_MIN_INTERVAL_S
                if retry_after and str(retry_after).isdigit():
                    delay = max(delay, float(retry_after))
                else:
                    delay = max(delay, 5.0 * (attempt + 1))
                last_error = httpx.HTTPStatusError(
                    "arXiv rate limit",
                    request=response.request,
                    response=response,
                )
                time.sleep(delay)
                continue
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                last_error = exc
                raise
            return response
    if last_error is not None:
        raise last_error
    raise RuntimeError("arXiv request failed after retries")


def _arxiv_error_message(exc: Exception) -> str:
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429:
        return (
            "arXiv rate limit reached (max 1 request per 3 seconds). "
            "Wait a few seconds, then reload this page."
        )
    return str(exc)


def _parse_path(subpath: str) -> tuple[str, dict[str, str]]:
    subpath = subpath.strip("/")
    if not subpath:
        return "root", {}
    parts = subpath.split("/")
    if parts[0] == "prefix" and len(parts) >= 2:
        return "prefix", {"prefix": parts[1]}
    if parts[0] == "cat" and len(parts) >= 2:
        params: dict[str, str] = {"category": parts[1]}
        if len(parts) >= 4 and parts[2] == "page":
            params["page"] = parts[3]
        return "category", params
    if parts[0] == "search" and len(parts) >= 2:
        params = {"query": unquote(parts[1])}
        if len(parts) >= 4 and parts[2] == "page":
            params["page"] = parts[3]
        return "search", params
    if parts[0] == "paper" and len(parts) >= 2:
        return "paper", {"paper_id": parts[1]}
    return "unknown", {}


def _arxiv_id_from_entry_id(entry_id: str) -> str:
    # http://arxiv.org/abs/2301.12345v2 -> 2301.12345
    match = re.search(r"arxiv\.org/abs/(.+)$", entry_id)
    if not match:
        return entry_id.rsplit("/", 1)[-1]
    paper_id = match.group(1)
    return re.sub(r"v\d+$", "", paper_id)


def _browse_root() -> list[CatalogItem]:
    items: list[CatalogItem] = []
    for cat_id, label in POPULAR_CATEGORIES:
        items.append(
            CatalogItem(
                name=f"{cat_id} - {label}",
                href=f"cat/{cat_id}",
                url="",
                is_directory=True,
                size_bytes=None,
                modified=None,
            )
        )
    for prefix, label in TOP_PREFIXES:
        items.append(
            CatalogItem(
                name=f"{prefix}/ - {label}",
                href=f"prefix/{prefix}",
                url="",
                is_directory=True,
                size_bytes=None,
                modified=None,
            )
        )
    items.append(
        CatalogItem(
            name='Search: "transformer architecture"',
            href="search/all%3Atransformer+architecture",
            url="",
            is_directory=True,
            size_bytes=None,
            modified=None,
        )
    )
    return items


def _browse_prefix(prefix: str) -> list[CatalogItem]:
    items: list[CatalogItem] = []
    if prefix == "cs":
        for cat_id, label in CS_CATEGORIES:
            items.append(
                CatalogItem(
                    name=f"{cat_id} - {label}",
                    href=f"cat/{cat_id}",
                    url="",
                    is_directory=True,
                    size_bytes=None,
                    modified=None,
                )
            )
        return items
    items.append(
        CatalogItem(
            name=f"Browse all {prefix}.* papers",
            href=f"cat/{prefix}",
            url="",
            is_directory=True,
            size_bytes=None,
            modified=None,
        )
    )
    return items


def _query_papers(
    search_query: str,
    *,
    page: int = 0,
    href_prefix: str,
) -> list[CatalogItem]:
    start = page * PAGE_SIZE
    params = {
        "search_query": search_query,
        "start": start,
        "max_results": PAGE_SIZE,
        "sortBy": "lastUpdatedDate",
        "sortOrder": "descending",
    }
    response = _arxiv_get(params)
    root = ET.fromstring(response.text)

    items: list[CatalogItem] = []
    for entry in root.findall("atom:entry", ATOM_NS):
        entry_id = entry.findtext("atom:id", default="", namespaces=ATOM_NS)
        paper_id = _arxiv_id_from_entry_id(entry_id)
        title = re.sub(r"\s+", " ", entry.findtext("atom:title", default="", namespaces=ATOM_NS)).strip()
        published = entry.findtext("atom:published", default="", namespaces=ATOM_NS)
        modified = entry.findtext("atom:updated", default="", namespaces=ATOM_NS)
        pdf_url = f"{ARXIV_PDF_URL}/{paper_id}.pdf"
        items.append(
            CatalogItem(
                name=f"{paper_id}: {title[:100]}",
                href=f"paper/{paper_id}",
                url=pdf_url,
                is_directory=False,
                size_bytes=None,
                modified=(modified or published)[:10] if (modified or published) else None,
                subscribable=True,
            )
        )

    total_results = 0
    total_node = root.find("{http://a9.com/-/spec/opensearch/1.1/}totalResults")
    if total_node is not None and total_node.text and total_node.text.isdigit():
        total_results = int(total_node.text)

    if start + PAGE_SIZE < total_results:
        next_page = page + 1
        items.append(
            CatalogItem(
                name="[Next page]",
                href=f"{href_prefix}/page/{next_page}",
                url="",
                is_directory=True,
                size_bytes=None,
                modified=None,
            )
        )
    return items


def _browse_paper(paper_id: str) -> list[CatalogItem]:
    params = {"id_list": paper_id}
    response = _arxiv_get(params)
    root = ET.fromstring(response.text)

    entry = root.find("atom:entry", ATOM_NS)
    if entry is None:
        return []

    title = re.sub(r"\s+", " ", entry.findtext("atom:title", default="", namespaces=ATOM_NS)).strip()
    summary = re.sub(r"\s+", " ", entry.findtext("atom:summary", default="", namespaces=ATOM_NS)).strip()
    pdf_url = f"{ARXIV_PDF_URL}/{paper_id}.pdf"
    abs_url = f"{ARXIV_ABS_URL}/{paper_id}"

    items = [
        CatalogItem(
            name=f"{paper_id}.pdf",
            href=pdf_url,
            url=pdf_url,
            is_directory=False,
            size_bytes=None,
            modified=None,
            subscribable=True,
        ),
        CatalogItem(
            name="Abstract (metadata only)",
            href=abs_url,
            url=abs_url,
            is_directory=False,
            size_bytes=len(summary.encode("utf-8")) if summary else None,
            modified=None,
            subscribable=False,
        ),
    ]
    if title:
        items.insert(
            0,
            CatalogItem(
                name=f"Title: {title[:100]}",
                href="",
                url="",
                is_directory=False,
                size_bytes=None,
                modified=None,
            ),
        )
    return items


def browse_arxiv(subpath: str = "") -> dict[str, Any]:
    kind, params = _parse_path(subpath)
    browse_url = ARXIV_API_URL
    try:
        if kind == "root":
            items = _browse_root()
        elif kind == "prefix":
            items = _browse_prefix(params["prefix"])
        elif kind == "category":
            category = params["category"]
            page = int(params.get("page", "0"))
            query = f"cat:{category}"
            href_prefix = f"cat/{category}"
            items = _query_papers(query, page=page, href_prefix=href_prefix)
            browse_url = f"{ARXIV_API_URL}?search_query={quote(query)}"
        elif kind == "search":
            query = params["query"]
            page = int(params.get("page", "0"))
            href_prefix = f"search/{quote(query, safe='')}"
            items = _query_papers(query, page=page, href_prefix=href_prefix)
            browse_url = f"{ARXIV_API_URL}?search_query={quote(query)}"
        elif kind == "paper":
            items = _browse_paper(params["paper_id"])
            browse_url = f"{ARXIV_ABS_URL}/{params['paper_id']}"
        else:
            return {
                "path": subpath,
                "items": [],
                "error": f"Unknown arXiv path: {subpath}",
                "browse_url": "https://arxiv.org/",
            }
    except Exception as exc:
        return {
            "path": subpath,
            "items": [],
            "error": _arxiv_error_message(exc),
            "browse_url": browse_url,
        }
    return {
        "path": subpath,
        "items": items,
        "error": None,
        "browse_url": browse_url,
    }

"""Parse Apache-style HTML directory listings."""

from __future__ import annotations

import re
from dataclasses import dataclass
from html import unescape
from urllib.parse import urljoin, urlparse

_ROW_RE = re.compile(r"<tr[^>]*>([\s\S]*?)</tr>", re.IGNORECASE)
_LINK_RE = re.compile(r'<a href="([^"]+)"[^>]*>([^<]*)</a>', re.IGNORECASE)
_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*([KMGTP]?i?B)", re.IGNORECASE)


@dataclass(frozen=True)
class CatalogItem:
    name: str
    href: str
    url: str
    is_directory: bool
    size_bytes: int | None
    modified: str | None
    subscribable: bool = False
    external_url: str | None = None
    package_key: str | None = None
    version_stamp: str | None = None
    hidden_older_versions: int = 0


def is_directory_listing(html: str) -> bool:
    """True when HTML looks like an Apache/nginx autoindex page."""
    lower = html.lower()
    return (
        "index of" in lower
        or '<table id="list"' in lower
        or ("<tbody>" in lower and "parent directory" in lower)
    )


def is_internal_href(href: str) -> bool:
    """Only relative paths are browsable inside a mirror."""
    if not href or href.startswith("#"):
        return False
    if href.startswith(("http://", "https://", "mailto:", "ftp://")):
        return False
    if href.startswith("//"):
        return False
    return True


def infer_subscribable(name: str, *, is_directory: bool) -> bool:
    if is_directory:
        return False
    return name.lower().endswith((".zim", ".pdf", ".txt", ".md"))


def _parse_size(token: str) -> int | None:
    token = token.strip()
    if token in ("-", ""):
        return None
    match = _SIZE_RE.search(token)
    if not match:
        return None
    value = float(match.group(1))
    unit = match.group(2).upper()
    multipliers = {
        "B": 1,
        "KIB": 1024,
        "KB": 1000,
        "MIB": 1024**2,
        "MB": 1000**2,
        "GIB": 1024**3,
        "GB": 1000**3,
        "TIB": 1024**4,
        "TB": 1000**4,
    }
    return int(value * multipliers.get(unit, 1))


def _listing_body(html: str) -> str:
    """Prefer tbody content so header/sort links are not parsed as rows."""
    match = re.search(r"<tbody[^>]*>(.*?)</tbody>", html, re.IGNORECASE | re.DOTALL)
    return match.group(1) if match else html


def _append_item(
    items: list[CatalogItem],
    seen: set[str],
    *,
    name: str,
    href: str,
    base_url: str,
    is_directory: bool,
    size_bytes: int | None,
    modified: str | None,
) -> None:
    if not is_internal_href(href):
        if href.startswith(("http://", "https://")):
            external = urljoin(base_url, href)
            key = external.rstrip("/")
            if key in seen:
                return
            seen.add(key)
            items.append(
                CatalogItem(
                    name=name.rstrip("/"),
                    href=href,
                    url=external,
                    is_directory=False,
                    size_bytes=None,
                    modified=None,
                    external_url=external,
                )
            )
        return
    url = urljoin(base_url, href)
    key = url.rstrip("/")
    if key in seen:
        return
    seen.add(key)
    subscribable = infer_subscribable(name, is_directory=is_directory)
    items.append(
        CatalogItem(
            name=name.rstrip("/"),
            href=href,
            url=url,
            is_directory=is_directory,
            size_bytes=size_bytes,
            modified=modified,
            subscribable=subscribable,
        )
    )


def _items_from_rows(html: str, base_url: str) -> list[CatalogItem]:
    items: list[CatalogItem] = []
    seen: set[str] = set()
    for row in _ROW_RE.finditer(html):
        row_html = row.group(1)
        link = _LINK_RE.search(row_html)
        if not link:
            continue
        href = unescape(link.group(1))
        name = unescape(link.group(2)).strip() or href
        if href in ("../", "/") or name.lower().startswith("parent"):
            continue
        if href.startswith("?"):
            continue
        cells = re.findall(r"<td[^>]*>(.*?)</td>", row_html, re.IGNORECASE | re.DOTALL)
        size_bytes = None
        modified = None
        if len(cells) >= 2:
            size_bytes = _parse_size(re.sub(r"<[^>]+>", "", cells[-2]))
        if len(cells) >= 3:
            modified = re.sub(r"<[^>]+>", "", cells[-1]).strip() or None
        _append_item(
            items,
            seen,
            name=name,
            href=href,
            base_url=base_url,
            is_directory=href.endswith("/"),
            size_bytes=size_bytes,
            modified=modified,
        )
    return sorted(items, key=lambda i: (not i.is_directory, i.name.lower()))


def _items_from_links(html: str, base_url: str) -> list[CatalogItem]:
    items: list[CatalogItem] = []
    seen: set[str] = set()
    for match in _LINK_RE.finditer(html):
        href = unescape(match.group(1))
        name = unescape(match.group(2)).strip() or href
        if href in ("../", "/") or name.lower().startswith("parent"):
            continue
        if href.startswith("?"):
            continue
        _append_item(
            items,
            seen,
            name=name,
            href=href,
            base_url=base_url,
            is_directory=href.endswith("/"),
            size_bytes=None,
            modified=None,
        )
    return sorted(items, key=lambda i: (not i.is_directory, i.name.lower()))


def same_origin(url: str, base_url: str) -> bool:
    left = urlparse(url)
    right = urlparse(base_url)
    return left.scheme == right.scheme and left.netloc == right.netloc


def parse_directory_listing(html: str, base_url: str) -> list[CatalogItem]:
    """Parse an index page into browseable items."""
    if not base_url.endswith("/"):
        base_url += "/"
    if not is_directory_listing(html):
        return []
    body = _listing_body(html)
    items = _items_from_rows(body, base_url)
    if items:
        return items
    return _items_from_links(body, base_url)

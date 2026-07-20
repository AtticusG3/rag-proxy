"""Extract articles from ZIM archives for embedding."""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from html import unescape
from typing import Iterator

from ingest.zim_sanitize import sanitize_zim_html

log = logging.getLogger("ingest.zim")

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class ZimArticle:
    title: str
    url: str
    text: str


def strip_html(html: str) -> str:
    """Legacy flat HTML strip (spike / comparisons). Prefer sanitize_zim_html for ingest."""
    text = _HTML_TAG_RE.sub(" ", html)
    text = unescape(text)
    return _WS_RE.sub(" ", text).strip()


def _normalize_entry(raw: bytes, mimetype: str, *, title: str, url: str) -> str | None:
    text = raw.decode("utf-8", errors="replace")
    mime = (mimetype or "").lower()
    if "html" in mime:
        return sanitize_zim_html(text, title=title, url=url)
    cleaned = _WS_RE.sub(" ", text).strip()
    if len(cleaned) < 80:
        return None
    return cleaned


def _iter_libzim(zim_path: str, max_articles: int) -> Iterator[ZimArticle]:
    from libzim.reader import Archive

    archive = Archive(zim_path)
    entry_count = int(getattr(archive, "all_entry_count", 0) or 0)
    if entry_count <= 0:
        return

    count = 0
    for index in range(entry_count):
        try:
            entry = archive._get_entry_by_id(index)
        except Exception:
            continue
        if entry.is_redirect:
            continue
        try:
            item = entry.get_item()
            mimetype = getattr(item, "mimetype", "") or ""
            raw = bytes(item.content)
        except Exception:
            continue
        if mimetype and not any(
            token in mimetype.lower() for token in ("text/html", "text/plain", "application/xhtml")
        ):
            continue
        title = entry.title or entry.path
        url = entry.path
        text = _normalize_entry(raw, mimetype, title=title, url=url)
        if text is None:
            continue
        yield ZimArticle(title=title, url=url, text=text)
        count += 1
        if max_articles > 0 and count >= max_articles:
            break


def _zimdump_path() -> str | None:
    return shutil.which("zimdump")


def _iter_zimdump(zim_path: str, max_articles: int) -> Iterator[ZimArticle]:
    zimdump = _zimdump_path()
    if not zimdump:
        raise RuntimeError(
            "ZIM ingest failed: libzim could not read the archive and zimdump is not installed. "
            "Install zim-tools (apt install zim-tools) or upgrade libzim."
        )
    proc = subprocess.run(
        [zimdump, "list", zim_path],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"zimdump list failed: {proc.stderr.strip()}")
    count = 0
    for line in proc.stdout.splitlines():
        path = line.strip()
        if not path or path.startswith("#"):
            continue
        if not path.endswith(".html") and "/A/" not in path and not path.startswith("A/"):
            continue
        dump = subprocess.run(
            [zimdump, "show", "--url", path, zim_path],
            capture_output=True,
            text=True,
            check=False,
        )
        if dump.returncode != 0:
            continue
        title = path.rsplit("/", 1)[-1]
        text = sanitize_zim_html(dump.stdout, title=title, url=path)
        if text is None:
            continue
        yield ZimArticle(title=title, url=path, text=text)
        count += 1
        if max_articles > 0 and count >= max_articles:
            break


def iter_zim_articles(
    zim_path: str,
    *,
    max_articles: int = 0,
) -> Iterator[ZimArticle]:
    """Yield text articles from a ZIM file."""
    libzim_error: Exception | None = None
    try:
        yield from _iter_libzim(zim_path, max_articles)
        return
    except ImportError as exc:
        libzim_error = exc
        log.info("libzim not installed, falling back to zimdump")
    except Exception as exc:
        libzim_error = exc
        log.warning("libzim reader failed (%s), trying zimdump", exc)

    try:
        yield from _iter_zimdump(zim_path, max_articles)
    except Exception as exc:
        if libzim_error is not None:
            raise RuntimeError(
                f"ZIM ingest failed via libzim ({libzim_error}) and zimdump ({exc})"
            ) from exc
        raise

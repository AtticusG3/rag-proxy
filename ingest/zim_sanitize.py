"""Sanitize ZIM article HTML before chunking.

Always: drop script/style/noscript, prefer the largest main-content root when
present, skip site chrome (nav/header/footer), preserve paragraph breaks.
MediaWiki-like titles/paths also skip non-article namespaces and drop wiki
chrome / trailing reference sections.
"""

from __future__ import annotations

import re
from html import unescape
from html.parser import HTMLParser
from typing import Iterable

_WS_LINE_RE = re.compile(r"[ \t\f\v]+")
_MULTI_NL_RE = re.compile(r"\n{3,}")

# Drop entirely (and their inner text).
_SKIP_TAGS = frozenset(
    {
        "script",
        "style",
        "noscript",
        "svg",
        "template",
        "nav",
        "header",
        "footer",
    }
)
_BLOCK_TAGS = frozenset(
    {
        "address",
        "article",
        "aside",
        "blockquote",
        "br",
        "dd",
        "div",
        "dl",
        "dt",
        "figcaption",
        "figure",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "hr",
        "li",
        "main",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "tbody",
        "td",
        "th",
        "thead",
        "tr",
        "ul",
    }
)

# Content roots seen in example_zim/ (DevDocs, NHS, SE) plus MediaWiki.
_CONTENT_ROOT_IDS = frozenset(
    {
        "mw-content-text",
        "content",
        "main-content",
        "maincontent",
        "main",
    }
)
_CONTENT_ROOT_CLASSES = frozenset(
    {
        "mw-parser-output",
        "content",
        "_content",
        "main-content",
        "post-text",
        "answer",
        "question",
        "s-prose",
    }
)
_CONTENT_ROOT_TAGS = frozenset({"main", "article"})

# Prefer a content root only if it keeps at least this share of full-doc text.
_CONTENT_ROOT_MIN_RATIO = 0.35

_WIKI_SKIP_IDS = frozenset({"toc"})
_WIKI_SKIP_CLASSES = frozenset(
    {
        "navbox",
        "vertical-navbox",
        "hatnote",
        "mw-editsection",
        "toc",
    }
)

_MW_NAMESPACE_RE = re.compile(
    r"(?i)\b(Category|File|Image|Template|Special|User|Wikipedia|Help|Portal|"
    r"Talk|Module|MediaWiki|Draft):"
)
_MW_PATH_HINTS = ("/wiki/", "A/Category:", "A/File:", "A/Template:", "A/Special:")

_DROP_SECTION_TITLES = frozenset(
    {
        "references",
        "external links",
        "see also",
        "further reading",
        "notes",
        "bibliography",
        "citations",
    }
)

_HEADING_TAGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})


def is_mediawiki_like(*, title: str = "", url: str = "") -> bool:
    """True when title/path looks like MediaWiki (enables wiki-only filters)."""
    blob = f"{title}\n{url}"
    if _MW_NAMESPACE_RE.search(blob):
        return True
    lower = blob.lower()
    return any(hint.lower() in lower for hint in _MW_PATH_HINTS)


def should_skip_mediawiki_title(title: str, url: str = "") -> bool:
    """Skip non-article MediaWiki namespaces."""
    if not is_mediawiki_like(title=title, url=url):
        return False
    return bool(_MW_NAMESPACE_RE.search(title) or _MW_NAMESPACE_RE.search(url))


def _attr_map(attrs: Iterable[tuple[str, str | None]]) -> dict[str, str]:
    return {k.lower(): (v or "") for k, v in attrs}


def _class_set(attrs: dict[str, str]) -> set[str]:
    return {c for c in attrs.get("class", "").split() if c}


def _root_label(tag: str, attrs: dict[str, str]) -> str:
    el_id = attrs.get("id", "")
    if el_id:
        return f"#{el_id}"
    classes = sorted(_class_set(attrs))
    if classes:
        return f".{classes[0]}"
    return tag


def _is_content_root(tag: str, attrs: dict[str, str]) -> bool:
    el_id = attrs.get("id", "")
    if el_id in _CONTENT_ROOT_IDS:
        return True
    if _class_set(attrs).intersection(_CONTENT_ROOT_CLASSES):
        return True
    return tag in _CONTENT_ROOT_TAGS


def _is_wiki_chrome(attrs: dict[str, str]) -> bool:
    if attrs.get("id", "") in _WIKI_SKIP_IDS:
        return True
    return bool(_class_set(attrs).intersection(_WIKI_SKIP_CLASSES))


def _extract_content_root(html: str, *, wiki_mode: bool) -> tuple[str, str]:
    """Return (fragment, root_label). Prefer largest root that keeps enough text."""
    finder = _ContentRootFinder()
    try:
        finder.feed(html)
        finder.close()
    except Exception:
        return html, ""
    if not finder.candidates:
        return html, ""

    full_text = html_to_text(html, wiki_mode=wiki_mode)
    full_n = len(full_text)
    best_html = ""
    best_label = ""
    best_n = -1
    for label, fragment in finder.candidates:
        text_n = len(html_to_text(fragment, wiki_mode=wiki_mode))
        if text_n > best_n:
            best_n = text_n
            best_html = fragment
            best_label = label

    if best_n < 80:
        return html, ""
    if full_n > 0 and best_n < int(_CONTENT_ROOT_MIN_RATIO * full_n):
        # Hub/TOC roots (e.g. NHS #maincontent) — keep full document instead.
        return html, ""
    return best_html, best_label


class _ContentRootFinder(HTMLParser):
    """Capture inner HTML of every preferred content root."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.candidates: list[tuple[str, str]] = []
        self._capturing = False
        self._depth = 0
        self._parts: list[str] = []
        self._label = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag_l = tag.lower()
        amap = _attr_map(attrs)
        if self._capturing:
            self._depth += 1
            self._parts.append(self.get_starttag_text() or f"<{tag}>")
            return
        if _is_content_root(tag_l, amap):
            self._capturing = True
            self._depth = 1
            self._parts = []
            self._label = _root_label(tag_l, amap)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self._capturing:
            self._parts.append(self.get_starttag_text() or f"<{tag} />")

    def handle_endtag(self, tag: str) -> None:
        if not self._capturing:
            return
        self._depth -= 1
        if self._depth <= 0:
            self._capturing = False
            self.candidates.append((self._label, "".join(self._parts)))
            self._parts = []
            return
        self._parts.append(f"</{tag}>")

    def handle_data(self, data: str) -> None:
        if self._capturing:
            self._parts.append(data)

    def handle_entityref(self, name: str) -> None:
        if self._capturing:
            self._parts.append(f"&{name};")

    def handle_charref(self, name: str) -> None:
        if self._capturing:
            self._parts.append(f"&#{name};")


class _HtmlTextExtractor(HTMLParser):
    """Convert HTML to text with paragraph breaks; optional wiki chrome/section drops."""

    def __init__(self, *, wiki_mode: bool) -> None:
        super().__init__(convert_charrefs=True)
        self.wiki_mode = wiki_mode
        self._chunks: list[str] = []
        self._skip_depth = 0
        self._chrome_depth = 0
        self._section_skip = False
        self._in_heading = False
        self._heading_parts: list[str] = []
        self._pending_break = False

    def _emit_break(self) -> None:
        self._pending_break = True

    def _write(self, text: str) -> None:
        if not text:
            return
        if self._pending_break and self._chunks:
            self._chunks.append("\n\n")
        self._pending_break = False
        self._chunks.append(text)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag_l = tag.lower()
        amap = _attr_map(attrs)
        if self._skip_depth:
            self._skip_depth += 1
            return
        if tag_l in _SKIP_TAGS:
            self._skip_depth = 1
            return
        if self.wiki_mode and self._chrome_depth == 0 and _is_wiki_chrome(amap):
            self._chrome_depth = 1
            return
        if self._chrome_depth:
            self._chrome_depth += 1
            return
        if self._section_skip and tag_l not in _HEADING_TAGS:
            return
        if tag_l in _HEADING_TAGS:
            self._in_heading = True
            self._heading_parts = []
            self._emit_break()
            return
        if tag_l == "br":
            self._emit_break()
            return
        if tag_l in _BLOCK_TAGS:
            self._emit_break()

    def handle_endtag(self, tag: str) -> None:
        tag_l = tag.lower()
        if self._skip_depth:
            self._skip_depth -= 1
            return
        if self._chrome_depth:
            self._chrome_depth -= 1
            return
        if tag_l in _HEADING_TAGS and self._in_heading:
            heading = _WS_LINE_RE.sub(" ", "".join(self._heading_parts)).strip()
            self._in_heading = False
            self._heading_parts = []
            if self.wiki_mode and heading.lower().rstrip(":") in _DROP_SECTION_TITLES:
                self._section_skip = True
                return
            if tag_l in ("h1", "h2") and self.wiki_mode:
                self._section_skip = False
            if heading:
                self._write(heading)
            self._emit_break()
            return
        if self._section_skip:
            return
        if tag_l in _BLOCK_TAGS:
            self._emit_break()

    def handle_data(self, data: str) -> None:
        if self._skip_depth or self._chrome_depth:
            return
        if self._in_heading:
            self._heading_parts.append(data)
            return
        if self._section_skip:
            return
        text = _WS_LINE_RE.sub(" ", data)
        if text.strip():
            self._write(text)
        elif text and self._chunks and not self._pending_break:
            if not self._chunks[-1].endswith((" ", "\n")):
                self._chunks.append(" ")

    def text(self) -> str:
        joined = "".join(self._chunks)
        joined = _MULTI_NL_RE.sub("\n\n", joined)
        return joined.strip()


def html_to_text(html: str, *, wiki_mode: bool = False) -> str:
    """HTML → text with structure-preserving breaks."""
    extractor = _HtmlTextExtractor(wiki_mode=wiki_mode)
    try:
        extractor.feed(html)
        extractor.close()
    except Exception:
        rough = re.sub(r"<[^>]+>", " ", html)
        return _WS_LINE_RE.sub(" ", unescape(rough)).strip()
    return extractor.text()


def sanitize_zim_html(html: str, *, title: str = "", url: str = "") -> str | None:
    """Sanitize ZIM HTML. Return None to skip the article entirely."""
    if should_skip_mediawiki_title(title, url):
        return None

    wiki_mode = is_mediawiki_like(title=title, url=url)
    fragment, _root = _extract_content_root(html, wiki_mode=wiki_mode)
    text = html_to_text(fragment, wiki_mode=wiki_mode)
    if len(text) < 80:
        return None
    return text


def sanitize_debug_info(html: str, *, title: str = "", url: str = "") -> dict:
    """Spike helper: describe which gates/roots would apply."""
    wiki = is_mediawiki_like(title=title, url=url)
    skip = should_skip_mediawiki_title(title, url)
    fragment, root = _extract_content_root(html, wiki_mode=wiki)
    text = None if skip else html_to_text(fragment, wiki_mode=wiki)
    return {
        "mediawiki_like": wiki,
        "skip_title": skip,
        "content_root": root or "(document)",
        "raw_html_chars": len(html),
        "fragment_chars": len(fragment),
        "text_chars": 0 if text is None else len(text),
        "text": text,
    }

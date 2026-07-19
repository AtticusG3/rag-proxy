#!/usr/bin/env python3
"""Clean Literotica Wayback archive: stories to markdown, drop comments/feedback junk.

Reads html/ + manifest.json from a wayback-site-archive.py export directory.
Extracts story text from old (b-story-body) and new (panel article) HTML layouts,
merges multi-page stories when multiple page files exist for one slug, and writes
markdown under out-dir/stories/.

Usage:
  python clean_literotica_wayback.py --in-dir ~/literotica-wayback-trial-500
  python clean_literotica_wayback.py --in-dir ~/literotica-wayback-trial-500 --remove-html
  python clean_literotica_wayback.py --in-dir ~/literotica-wayback-trial-500 --dry-run
"""

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

JUNK_MARKERS = (
    "/comments",
    "/feedback",
)

NEW_LAYOUT_SKIP_LINES = (
    "Log In Sign Up",
    "Classic Literotica",
    "LITEROTICA WEBCAMS",
    "Swipe to see",
    "You can temporarily switch",
    "font size, font face",
    "Try the free",
    "Live Webcams",
)


@dataclass
class StoryMeta:
    slug: str
    path: str
    title: str = ""
    author: str = ""
    category: str = ""
    rating: str = ""
    rating_source: str = ""
    comments: int | None = None
    views: int | None = None
    favorites: int | None = None
    live_url: str = ""
    wayback_url: str = ""
    manifest_pages: list[int] = field(default_factory=list)
    source_files: list[Path] = field(default_factory=list)
    detected_pages: list[int] = field(default_factory=list)
    missing_pages: list[int] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def load_manifest(in_dir: Path) -> list[dict[str, str]]:
    manifest_path = in_dir / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"Missing manifest: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def is_junk_manifest_path(path: str) -> bool:
    return any(marker in path for marker in JUNK_MARKERS)


def manifest_page(original: str) -> int:
    match = re.search(r"[?&]page=(\d+)", original)
    return int(match.group(1)) if match else 1


def slug_from_manifest_path(path: str) -> str:
    return path.rsplit("/", 1)[-1]


def decode_text(text: str) -> str:
    text = html.unescape(text)
    text = text.replace("\xa0", " ")
    return text


def strip_tags(fragment: str) -> str:
    fragment = re.sub(r"<br\s*/?>", "\n", fragment, flags=re.I)
    fragment = re.sub(r"</p>\s*<p[^>]*>", "\n\n", fragment, flags=re.I)
    fragment = re.sub(r"<[^>]+>", "", fragment)
    return decode_text(fragment)


def clean_paragraphs(text: str, *, skip_lines: tuple[str, ...] = ()) -> str:
    lines: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            if lines and lines[-1] != "":
                lines.append("")
            continue
        if any(marker in line for marker in skip_lines):
            continue
        lines.append(line)
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def extract_rating(text: str) -> str:
    """Return story average rating (0.00-5.00) when embedded in static HTML."""
    for pattern in (
        r'property="ratingValue"[^>]*>\s*([0-5]\.[0-9]{2})\s*</span>',
        r'"@type"\s*:\s*"AggregateRating"[\s\S]{0,400}?"ratingValue"\s*:\s*([0-5]\.[0-9]{2})',
        r'itemprop="ratingValue"[^>]*content="([0-5]\.[0-9]{2})"',
    ):
        match = re.search(pattern, text, re.I)
        if match:
            return match.group(1)
    return ""


def extract_story_stats(text: str) -> tuple[int, int, int] | None:
    """Return (comments, views, favorites) from old or new layout HTML."""
    if "b-story-body" in text:
        block = re.search(
            r'<div class="b-story-stats-block">(.*?)<div class="b-sidebar">',
            text,
            re.I | re.S,
        )
        if not block:
            return None
        chunk = decode_text(re.sub(r"<[^>]+>", " ", block.group(1)))
        chunk = re.sub(r"\s+", " ", chunk).strip()
        match = re.search(
            r"(\d[\d,]*)\s*comments?\s*/\s*(\d[\d,]*)\s*views?\s*/\s*(\d[\d,]*)\s*favorites?",
            chunk,
            re.I,
        )
        if not match:
            return None
        return tuple(int(value.replace(",", "")) for value in match.groups())  # type: ignore[return-value]

    comments = views = favorites = None
    for match in re.finditer(
        r'"@type"\s*:\s*"InteractionCounter"[\s\S]*?"userInteractionCount"\s*:\s*(\d+)',
        text,
        re.I,
    ):
        blob = match.group(0)
        count = int(match.group(1))
        if "ReadAction" in blob:
            views = count
        elif "CommentAction" in blob:
            comments = count
        elif "LikeAction" in blob or "Favorites" in blob:
            favorites = count

    if views is None:
        for prop, value in re.findall(
            r'property="(commentCount|viewCount|favoriteCount)"[^>]*>\s*([0-9,]+)',
            text,
            re.I,
        ):
            count = int(value.replace(",", ""))
            if prop.lower() == "commentcount":
                comments = count
            elif prop.lower() == "viewcount":
                views = count
            elif prop.lower() == "favoritecount":
                favorites = count

    if views is None:
        return None
    return (comments or 0, views, favorites or 0)


# favorites-per-10k-views at the high end of the trial corpus maps to 5.00
_FAVORITES_PER_10K_AT_MAX = 6.0


def proxy_rating_from_engagement(views: int, favorites: int) -> str:
    """Derive a 1.00-5.00 score from favorites/views when no aggregate rating exists."""
    if views <= 0:
        return "1.00"
    per_10k = favorites * 10_000 / views
    score = 1.0 + min(4.0, per_10k / _FAVORITES_PER_10K_AT_MAX * 4.0)
    return f"{score:.2f}"


def apply_rating_and_stats(meta: StoryMeta, text: str) -> None:
    stats = extract_story_stats(text)
    if stats:
        comments, views, favorites = stats
        if meta.comments is None:
            meta.comments = comments
        if meta.views is None:
            meta.views = views
        if meta.favorites is None:
            meta.favorites = favorites

    if meta.rating:
        if not meta.rating_source:
            meta.rating_source = "aggregate"
        return

    embedded = extract_rating(text)
    if embedded:
        meta.rating = embedded
        meta.rating_source = "aggregate"
        return

    if stats and stats[1] > 0:
        meta.rating = proxy_rating_from_engagement(stats[1], stats[2])
        meta.rating_source = "favorites_ratio"


def extract_old_layout(text: str) -> tuple[str, str, str, str]:
    title = ""
    author = ""
    category = ""
    body = ""

    title_match = re.search(r"<title>([^<]+)</title>", text, re.I)
    if title_match:
        title = decode_text(title_match.group(1))
        title = re.sub(r"\s*-\s*[^-]+-\s*Literotica\.com\s*$", "", title, flags=re.I).strip()

    h1_match = re.search(r"<h1[^>]*>(.*?)</h1>", text, re.S | re.I)
    if h1_match:
        title = decode_text(re.sub(r"<[^>]+>", "", h1_match.group(1)).strip()) or title

    author_match = re.search(
        r'class="b-story-user"[^>]*>\s*<a[^>]*>([^<]+)</a>',
        text,
        re.I,
    )
    if author_match:
        author = decode_text(author_match.group(1)).strip()

    crumb_match = re.search(
        r'b-breadcrumbs.*?category=(\d+)">([^<]+)</a>',
        text,
        re.S | re.I,
    )
    if crumb_match:
        category = decode_text(crumb_match.group(2)).strip()

    body_match = re.search(
        r'<div class="b-story-body">(.*?)</div>\s*<div class="b-story-stats-block">',
        text,
        re.S | re.I,
    )
    if body_match:
        body = clean_paragraphs(strip_tags(body_match.group(1)))

    return title, author, category, body


def _paragraphs_from_html(fragment: str, *, skip_lines: tuple[str, ...] = ()) -> str:
    parts: list[str] = []
    for match in re.finditer(r"<p[^>]*>(.*?)</p>", fragment, re.S | re.I):
        plain = strip_tags(match.group(1)).strip()
        if not plain or len(plain) < 20:
            continue
        if any(marker in plain for marker in skip_lines):
            continue
        parts.append(plain)
    return "\n\n".join(parts)


def _slice_after_h1(text: str) -> str:
    h1 = re.search(r"<h1[^>]*>.*?</h1>", text, re.S | re.I)
    if not h1:
        return text
    return text[h1.end() :]


def _trim_new_layout_tail(fragment: str) -> str:
    for marker in (
        "Recent Comments",
        "Share this story",
        "Report Story",
        "Public Beta",
        "Story Tags",
        "More stories by",
        "Log In Sign Up",
    ):
        idx = fragment.find(marker)
        if idx > 0:
            fragment = fragment[:idx]
    return fragment


def extract_new_layout(text: str) -> tuple[str, str, str, str]:
    title = ""
    author = ""
    category = ""
    body = ""

    title_match = re.search(r"<title[^>]*>([^<]+)</title>", text, re.I)
    if title_match:
        raw_title = decode_text(title_match.group(1))
        raw_title = re.sub(r"\s*-\s*Literotica\.com\s*$", "", raw_title, flags=re.I)
        page_prefix = re.match(r"Page\s+(\d+)\s*-\s*(.+)", raw_title, re.I)
        if page_prefix:
            title = page_prefix.group(2).strip()
            title = re.sub(r"\s*-\s*[^-]+$", "", title).strip()
        else:
            title = re.sub(r"\s*-\s*[^-]+$", "", raw_title).strip()

    h1_match = re.search(r"<h1[^>]*>(.*?)</h1>", text, re.S | re.I)
    if h1_match:
        title = decode_text(re.sub(r"<[^>]+>", "", h1_match.group(1)).strip()) or title

    author_match = re.search(
        r'class="[^"]*\bauthor[^"]*"[^>]*>.*?<a[^>]*>([^<]+)</a>',
        text,
        re.S | re.I,
    )
    if not author_match:
        author_match = re.search(
            r'class="[^"]*\busername[^"]*"[^>]*>\s*<a[^>]*>([^<]+)</a>',
            text,
            re.S | re.I,
        )
    if author_match:
        author = decode_text(author_match.group(1)).strip()

    category_match = re.search(
        r'<a[^>]+href="[^"]*/categories/[^"]*"[^>]*>([^<]+)</a>',
        text,
        re.I,
    )
    if category_match:
        category = decode_text(category_match.group(1)).strip()

    panel_match = re.search(
        r'<div[^>]*class="[^"]*\bpanel\b[^"]*\barticle\b[^"]*"[^>]*>',
        text,
        re.I,
    )
    if panel_match:
        fragment = _trim_new_layout_tail(text[panel_match.end() :])
        body = _paragraphs_from_html(fragment, skip_lines=NEW_LAYOUT_SKIP_LINES)
    if not body:
        fragment = _trim_new_layout_tail(_slice_after_h1(text))
        body = _paragraphs_from_html(fragment, skip_lines=NEW_LAYOUT_SKIP_LINES)

    return title, author, category, body


def extract_story_html(text: str) -> tuple[str, str, str, str]:
    if "b-story-body" in text:
        return extract_old_layout(text)
    return extract_new_layout(text)


def detect_page_number(text: str, *, manifest_page_no: int) -> int:
    title_match = re.search(r"<title[^>]*>([^<]+)</title>", text, re.I)
    if title_match:
        page_match = re.match(
            r"Page\s+(\d+)\s*-",
            decode_text(title_match.group(1)),
            re.I,
        )
        if page_match:
            return int(page_match.group(1))

    active = re.search(r'b-pager-active[^>]*>\s*(\d+)\s*<', text, re.I)
    if active:
        return int(active.group(1))

    return manifest_page_no


def detect_total_pages(text: str) -> int | None:
    match = re.search(r"(\d+)\s*Pages?:", text, re.I)
    if match:
        return int(match.group(1))
    pages = [int(n) for n in re.findall(r'b-pager[^>]*>\s*(\d+)\s*<', text, re.I)]
    if pages:
        return max(pages)
    return None


def story_source_paths(html_root: Path, slug: str) -> list[Path]:
    paths: list[Path] = []
    page_html = html_root / "s" / f"{slug}.page.html"
    plain = html_root / "s" / slug
    if page_html.is_file():
        paths.append(page_html)
    if plain.is_file():
        paths.append(plain)
    return paths


def build_story_index(manifest: list[dict[str, str]], html_root: Path) -> dict[str, StoryMeta]:
    index: dict[str, StoryMeta] = {}
    for entry in manifest:
        path = entry["path"]
        if is_junk_manifest_path(path):
            continue
        slug = slug_from_manifest_path(path)
        meta = index.get(slug)
        if meta is None:
            meta = StoryMeta(
                slug=slug,
                path=path,
                live_url=entry.get("live", ""),
                wayback_url=entry.get("wayback", ""),
            )
            index[slug] = meta
        page_no = manifest_page(entry.get("original", ""))
        if page_no not in meta.manifest_pages:
            meta.manifest_pages.append(page_no)

    for slug, meta in index.items():
        meta.manifest_pages.sort()
        if meta.manifest_pages and meta.manifest_pages[0] > 1:
            missing = list(range(1, meta.manifest_pages[0]))
            meta.warnings.append(f"manifest starts at page {meta.manifest_pages[0]}; missing {missing}")
        meta.source_files = story_source_paths(html_root, slug)
        if not meta.source_files:
            meta.warnings.append("no story html on disk")
    return index


def merge_story_parts(meta: StoryMeta) -> tuple[str, list[int], list[int]]:
    parts: list[tuple[int, str, str, str, str]] = []
    total_pages: int | None = None

    for source in meta.source_files:
        text = source.read_text(encoding="utf-8", errors="replace")
        manifest_hint = meta.manifest_pages[0] if len(meta.manifest_pages) == 1 else 1
        page_no = detect_page_number(text, manifest_page_no=manifest_hint)
        title, author, category, body = extract_story_html(text)
        if not body:
            meta.warnings.append(f"empty body in {source.name}")
            continue
        if not meta.title and title:
            meta.title = title
        if not meta.author and author:
            meta.author = author
        if not meta.category and category:
            meta.category = category
        apply_rating_and_stats(meta, text)
        total_pages = detect_total_pages(text) or total_pages
        parts.append((page_no, title, author, category, body))

    if not parts:
        return "", [], []

    parts.sort(key=lambda item: item[0])
    detected_pages = [page for page, *_ in parts]
    missing_pages: list[int] = []
    if total_pages and total_pages > 1:
        have = set(detected_pages)
        missing_pages = [n for n in range(1, total_pages + 1) if n not in have]
        if missing_pages:
            meta.warnings.append(f"missing page(s): {missing_pages}")

    if len(parts) == 1:
        merged_body = parts[0][4]
    else:
        merged_body = "\n\n".join(part[4] for part in parts)

    return merged_body, detected_pages, missing_pages


def to_markdown(meta: StoryMeta, body: str) -> str:
    lines = ["---"]
    lines.append(f"title: {json.dumps(meta.title or meta.slug)}")
    if meta.author:
        lines.append(f"author: {json.dumps(meta.author)}")
    if meta.category:
        lines.append(f"category: {json.dumps(meta.category)}")
    if meta.rating:
        lines.append(f"rating: {meta.rating}")
    if meta.rating_source:
        lines.append(f"rating_source: {json.dumps(meta.rating_source)}")
    if meta.views is not None:
        lines.append(f"views: {meta.views}")
    if meta.favorites is not None:
        lines.append(f"favorites: {meta.favorites}")
    if meta.comments is not None:
        lines.append(f"comments: {meta.comments}")
    if meta.views and meta.favorites is not None and meta.views > 0:
        rate = meta.favorites / meta.views
        lines.append(f"favorite_rate: {rate:.6f}")
    lines.append(f"slug: {json.dumps(meta.slug)}")
    if meta.live_url:
        lines.append(f"source: {json.dumps(meta.live_url)}")
    if meta.wayback_url:
        lines.append(f"wayback: {json.dumps(meta.wayback_url)}")
    if meta.detected_pages:
        lines.append(f"pages: {meta.detected_pages}")
    if meta.missing_pages:
        lines.append(f"missing_pages: {meta.missing_pages}")
    lines.append("---")
    lines.append("")
    lines.append(f"# {meta.title or meta.slug}")
    if meta.author:
        lines.append("")
        lines.append(f"*by {meta.author}*")
    lines.append("")
    lines.append(body)
    lines.append("")
    return "\n".join(lines)


def junk_paths(html_root: Path) -> list[Path]:
    junk: list[Path] = []
    if not html_root.is_dir():
        return junk
    for path in sorted(html_root.rglob("*")):
        rel = path.relative_to(html_root).as_posix()
        if "/comments" in rel or rel.endswith("/comments"):
            junk.append(path)
        elif path.name == "feedback":
            junk.append(path)
    return junk


def remove_junk_html(html_root: Path, *, dry_run: bool) -> int:
    removed = 0
    for path in junk_paths(html_root):
        if dry_run:
            removed += 1
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)
        removed += 1

    if not dry_run:
        for path in sorted(html_root.rglob("*"), reverse=True):
            if path.is_dir() and not any(path.iterdir()):
                path.rmdir()
    return removed


def remove_converted_html(html_root: Path, converted_sources: set[Path], *, dry_run: bool) -> int:
    removed = 0
    for path in sorted(converted_sources):
        if not path.is_file():
            continue
        if dry_run:
            removed += 1
            continue
        path.unlink(missing_ok=True)
        removed += 1
    if not dry_run:
        for path in sorted(html_root.rglob("*"), reverse=True):
            if path.is_dir() and not any(path.iterdir()):
                path.rmdir()
    return removed


def cmd_clean(
    in_dir: Path,
    out_dir: Path,
    *,
    dry_run: bool,
    remove_html: bool,
) -> int:
    manifest = load_manifest(in_dir)
    html_root = in_dir / "html"
    stories_out = out_dir / "stories"
    if not dry_run:
        stories_out.mkdir(parents=True, exist_ok=True)

    index = build_story_index(manifest, html_root)
    converted_sources: set[Path] = set()
    report = {
        "in_dir": str(in_dir),
        "out_dir": str(out_dir),
        "dry_run": dry_run,
        "stories_total": len(index),
        "written": 0,
        "skipped": 0,
        "warnings": 0,
        "ratings_found": 0,
        "proxy_ratings": 0,
        "junk_paths": len(junk_paths(html_root)),
        "items": [],
    }

    for slug in sorted(index):
        meta = index[slug]
        body, detected_pages, missing_pages = merge_story_parts(meta)
        meta.detected_pages = detected_pages
        meta.missing_pages = missing_pages

        item = {
            "slug": slug,
            "title": meta.title,
            "rating": meta.rating or None,
            "rating_source": meta.rating_source or None,
            "views": meta.views,
            "favorites": meta.favorites,
            "comments": meta.comments,
            "sources": [str(p.relative_to(in_dir)) for p in meta.source_files],
            "pages": detected_pages,
            "missing_pages": missing_pages,
            "warnings": meta.warnings,
        }
        report["items"].append(item)

        if not body:
            report["skipped"] += 1
            continue

        if len(body) < 200:
            meta.warnings.append(f"short body ({len(body)} chars)")

        if meta.warnings:
            report["warnings"] += 1

        if meta.rating:
            if meta.rating_source == "aggregate":
                report["ratings_found"] += 1
            elif meta.rating_source == "favorites_ratio":
                report["proxy_ratings"] += 1

        dest = stories_out / f"{slug}.md"
        markdown = to_markdown(meta, body)
        if dry_run:
            print(f"[dry-run] would write {dest} ({len(body)} chars, pages={detected_pages})")
        else:
            dest.write_text(markdown, encoding="utf-8")
        converted_sources.update(meta.source_files)
        report["written"] += 1

    junk_count = len(junk_paths(html_root))
    if remove_html:
        junk_removed = remove_junk_html(html_root, dry_run=dry_run)
        html_removed = remove_converted_html(html_root, converted_sources, dry_run=dry_run)
        report["junk_removed"] = junk_removed
        report["html_removed"] = html_removed
    else:
        report["junk_removed"] = 0
        report["html_removed"] = 0

    report_path = out_dir / "cleanup-report.json"
    if dry_run:
        print(
            f"[dry-run] stories={report['stories_total']} write={report['written']} "
            f"skip={report['skipped']} warn={report['warnings']} junk={junk_count}"
        )
    else:
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        f"Done: wrote {report['written']} markdown files, skipped {report['skipped']}, "
        f"warnings {report['warnings']}, ratings {report['ratings_found']} "
        f"(+{report['proxy_ratings']} proxy), junk paths {junk_count}"
    )
    if not dry_run:
        print(f"Report: {report_path}")
        print(f"Stories: {stories_out}")
    return 0 if report["written"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean Literotica Wayback HTML archive to markdown")
    parser.add_argument(
        "--in-dir",
        type=Path,
        default=Path.home() / "literotica-wayback-trial-500",
        help="Archive directory (manifest.json + html/)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: IN_DIR/clean)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without writing files",
    )
    parser.add_argument(
        "--remove-html",
        action="store_true",
        help="Delete comments/feedback junk and converted story HTML from html/",
    )
    args = parser.parse_args()
    out_dir = args.out_dir or (args.in_dir / "clean")
    return cmd_clean(args.in_dir, out_dir, dry_run=args.dry_run, remove_html=args.remove_html)


if __name__ == "__main__":
    raise SystemExit(main())

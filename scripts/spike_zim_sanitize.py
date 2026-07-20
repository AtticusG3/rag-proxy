#!/usr/bin/env python3
"""Compare current ZIM strip_html vs gated sanitize on real archives.

Example:
  python scripts/spike_zim_sanitize.py --zim example_zim/devdocs_en_playwright_2026-07.zim
  python scripts/spike_zim_sanitize.py --zim example_zim/nhs.uk_en_medicines_2025-12.zim --limit 3 --out /tmp/zim_spike
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ingest.zim_reader import strip_html  # noqa: E402
from ingest.zim_sanitize import sanitize_debug_info  # noqa: E402

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def _safe(name: str, limit: int = 80) -> str:
    cleaned = _SAFE_NAME.sub("_", name).strip("_")
    return (cleaned or "article")[:limit]


def _iter_html_entries(zim_path: str, *, limit: int, url_filter: str, title_filter: str):
    from libzim.reader import Archive

    archive = Archive(zim_path)
    entry_count = int(getattr(archive, "all_entry_count", 0) or 0)
    yielded = 0
    for index in range(entry_count):
        if limit > 0 and yielded >= limit:
            break
        try:
            entry = archive._get_entry_by_id(index)
        except Exception:
            continue
        if entry.is_redirect:
            continue
        title = entry.title or entry.path
        url = entry.path
        if url_filter and url_filter not in url:
            continue
        if title_filter and title_filter.lower() not in title.lower():
            continue
        try:
            item = entry.get_item()
            mimetype = (getattr(item, "mimetype", "") or "").lower()
            raw = bytes(item.content)
        except Exception:
            continue
        if mimetype and not any(
            token in mimetype for token in ("text/html", "text/plain", "application/xhtml")
        ):
            continue
        html = raw.decode("utf-8", errors="replace")
        if "html" not in mimetype and "<" not in html[:200]:
            continue
        yield title, url, html
        yielded += 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zim", required=True, help="Path to a .zim file")
    parser.add_argument("--limit", type=int, default=3, help="Max articles to dump (0=all matching)")
    parser.add_argument("--url", default="", help="Substring filter on entry path")
    parser.add_argument("--title", default="", help="Substring filter on entry title")
    parser.add_argument(
        "--out",
        default="",
        help="Directory for before/after dumps (default: ./zim_spike_out/<zim-stem>)",
    )
    args = parser.parse_args()

    zim_path = Path(args.zim)
    if not zim_path.is_file():
        print(f"error: zim not found: {zim_path}", file=sys.stderr)
        return 1

    out_dir = Path(args.out) if args.out else Path("zim_spike_out") / zim_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"zim: {zim_path}")
    print(f"out: {out_dir.resolve()}")
    print("-" * 72)

    count = 0
    for title, url, html in _iter_html_entries(
        str(zim_path),
        limit=args.limit,
        url_filter=args.url,
        title_filter=args.title,
    ):
        count += 1
        legacy = strip_html(html)
        info = sanitize_debug_info(html, title=title, url=url)
        sanitized = info["text"]

        stem = f"{count:02d}_{_safe(title) or _safe(url)}"
        (out_dir / f"{stem}.raw.html").write_text(html[:200_000], encoding="utf-8")
        (out_dir / f"{stem}.legacy.txt").write_text(legacy, encoding="utf-8")
        if sanitized is None:
            (out_dir / f"{stem}.sanitized.txt").write_text("[SKIPPED]\n", encoding="utf-8")
        else:
            (out_dir / f"{stem}.sanitized.txt").write_text(sanitized, encoding="utf-8")

        legacy_n = len(legacy)
        new_n = 0 if sanitized is None else len(sanitized)
        delta = "SKIP" if sanitized is None else f"{new_n - legacy_n:+d}"
        print(f"[{count}] {title!r}")
        print(f"     url={url}")
        print(
            f"     raw={info['raw_html_chars']} root={info['content_root']} "
            f"mw={info['mediawiki_like']} skip_title={info['skip_title']}"
        )
        print(f"     legacy_chars={legacy_n} sanitized_chars={new_n} delta={delta}")
        preview = (sanitized or legacy)[:180].replace("\n", " ")
        print(f"     preview={preview!r}")
        print()

    if count == 0:
        print("no matching HTML entries")
        return 2
    print(f"done: {count} article(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

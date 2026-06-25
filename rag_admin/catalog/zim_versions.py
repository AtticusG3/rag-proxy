"""Kiwix ZIM filename parsing, dedupe, and latest-version resolution."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Iterable

from rag_admin.catalog.listing_parser import CatalogItem

# Examples:
#   devdocs_en_python_2026-05.zim
#   wikipedia_en_all_maxi_2026-04.zim
#   package_latest.zim / package_previous.zim (legacy labels)
ZIM_STAMP_RE = re.compile(
    r"^(?P<base>.+)_(?P<stamp>(?:\d{4}-\d{2}|latest|previous))\.zim$",
    re.IGNORECASE,
)

KIWIX_SOURCES = frozenset({"dotsrc", "kiwix"})


@dataclass(frozen=True)
class ZimStamp:
    raw: str
    sort_key: tuple[int, int]

    @property
    def is_dated(self) -> bool:
        return bool(re.fullmatch(r"\d{4}-\d{2}", self.raw))


def parse_zim_stamp(name: str) -> tuple[str, ZimStamp] | None:
    """Return (package_key, stamp) for a .zim filename."""
    match = ZIM_STAMP_RE.match(name)
    if not match:
        return None
    stamp_raw = match.group("stamp").lower()
    if stamp_raw == "latest":
        sort_key = (9999, 12)
    elif stamp_raw == "previous":
        sort_key = (9998, 12)
    else:
        year_s, month_s = stamp_raw.split("-", 1)
        sort_key = (int(year_s), int(month_s))
    return match.group("base"), ZimStamp(raw=stamp_raw, sort_key=sort_key)


def format_version_label(stamp: str) -> str:
    if re.fullmatch(r"\d{4}-\d{2}", stamp):
        year_s, month_s = stamp.split("-", 1)
        months = (
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        )
        month_idx = max(1, min(12, int(month_s))) - 1
        return f"{months[month_idx]} {year_s}"
    return stamp.title()


def pick_latest_zim(items: Iterable[CatalogItem], package_key: str) -> CatalogItem | None:
    matches: list[tuple[tuple[int, int], CatalogItem]] = []
    for item in items:
        if item.is_directory or not item.name.lower().endswith(".zim"):
            continue
        parsed = parse_zim_stamp(item.name)
        if parsed is None or parsed[0] != package_key:
            continue
        matches.append((parsed[1].sort_key, item))
    if not matches:
        return None
    matches.sort(key=lambda row: row[0])
    return matches[-1][1]


def dedupe_kiwix_items(items: list[CatalogItem]) -> list[CatalogItem]:
    """Keep one row per logical ZIM package (newest YYYY-MM or latest label)."""
    passthrough: list[CatalogItem] = []
    groups: dict[str, list[tuple[tuple[int, int], CatalogItem, ZimStamp]]] = {}

    for item in items:
        if item.is_directory or not item.name.lower().endswith(".zim"):
            passthrough.append(item)
            continue
        parsed = parse_zim_stamp(item.name)
        if parsed is None:
            passthrough.append(item)
            continue
        package_key, stamp = parsed
        groups.setdefault(package_key, []).append((stamp.sort_key, item, stamp))

    deduped: list[CatalogItem] = []
    for package_key, rows in groups.items():
        rows.sort(key=lambda row: row[0])
        hidden = len(rows) - 1
        _sort, winner, stamp = rows[-1]
        deduped.append(
            replace(
                winner,
                package_key=package_key,
                version_stamp=stamp.raw,
                hidden_older_versions=hidden,
            )
        )

    combined = passthrough + deduped
    return sorted(combined, key=lambda i: (not i.is_directory, i.name.lower()))


def resolve_latest_item(
    items: list[CatalogItem],
    package_key: str,
) -> CatalogItem | None:
    """Find newest catalog row for a package key (deduped or raw listing)."""
    latest = pick_latest_zim(items, package_key)
    if latest is not None:
        return latest
    for item in items:
        if item.package_key == package_key:
            return item
    return None

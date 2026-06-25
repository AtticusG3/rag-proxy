"""Tests for Kiwix ZIM version deduplication."""

from __future__ import annotations

from rag_admin.catalog.listing_parser import CatalogItem
from rag_admin.catalog.providers import browse_source
from rag_admin.catalog.zim_versions import (
    dedupe_kiwix_items,
    parse_zim_stamp,
    pick_latest_zim,
)


def _zim(name: str, url: str) -> CatalogItem:
    return CatalogItem(
        name=name,
        href=name,
        url=url,
        is_directory=False,
        size_bytes=1000,
        modified="2026-05-01",
        subscribable=True,
    )


def test_parse_zim_stamp() -> None:
    parsed = parse_zim_stamp("devdocs_en_angular_2026-05.zim")
    assert parsed is not None
    assert parsed[0] == "devdocs_en_angular"
    assert parsed[1].raw == "2026-05"


def test_dedupe_keeps_newest_month() -> None:
    items = [
        _zim("devdocs_en_angular_2026-02.zim", "https://x/a1.zim"),
        _zim("devdocs_en_angular_2026-05.zim", "https://x/a2.zim"),
        _zim("devdocs_en_bash_2026-01.zim", "https://x/b1.zim"),
        _zim("devdocs_en_bash_2026-04.zim", "https://x/b2.zim"),
    ]
    out = dedupe_kiwix_items(items)
    zims = [i for i in out if i.name.endswith(".zim")]
    assert len(zims) == 2
    assert any(i.name == "devdocs_en_angular_2026-05.zim" for i in zims)
    winner = next(i for i in zims if i.package_key == "devdocs_en_angular")
    assert winner.hidden_older_versions == 1
    assert winner.version_stamp == "2026-05"


def test_pick_latest_zim() -> None:
    items = dedupe_kiwix_items(
        [
            _zim("wikipedia_en_all_maxi_2026-01.zim", "https://x/w1.zim"),
            _zim("wikipedia_en_all_maxi_2026-04.zim", "https://x/w2.zim"),
        ]
    )
    latest = pick_latest_zim(items, "wikipedia_en_all_maxi")
    assert latest is not None
    assert latest.name.endswith("2026-04.zim")


def test_browse_devdocs_deduped() -> None:
    result = browse_source("kiwix", "devdocs")
    assert result["error"] is None
    zims = [i for i in result["items"] if i.name.endswith(".zim")]
    assert len(zims) < 454
    assert result.get("dedupe_hidden", 0) > 0
    keys = [i.package_key for i in zims if i.package_key]
    assert len(keys) == len(set(keys))

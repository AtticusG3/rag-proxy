"""Tests for mirror URL guards in catalog browse."""

from __future__ import annotations

import pytest

from rag_admin.catalog.listing_parser import is_directory_listing, is_internal_href, parse_directory_listing
from rag_admin.catalog.providers import browse_source


def test_is_internal_href_rejects_absolute_urls() -> None:
    assert is_internal_href("wikipedia/")
    assert not is_internal_href("https://browse.library.kiwix.org/")
    assert not is_internal_href("//cdn.example.com/zim/")


def test_hub_page_is_not_directory_listing() -> None:
    html = "<html><body><a href='https://browse.library.kiwix.org/'>Browse</a></body></html>"
    assert not is_directory_listing(html)
    assert parse_directory_listing(html, "https://hub.kiwix.org/downloads/") == []


def test_browse_kiwix_official_uses_lb_mirror() -> None:
    result = browse_source("kiwix", "")
    assert result["error"] is None
    assert len(result["items"]) >= 10
    assert any(i.name == "wikipedia" for i in result["items"])


def test_browse_kiwix_rejects_external_path() -> None:
    result = browse_source("kiwix", "https://browse.library.kiwix.org/")
    assert result["items"] == []
    assert result["error"] is not None

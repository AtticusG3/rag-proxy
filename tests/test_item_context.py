"""Tests for catalog item context helpers."""

from __future__ import annotations

from rag_admin.catalog.item_context import describe_item


def test_describe_zim_file() -> None:
    ctx = describe_item(
        "kiwix",
        name="devdocs_en_python_2026-05.zim",
        href="devdocs_en_python_2026-05.zim",
        path="devdocs",
        is_directory=False,
        subscribable=True,
        external_url=None,
        modified="2026-05-01",
        package_key="devdocs_en_python",
        version_stamp="2026-05",
        hidden_older_versions=1,
    )
    assert ctx["kind"] == "zim"
    assert ctx["title"] == "Python"
    assert "May 2026" in ctx["subtitle"]
    assert "older dated" in ctx["hint"]


def test_describe_folder_hint() -> None:
    ctx = describe_item(
        "dotsrc",
        name="stack_exchange",
        href="stack_exchange/",
        path="",
        is_directory=True,
        subscribable=False,
        external_url=None,
        modified="",
    )
    assert ctx["kind"] == "folder"
    assert "Q&A" in ctx["hint"]

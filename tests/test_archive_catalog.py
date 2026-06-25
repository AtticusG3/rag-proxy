"""Tests for Internet Archive catalog browse."""

from __future__ import annotations

from rag_admin.catalog.archive import _browse_item, _parse_path, browse_archive

IA_ITEM_JSON = """
{
  "files": [
    {"name": "example.zim", "format": "ZIM", "size": "1048576", "mtime": "1700000000"},
    {"name": "notes.txt", "format": "Text", "size": "2048", "mtime": "1700000001"},
    {"name": "cover.epub", "format": "EPUB", "size": "999", "mtime": "1700000002"},
    {"name": "meta.xml", "format": "Metadata", "size": "100", "mtime": "1700000003"}
  ]
}
"""


def test_parse_archive_paths() -> None:
    assert _parse_path("") == ("root", {})
    assert _parse_path("collection/opensource") == (
        "collection",
        {"collection": "opensource"},
    )
    assert _parse_path("collection/opensource/page/2") == (
        "collection",
        {"collection": "opensource", "page": "2"},
    )
    assert _parse_path("item/foo_bar") == ("item", {"identifier": "foo_bar"})


def test_browse_archive_root() -> None:
    result = browse_archive("")
    assert result["error"] is None
    assert len(result["items"]) >= 5
    assert any(i.href == "collection/opensource" for i in result["items"])


def test_browse_item_filters_ingestable(monkeypatch) -> None:
    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            import json

            return json.loads(IA_ITEM_JSON)

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def get(self, url: str):
            return FakeResponse()

    monkeypatch.setattr("rag_admin.catalog.archive._client", lambda: FakeClient())
    items = _browse_item("example")
    names = [i.name for i in items]
    assert "example.zim" in names
    assert "notes.txt" in names
    assert "cover.epub" not in names
    assert all(i.subscribable for i in items)

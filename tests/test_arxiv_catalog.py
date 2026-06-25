"""Tests for arXiv catalog browse."""

from __future__ import annotations

from rag_admin.catalog.arxiv import _arxiv_id_from_entry_id, _parse_path, browse_arxiv

ARXIV_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">
  <opensearch:totalResults>2</opensearch:totalResults>
  <entry>
    <id>http://arxiv.org/abs/2301.12345v1</id>
    <title>Sample Paper Title</title>
    <summary>Abstract text here.</summary>
    <published>2023-01-30T00:00:00Z</published>
    <updated>2023-01-31T00:00:00Z</updated>
  </entry>
</feed>
"""


def test_parse_arxiv_paths() -> None:
    assert _parse_path("") == ("root", {})
    assert _parse_path("cat/cs.AI") == ("category", {"category": "cs.AI"})
    assert _parse_path("cat/cs.AI/page/2") == (
        "category",
        {"category": "cs.AI", "page": "2"},
    )
    assert _parse_path("paper/2301.12345") == ("paper", {"paper_id": "2301.12345"})


def test_arxiv_id_from_entry() -> None:
    assert (
        _arxiv_id_from_entry_id("http://arxiv.org/abs/2301.12345v2") == "2301.12345"
    )


def test_browse_arxiv_root() -> None:
    result = browse_arxiv("")
    assert result["error"] is None
    assert any(i.href == "prefix/cs" for i in result["items"])
    assert any(i.href == "cat/stat.ML" for i in result["items"])


def test_query_papers_parses_atom(monkeypatch) -> None:
    from rag_admin.catalog import arxiv as arxiv_mod

    class FakeResponse:
        text = ARXIV_ATOM

        def raise_for_status(self) -> None:
            return None

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def get(self, url: str, params: dict):
            return FakeResponse()

    monkeypatch.setattr(arxiv_mod, "_arxiv_get", lambda params: FakeResponse())
    items = arxiv_mod._query_papers("cat:cs.AI", page=0, href_prefix="cat/cs.AI")
    assert len(items) == 1
    assert items[0].subscribable
    assert items[0].url.endswith("2301.12345.pdf")

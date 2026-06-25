"""Tests for Apache directory listing parser."""

from __future__ import annotations

import httpx

from rag_admin.catalog.listing_parser import parse_directory_listing

DOTSRC_SAMPLE = """\
<tbody><tr><td class="link"><a href="../">Parent directory/</a></td><td class="size">-</td><td class="date">-</td></tr>
<tr><td class="link"><a href="devdocs/" title="devdocs">devdocs/</a></td><td class="size">-</td><td class="date">2026-06-06 12:38:12</td></tr>
<tr><td class="link"><a href="wikipedia/" title="wikipedia">wikipedia/</a></td><td class="size">-</td><td class="date">2026-06-23 12:03:09</td></tr>
<tr><td class="link"><a href="README">README</a></td><td class="size">1.2 KiB</td><td class="date">2019-11-23 15:58:39</td></tr>
</tbody>"""


def test_parse_dotsrc_tbody_sample() -> None:
    items = parse_directory_listing(DOTSRC_SAMPLE, "https://mirrors.dotsrc.org/kiwix/zim/")
    names = [i.name for i in items]
    assert "devdocs" in names
    assert "wikipedia" in names
    assert "README" in names
    assert "Parent directory" not in names
    readme = next(i for i in items if i.name == "README")
    assert readme.size_bytes == int(1.2 * 1024)
    assert readme.subscribable is False


def test_zim_files_are_subscribable() -> None:
    from rag_admin.catalog.listing_parser import infer_subscribable

    assert infer_subscribable("devdocs_en_python_2026-05.zim", is_directory=False)
    assert not infer_subscribable("devdocs", is_directory=True)


def test_parse_live_dotsrc_root() -> None:
    response = httpx.get("https://mirrors.dotsrc.org/kiwix/zim/", timeout=60.0)
    response.raise_for_status()
    items = parse_directory_listing(response.text, "https://mirrors.dotsrc.org/kiwix/zim/")
    assert len(items) >= 10
    assert any(i.name == "wikipedia" and i.is_directory for i in items)

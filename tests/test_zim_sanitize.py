"""Tests for gated ZIM HTML sanitisation."""

from __future__ import annotations

from ingest.zim_sanitize import (
    html_to_text,
    is_mediawiki_like,
    sanitize_zim_html,
    should_skip_mediawiki_title,
)


def test_drops_script_and_style_bodies():
    html = """
    <html><head><style>.x{color:red}</style>
    <script>window.TRACK='secret-analytics'</script></head>
    <body><main><p>Real paragraph about catalysts used in industrial chemistry
    processes, with enough detail to clear the minimum article length gate.</p></main></body></html>
    """
    text = sanitize_zim_html(html, title="Catalysts", url="catalysts")
    assert text is not None
    assert "secret-analytics" not in text
    assert "color:red" not in text
    assert "Real paragraph about catalysts" in text


def test_preserves_paragraph_breaks():
    html = "<article><p>First block.</p><p>Second block.</p></article>"
    text = html_to_text(html)
    assert "First block." in text
    assert "Second block." in text
    assert "\n\n" in text


def test_devdocs_like_content_root_preferred():
    html = """
    <html><body>
    <nav>Skip nav chrome</nav>
    <div class="_content"><p>Playwright auto-waiting details go here with enough
    words so the article clears the minimum length gate for indexing.</p></div>
    </body></html>
    """
    text = sanitize_zim_html(html, title="Auto-waiting", url="actionability")
    assert text is not None
    assert "Playwright auto-waiting" in text
    assert "Skip nav chrome" not in text


def test_non_wiki_does_not_drop_references_heading():
    html = """
    <main>
      <p>Body prose about a library API with enough characters for the length gate
      when indexing non-wiki documentation ZIM archives from DevDocs.</p>
      <h2>References</h2>
      <p>Should remain for non-wiki ZIM sources like DevDocs.</p>
    </main>
    """
    text = sanitize_zim_html(html, title="API testing", url="api-testing")
    assert text is not None
    assert "Should remain" in text


def test_mediawiki_category_title_skipped():
    html = "<main><p>" + ("category listing " * 20) + "</p></main>"
    assert sanitize_zim_html(html, title="Category:Chemistry", url="A/Category:Chemistry") is None
    assert should_skip_mediawiki_title("Category:Chemistry", "A/Category:Chemistry")


def test_mediawiki_drops_references_section():
    html = """
    <div id="mw-content-text" class="mw-parser-output">
      <p>Useful article body about benzene rings in organic chemistry with enough
      text to pass the minimum length check used during ZIM ingest.</p>
      <h2>References</h2>
      <p>Smith 1999 junk citation should be dropped.</p>
      <h2>See also</h2>
      <p>Also dropped.</p>
    </div>
    """
    text = sanitize_zim_html(html, title="Benzene", url="/wiki/Benzene")
    assert text is not None
    assert "Useful article body" in text
    assert "Smith 1999" not in text
    assert "Also dropped" not in text
    assert is_mediawiki_like(title="Benzene", url="/wiki/Benzene")


def test_mediawiki_strips_navbox_and_toc():
    html = """
    <div id="mw-content-text">
      <div id="toc">Contents list</div>
      <div class="navbox">Related portal chrome</div>
      <p>Article prose about titration that clears the eighty character minimum
      easily when sanitising MediaWiki HTML from a ZIM archive.</p>
    </div>
    """
    text = sanitize_zim_html(html, title="Titration", url="/wiki/Titration")
    assert text is not None
    assert "Article prose about titration" in text
    assert "Contents list" not in text
    assert "Related portal chrome" not in text


def test_short_html_returns_none():
    assert sanitize_zim_html("<p>tiny</p>", title="x", url="x") is None

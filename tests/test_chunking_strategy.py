"""Tests for ingest chunk strategy selection."""

from __future__ import annotations

from ingest.chunking_strategy import ChunkContext, ChunkStrategy, select_chunk_strategy


def test_arxiv_pdf_selects_semantic():
    ctx = ChunkContext.from_path("/data/arxiv/2301.12345.pdf", "pdf")
    text = "Abstract\nWe study agents.\nIntroduction\nMethods and results follow."
    assert select_chunk_strategy(ctx, text) is ChunkStrategy.SEMANTIC


def test_markdown_file_selects_recursive():
    ctx = ChunkContext.from_path("/docs/node.md", "text")
    text = "---\ntitle: Node\n---\n\n# Section\n\nBody paragraph one.\n\n## Sub\n\nMore text."
    assert select_chunk_strategy(ctx, text) is ChunkStrategy.RECURSIVE


def test_zim_article_selects_sentence():
    ctx = ChunkContext.from_path("/wiki/all.zim", "zim")
    text = "This is a wiki article. It has sentences but no markdown headers."
    assert select_chunk_strategy(ctx, text) is ChunkStrategy.SENTENCE


def test_unstructured_scrape_selects_token():
    ctx = ChunkContext.from_path("/dump/page.txt", "text")
    text = "word " * 400
    assert select_chunk_strategy(ctx, text) is ChunkStrategy.TOKEN


def test_code_like_text_selects_code():
    ctx = ChunkContext.from_path("/repo/playbook.yml", "text")
    text = "\n".join(
        [
            "```yaml",
            "tasks:",
            "```",
            "def configure(host):",
            "    return host",
            "class Node:",
            "    pass",
        ]
        * 5
    )
    assert select_chunk_strategy(ctx, text) is ChunkStrategy.CODE


def test_plain_pdf_without_academic_markers_uses_sentence():
    ctx = ChunkContext.from_path("/uploads/manual.pdf", "pdf")
    text = "Chapter 1\n\nThis manual explains the product. It is not a research paper."
    assert select_chunk_strategy(ctx, text) is ChunkStrategy.SENTENCE

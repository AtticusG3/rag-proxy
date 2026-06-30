"""Tests for PDF page-level chunking in the ingest worker."""

from __future__ import annotations

from unittest.mock import patch

from ingest.worker import ChunkConfig, _iter_chunks_for_file


def test_pdf_chunks_per_page_not_whole_document() -> None:
    pages = [
        ("Page 1", "Alpha content for page one."),
        ("Page 2", "Beta content for page two."),
    ]

    with patch("ingest.worker.iter_pdf_pages", return_value=iter(pages)):
        with patch(
            "ingest.worker.chunk_text",
            side_effect=lambda text, **kwargs: [text],
        ) as chunk_mock:
            chunks = list(
                _iter_chunks_for_file(
                    "/uploads/report.pdf",
                    max_articles=0,
                    chunk_config=ChunkConfig(),
                )
            )

    assert len(chunks) == 2
    titles = [title for title, _source, _text in chunks]
    assert titles == ["Report (Page 1)", "Report (Page 2)"]
    assert chunk_mock.call_count == 2
    chunk_mock.assert_any_call("Alpha content for page one.", context=chunk_mock.call_args_list[0].kwargs["context"], config=chunk_mock.call_args_list[0].kwargs["config"])
    second_text = chunk_mock.call_args_list[1].args[0]
    assert second_text.startswith("Alpha content for page one.")
    assert "Beta content for page two." in second_text


def test_pdf_skips_blank_pages() -> None:
    pages = [
        ("Page 1", "   "),
        ("Page 2", "Real text here."),
    ]

    with patch("ingest.worker.iter_pdf_pages", return_value=iter(pages)):
        with patch("ingest.worker.chunk_text", side_effect=lambda text, **kwargs: [text]):
            chunks = list(
                _iter_chunks_for_file(
                    "/uploads/report.pdf",
                    max_articles=0,
                    chunk_config=ChunkConfig(),
                )
            )

    assert len(chunks) == 1
    assert chunks[0][0] == "Report (Page 2)"


def test_pdf_carries_last_paragraph_into_next_page() -> None:
    pages = [
        ("Page 1", "Intro paragraph.\n\nTrailing carry paragraph."),
        ("Page 2", "New page opener."),
    ]

    with patch("ingest.worker.iter_pdf_pages", return_value=iter(pages)):
        with patch(
            "ingest.worker.chunk_text",
            side_effect=lambda text, **kwargs: [text],
        ) as chunk_mock:
            list(
                _iter_chunks_for_file(
                    "/uploads/report.pdf",
                    max_articles=0,
                    chunk_config=ChunkConfig(),
                )
            )

    second_call_text = chunk_mock.call_args_list[1].args[0]
    assert second_call_text.startswith("Trailing carry paragraph.")
    assert "New page opener." in second_call_text

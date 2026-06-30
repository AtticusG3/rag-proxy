"""Extract text from PDF files for embedding."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path


def _open_pdf_reader(path: str | Path):
    from pypdf import PdfReader

    pdf_path = Path(path)
    reader = PdfReader(str(pdf_path))
    if reader.is_encrypted:
        # Many "searchable" scans use owner encryption with an empty user password.
        result = reader.decrypt("")
        if result == 0:
            raise ValueError(
                f"PDF is password-protected and could not be opened: {pdf_path.name}"
            )
    return reader


def iter_pdf_pages(path: str | Path) -> Iterator[tuple[str, str]]:
    """Yield (page_label, page_text) for each PDF page."""
    reader = _open_pdf_reader(path)
    for index, page in enumerate(reader.pages, start=1):
        yield f"Page {index}", page.extract_text() or ""


def read_pdf_text(path: str | Path) -> tuple[str, str]:
    """Return (title, full_text) for a PDF file."""
    pdf_path = Path(path)
    title = pdf_path.stem.replace("_", " ")
    pages = [text for _, text in iter_pdf_pages(path)]
    return title, "\n\n".join(pages).strip()

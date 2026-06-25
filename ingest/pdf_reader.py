"""Extract text from PDF files for embedding."""

from __future__ import annotations

from pathlib import Path


def read_pdf_text(path: str | Path) -> tuple[str, str]:
    """Return (title, full_text) for a PDF file."""
    from pypdf import PdfReader

    pdf_path = Path(path)
    reader = PdfReader(str(pdf_path))
    pages: list[str] = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    title = pdf_path.stem.replace("_", " ")
    return title, "\n\n".join(pages).strip()

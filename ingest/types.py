"""File type detection for ingest pipeline."""

from __future__ import annotations

from pathlib import Path


class IngestAborted(Exception):
    """Ingest stopped cooperatively (pause or shutdown)."""


EMBEDDABLE_SUFFIXES = {
    ".zim": "zim",
    ".txt": "text",
    ".md": "text",
    ".pdf": "pdf",
}

SKIP_NAMES = frozenset({"kiwix-library.xml"})


def determine_file_type(path: str | Path) -> str:
    """Return zim, text, pdf, or unknown."""
    p = Path(path)
    if p.name in SKIP_NAMES:
        return "unknown"
    suffix = p.suffix.lower()
    if suffix in EMBEDDABLE_SUFFIXES:
        return EMBEDDABLE_SUFFIXES[suffix]
    return "unknown"

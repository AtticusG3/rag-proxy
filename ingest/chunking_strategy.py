"""Select a Chonkie chunking strategy from document source and content."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

_ARXIV_ID_RE = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$", re.IGNORECASE)
_MD_HEADER_RE = re.compile(r"^#{1,6}\s+\S", re.MULTILINE)
_YAML_FRONTMATTER_RE = re.compile(r"^---\s*\n", re.MULTILINE)
_MD_TABLE_RE = re.compile(r"^\|.+\|\s*$", re.MULTILINE)
_MD_TABLE_SEP_RE = re.compile(r"^\|[\s\-:|]+\|\s*$", re.MULTILINE)
_CODE_LINE_RE = re.compile(
    r"^\s*(def |class |import |from |function |const |var |#include |package )",
    re.MULTILINE,
)


class ChunkStrategy(str, Enum):
    RECURSIVE = "recursive"
    SENTENCE = "sentence"
    SEMANTIC = "semantic"
    TOKEN = "token"
    CODE = "code"


@dataclass(frozen=True)
class ChunkContext:
    """Source metadata for chunk strategy selection."""

    file_type: str = "text"
    source_path: str = ""

    @classmethod
    def from_path(cls, path: str, file_type: str) -> ChunkContext:
        return cls(file_type=file_type, source_path=path)

    @property
    def suffix(self) -> str:
        return Path(self.source_path).suffix.lower()

    @property
    def is_arxiv_pdf(self) -> bool:
        path_lower = self.source_path.lower()
        if "arxiv" in path_lower:
            return True
        stem = Path(self.source_path).stem
        return bool(_ARXIV_ID_RE.match(stem))


def _looks_like_code(text: str) -> bool:
    if text.count("```") >= 2:
        return True
    lines = text.splitlines()[:120]
    if not lines:
        return False
    hits = sum(1 for line in lines if _CODE_LINE_RE.match(line))
    return hits / len(lines) >= 0.25


def _is_unstructured_junk(text: str) -> bool:
    """Scraped/OCR/plain dumps with weak paragraph structure."""
    if not text.strip():
        return False
    paragraph_breaks = text.count("\n\n")
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return True
    avg_line = sum(len(line) for line in lines) / len(lines)
    if paragraph_breaks == 0 and avg_line > 160:
        return True
    if paragraph_breaks <= 1 and len(lines) > 40 and avg_line > 120:
        return True
    return False


def _has_markdown_structure(text: str) -> bool:
    if _YAML_FRONTMATTER_RE.search(text):
        return True
    if _MD_HEADER_RE.search(text):
        return True
    if _MD_TABLE_RE.search(text) and _MD_TABLE_SEP_RE.search(text):
        return True
    return text.count("\n\n") >= 3


def _looks_academic_pdf(context: ChunkContext, text: str) -> bool:
    if context.file_type != "pdf":
        return False
    if context.is_arxiv_pdf:
        return True
    lowered = text[:4000].lower()
    markers = (
        "abstract",
        "introduction",
        "arxiv:",
        "doi:",
        "proceedings",
        "figure ",
        "theorem",
        "references\n",
    )
    return sum(1 for marker in markers if marker in lowered) >= 2


def select_chunk_strategy(context: ChunkContext, text: str) -> ChunkStrategy:
    """Pick the best Chonkie strategy for this document unit."""
    sample = text.strip()
    if not sample:
        return ChunkStrategy.RECURSIVE

    if _looks_like_code(sample):
        return ChunkStrategy.CODE

    if _is_unstructured_junk(sample):
        return ChunkStrategy.TOKEN

    if context.suffix == ".md" or _has_markdown_structure(sample):
        return ChunkStrategy.RECURSIVE

    if _looks_academic_pdf(context, sample):
        return ChunkStrategy.SEMANTIC

    if context.file_type == "pdf":
        return ChunkStrategy.SENTENCE

    if context.file_type == "zim":
        return ChunkStrategy.SENTENCE

    if context.suffix == ".txt" and _has_markdown_structure(sample):
        return ChunkStrategy.RECURSIVE

    if context.file_type == "text":
        return ChunkStrategy.SENTENCE

    return ChunkStrategy.RECURSIVE

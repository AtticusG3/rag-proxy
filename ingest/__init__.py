"""Ingest pipeline: ZIM/text -> embed -> Qdrant."""

from ingest.chunking import chunk_text
from ingest.embedder import embed_texts
from ingest.qdrant_writer import (
    build_point,
    delete_by_source,
    get_collection_count,
    upsert_points,
)
from ingest.scanner import scan_storage
from ingest.types import determine_file_type
from ingest.worker import IngestConfig, IngestWorker, process_file
from ingest.zim_reader import iter_zim_articles

__all__ = [
    "IngestConfig",
    "IngestWorker",
    "build_point",
    "chunk_text",
    "delete_by_source",
    "determine_file_type",
    "embed_texts",
    "get_collection_count",
    "iter_zim_articles",
    "process_file",
    "scan_storage",
    "upsert_points",
]

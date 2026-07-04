"""Unit tests for BM25 sparse sidecar index helpers."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPARSE_DIR = ROOT / "sidecars" / "sparse"
sys.path.insert(0, str(SPARSE_DIR))

from core import IndexRegistry, SparseIndex, point_to_doc, slim_payload  # noqa: E402


def test_slim_payload_keeps_text_and_recency_only() -> None:
    full = {
        "text": "hello world",
        "source_path": "/data/huge/archive.zim",
        "title": "Very Long Title" * 100,
        "updated_at": "2026-01-01T00:00:00Z",
    }
    slim = slim_payload(full, "hello world")
    assert slim["text"] == "hello world"
    assert slim["updated_at"] == "2026-01-01T00:00:00Z"
    assert "source_path" not in slim
    assert "title" not in slim


def test_point_to_doc_skips_empty_text() -> None:
    assert point_to_doc({"id": "1", "payload": {}}) is None


def test_sparse_index_search_returns_matching_docs() -> None:
    points = [
        {"id": "a", "payload": {"text": "python asyncio tutorial"}},
        {"id": "b", "payload": {"text": "rust ownership basics"}},
        {"id": "c", "payload": {"text": "python asyncio patterns"}},
    ]
    index = SparseIndex()
    index.add_points(points)
    index.finalize("test")

    hits = index.search("python asyncio", limit=2)
    ids = {h["id"] for h in hits}
    assert ids == {"a", "c"}
    for hit in hits:
        assert "text" in hit["payload"]
        assert "source_path" not in hit["payload"]


def test_registry_install_replaces_old_index() -> None:
    registry = IndexRegistry()
    first = SparseIndex()
    first.add_points([{"id": "1", "payload": {"text": "alpha beta"}}])
    first.finalize("test")
    registry.install("test", first)

    second = SparseIndex()
    second.add_points([{"id": "2", "payload": {"text": "gamma delta"}}])
    second.finalize("test")
    registry.install("test", second)

    assert registry.doc_count("test") == 1
    hits = registry.search("test", "gamma", limit=5)
    assert hits and hits[0]["id"] == "2"

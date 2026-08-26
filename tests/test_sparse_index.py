"""Unit tests for BM25 sparse sidecar index helpers."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPARSE_DIR = ROOT / "sidecars" / "sparse"
sys.path.insert(0, str(SPARSE_DIR))

from core import (  # noqa: E402
    IndexRegistry,
    SparseIndex,
    collection_persist_dir,
    point_to_doc,
    slim_payload,
)


def test_slim_payload_keeps_text_provenance_and_recency() -> None:
    """Sparse-only hits never re-fetch Qdrant, so title/source must stay in the slim payload."""
    full = {
        "text": "hello world",
        "source": "/zim/archive.zim",
        "source_path": "/data/huge/archive.zim",
        "title": "Catalysts",
        "updated_at": "2026-01-01T00:00:00Z",
    }
    slim = slim_payload(full, "hello world")
    assert slim["text"] == "hello world"
    assert slim["updated_at"] == "2026-01-01T00:00:00Z"
    assert slim["title"] == "Catalysts"
    assert slim["source"] == "/zim/archive.zim"
    assert "source_path" not in slim


def test_point_to_doc_skips_empty_text() -> None:
    assert point_to_doc({"id": "1", "payload": {}}) is None


def test_sparse_index_search_returns_matching_docs() -> None:
    points = [
        {
            "id": "a",
            "payload": {
                "text": "python asyncio tutorial",
                "title": "Asyncio",
                "source": "/zim/python.zim",
            },
        },
        {"id": "b", "payload": {"text": "rust ownership basics"}},
        {"id": "c", "payload": {"text": "python asyncio patterns"}},
    ]
    index = SparseIndex()
    index.add_points(points)
    index.finalize("test")

    hits = index.search("python asyncio", limit=2)
    ids = {h["id"] for h in hits}
    assert ids == {"a", "c"}
    by_id = {h["id"]: h for h in hits}
    assert by_id["a"]["payload"]["title"] == "Asyncio"
    assert by_id["a"]["payload"]["source"] == "/zim/python.zim"
    for hit in hits:
        assert "text" in hit["payload"]
        assert "source_path" not in hit["payload"]


def test_search_does_not_pad_results_with_unmatched_docs() -> None:
    """bm25s always returns k rows, padding with zero-score docs. Those are not
    matches -- letting them through would feed unrelated ids into the proxy's RRF
    merge, where rank alone decides and a zero score is invisible."""
    points = [
        {"id": "a", "payload": {"text": "python asyncio tutorial"}},
        {"id": "b", "payload": {"text": "rust ownership basics"}},
        {"id": "c", "payload": {"text": "baking sourdough bread"}},
    ]
    index = SparseIndex()
    index.add_points(points)
    index.finalize("test")

    hits = index.search("asyncio", limit=5)
    assert [h["id"] for h in hits] == ["a"]
    assert hits[0]["score"] > 0


def test_stopword_only_query_matches_nothing() -> None:
    """Stopwords are dropped at index time to keep the full corpus in RAM, so a
    query made only of them has no terms left and must return nothing rather than
    the entire corpus."""
    index = SparseIndex()
    index.add_points([{"id": "a", "payload": {"text": "python asyncio tutorial"}}])
    index.finalize("test")

    assert index.search("the and of", limit=5) == []


def test_search_ranks_denser_term_matches_first() -> None:
    """Score ordering drives the sparse leg's rank, so the doc covering both query
    terms must outrank the one covering a single term."""
    points = [
        {"id": "both", "payload": {"text": "python asyncio guide to asyncio"}},
        {"id": "one", "payload": {"text": "python packaging guide"}},
    ]
    index = SparseIndex()
    index.add_points(points)
    index.finalize("test")

    hits = index.search("python asyncio", limit=2)
    assert [h["id"] for h in hits] == ["both", "one"]
    assert hits[0]["score"] > hits[1]["score"]


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


def _sample_points() -> list[dict]:
    return [
        {
            "id": "a",
            "payload": {
                "text": "python asyncio tutorial",
                "title": "Asyncio",
                "source": "/zim/python.zim",
            },
        },
        {"id": "b", "payload": {"text": "rust ownership basics"}},
        {"id": "c", "payload": {"text": "python asyncio patterns"}},
    ]


def test_mmap_roundtrip_keeps_hits_and_payloads(tmp_path: Path) -> None:
    """A restart must serve the same sparse hits from NVMe without a Qdrant rebuild.

    The sidecar writes the BM25 CSC + jsonl corpus, drops the heap copy, and
    reopens both via mmap. Search still has to return provenance fields because
    sparse-only hits never re-fetch Qdrant.
    """
    persist = tmp_path / "sparse" / "test"
    built = SparseIndex()
    built.add_points(_sample_points())
    built.finalize("test", persist_dir=persist, mmap=True)

    assert built.mmap is True
    hits = built.search("python asyncio", limit=2)
    assert {h["id"] for h in hits} == {"a", "c"}
    assert {h["id"]: h for h in hits}["a"]["payload"]["title"] == "Asyncio"

    built.close()

    loaded = SparseIndex.load_mmap(persist, "test")
    assert loaded is not None
    assert loaded.mmap is True
    assert loaded.point_count == 3
    loaded_hits = loaded.search("python asyncio", limit=2)
    assert {h["id"] for h in loaded_hits} == {"a", "c"}
    by_id = {h["id"]: h for h in loaded_hits}
    assert by_id["a"]["payload"]["title"] == "Asyncio"
    assert by_id["a"]["payload"]["source"] == "/zim/python.zim"
    assert by_id["a"]["score"] > 0
    loaded.close()


def test_mmap_search_still_drops_zero_score_padding(tmp_path: Path) -> None:
    """bm25s pads retrieve() to k with zeros; mmap must not feed those into RRF."""
    persist = tmp_path / "sparse" / "pad"
    index = SparseIndex()
    index.add_points(
        [
            {"id": "a", "payload": {"text": "python asyncio tutorial"}},
            {"id": "b", "payload": {"text": "rust ownership basics"}},
            {"id": "c", "payload": {"text": "baking sourdough bread"}},
        ]
    )
    index.finalize("pad", persist_dir=persist, mmap=True)

    hits = index.search("asyncio", limit=5)
    assert [h["id"] for h in hits] == ["a"]
    index.close()


def test_registry_load_persisted_skips_rebuild(tmp_path: Path) -> None:
    """Boot should mmap the last snapshot instead of scrolling Qdrant again."""
    root = tmp_path / "sparse-root"
    writer = IndexRegistry(persist_root=root, mmap=True)
    writer.rebuild("kb", _sample_points())
    persist = collection_persist_dir(root, "kb")
    assert (persist / "params.index.json").is_file()
    writer_hits = writer.search("kb", "asyncio", limit=5)

    reader = IndexRegistry(persist_root=root, mmap=True)
    assert reader.load_persisted("kb") == 3
    assert reader.mmap_enabled("kb") is True
    reader_hits = reader.search("kb", "asyncio", limit=5)
    assert [h["id"] for h in reader_hits] == [h["id"] for h in writer_hits]


def test_load_mmap_missing_dir_returns_none(tmp_path: Path) -> None:
    """A first boot has no snapshot, so the sidecar must fall through to Qdrant sync."""
    assert SparseIndex.load_mmap(tmp_path / "missing", "kb") is None


def test_threaded_tokenize_keeps_the_same_top_hits() -> None:
    """Rebuild ranking must not change when regex split fans out across threads."""
    points = [
        {
            "id": str(i),
            "payload": {
                "text": (
                    f"python asyncio tutorial {i}"
                    if i % 21 == 0
                    else f"rust ownership basics {i}"
                )
            },
        }
        for i in range(1200)
    ]
    serial = SparseIndex(n_threads=1)
    serial.add_points(points)
    serial.finalize("t")
    threaded = SparseIndex(n_threads=4)
    threaded.add_points(points)
    threaded.finalize("t")

    query = "python asyncio"
    assert [h["id"] for h in serial.search(query, 8)] == [
        h["id"] for h in threaded.search(query, 8)
    ]
    serial.close()
    threaded.close()

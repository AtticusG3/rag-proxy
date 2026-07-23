"""Corpus JSONL ingest (rag-scrape-pipeline output)."""

from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import patch

from ingest.chunking_strategy import ChunkContext, ChunkStrategy, select_chunk_strategy
from ingest.qdrant_writer import DEFAULT_VECTOR_SIZE
from ingest.scanner import scan_storage
from ingest.types import determine_file_type
from ingest.worker import (
    IngestConfig,
    _corpus_chunk_prefix,
    _corpus_title_and_source,
    _delete_file_points,
    _iter_chunks_for_file,
    _iter_corpus_jsonl_records,
    process_file,
)
from ingest.chunking import ChunkConfig


def test_determine_file_type_recognizes_corpus_jsonl() -> None:
    assert determine_file_type("/data/my_corpus.jsonl") == "corpus_jsonl"
    assert determine_file_type("/data/corpus-2026-07.jsonl") == "corpus_jsonl"
    assert determine_file_type("/data/finetune.jsonl") == "unknown"
    assert determine_file_type("/data/notes.jsonl") == "unknown"


def test_scan_storage_includes_corpus_jsonl() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        corpus_path = os.path.join(tmp, "web_corpus.jsonl")
        skip_path = os.path.join(tmp, "capture.jsonl")
        with open(corpus_path, "w", encoding="utf-8") as handle:
            handle.write("{}\n")
        with open(skip_path, "w", encoding="utf-8") as handle:
            handle.write("{}\n")

        found = scan_storage(tmp)

    assert corpus_path in found
    assert skip_path not in found


def test_corpus_jsonl_selects_recursive_strategy() -> None:
    """Corpus must use recursive chunking so ## [N] / paragraph breaks win."""
    ctx = ChunkContext.from_path("/data/austlii.corpus.jsonl", "corpus_jsonl")
    text = "## [1]\n\nReasons for judgment.\n\n## [2]\n\nFurther reasons."
    assert select_chunk_strategy(ctx, text) is ChunkStrategy.RECURSIVE
    # Explicit file_type beats unstructured/token heuristics.
    plain = "word " * 400
    assert select_chunk_strategy(ctx, plain) is ChunkStrategy.RECURSIVE


def test_corpus_chunk_prefix_uses_prebuilt_then_rebuilds() -> None:
    prebuilt = (
        "[2022] FedCFamC1F 771 | FedCFamC1F 2022 | Topics: JURISDICTION; PROPERTY"
    )
    assert (
        _corpus_chunk_prefix({"chunk_prefix": prebuilt, "citation": "[2022] X 1"})
        == prebuilt
    )
    rebuilt = _corpus_chunk_prefix(
        {
            "citation": "[2022] FedCFamC1F 771",
            "court": "FedCFamC1F",
            "year": 2022,
            "topics": ["JURISDICTION", "PROPERTY"],
        }
    )
    assert rebuilt.startswith("[2022] FedCFamC1F 771 | FedCFamC1F 2022")
    assert "Topics: JURISDICTION; PROPERTY" in rebuilt


def test_corpus_title_prefers_citation_source_prefers_url() -> None:
    title, source = _corpus_title_and_source(
        {
            "citation": "[2022] FedCFamC1F 771",
            "title": "Display title",
            "source_url": "https://www.austlii.edu.au/cgi-bin/viewdoc/au/cases/cth/FedCFamC1F/2022/771.html",
            "doc_id": "[2022] FedCFamC1F 771",
            "path": "771.html",
        },
        file_path="/uploads/austlii.corpus.jsonl",
    )
    assert title == "[2022] FedCFamC1F 771"
    assert source.endswith("/2022/771.html")

    title2, source2 = _corpus_title_and_source(
        {
            "citation": "[2022] FedCFamC1F 771",
            "doc_id": "doc-771",
            "path": "771.html",
        },
        file_path="/uploads/austlii.corpus.jsonl",
    )
    assert title2 == "[2022] FedCFamC1F 771"
    assert source2 == "doc-771"


def test_iter_corpus_skips_empty_text() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "site_corpus.jsonl")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(json.dumps({"title": "Empty", "text": "  "}) + "\n")
            handle.write(
                json.dumps(
                    {
                        "title": "Keep",
                        "source_url": "https://example.com/keep",
                        "text": "Body text.",
                    }
                )
                + "\n"
            )
        rows = list(_iter_corpus_jsonl_records(path))
    assert len(rows) == 1
    assert rows[0][0] == "Keep"
    assert rows[0][2] == "Body text."


def test_iter_chunks_prefixes_corpus_pieces_and_sets_source() -> None:
    """Dense/BM25 keep case identity: every chunk starts with chunk_prefix."""
    prefix = "[2022] FedCFamC1F 771 | FedCFamC1F 2022 | Topics: PROPERTY"
    body = "## [1]\n\nFirst reason paragraph.\n\n## [2]\n\nSecond reason paragraph."
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "austlii.corpus.jsonl")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "citation": "[2022] FedCFamC1F 771",
                        "title": "Should not win",
                        "source_url": "https://example.com/judgment/771",
                        "chunk_prefix": prefix,
                        "doc_id": "[2022] FedCFamC1F 771",
                        "court": "FedCFamC1F",
                        "year": 2022,
                        "topics": ["PROPERTY"],
                        "text": body,
                    }
                )
                + "\n"
            )

        with patch(
            "ingest.worker.chunk_text",
            side_effect=lambda text, **kwargs: ["## [1]\n\nFirst", "## [2]\n\nSecond"],
        ) as chunk_mock:
            chunks = list(
                _iter_chunks_for_file(
                    path,
                    max_articles=0,
                    chunk_config=ChunkConfig(),
                )
            )

    assert len(chunks) == 2
    for row in chunks:
        title, source, text = row[0], row[1], row[2]
        assert title == "[2022] FedCFamC1F 771"
        assert source == "https://example.com/judgment/771"
        assert text.startswith(prefix + "\n\n")
        assert len(row) == 4
        assert row[3]["doc_id"] == "[2022] FedCFamC1F 771"
        assert row[3]["court"] == "FedCFamC1F"
    ctx = chunk_mock.call_args.kwargs["context"]
    assert ctx.file_type == "corpus_jsonl"
    assert not str(ctx.source_path).endswith(".md")


def test_iter_chunks_rebuilds_prefix_when_missing() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "cases.corpus.jsonl")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "citation": "[2022] FedCFamC1F 771",
                        "court": "FedCFamC1F",
                        "year": 2022,
                        "topics": ["JURISDICTION"],
                        "source_url": "https://example.com/771",
                        "text": "## [1]\n\nBody.",
                    }
                )
                + "\n"
            )

        with patch(
            "ingest.worker.chunk_text",
            side_effect=lambda text, **kwargs: [text],
        ):
            chunks = list(
                _iter_chunks_for_file(
                    path,
                    max_articles=0,
                    chunk_config=ChunkConfig(),
                )
            )

    assert len(chunks) == 1
    assert chunks[0][2].startswith(
        "[2022] FedCFamC1F 771 | FedCFamC1F 2022 | Topics: JURISDICTION\n\n"
    )


def test_process_file_corpus_jsonl_uses_per_record_source_url() -> None:
    """Each JSONL row should chunk with its own title and source_url in Qdrant."""
    active_embeds = 0
    max_active_embeds = 0
    all_points: list[dict] = []

    class FakeResponse:
        def __init__(self, status_code: int = 200, body: dict | None = None) -> None:
            self.status_code = status_code
            self._body = body or {}

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise RuntimeError(f"HTTP {self.status_code}")

        def json(self) -> dict:
            return self._body

    class FakeHttpClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def close(self) -> None:
            return None

        def get(self, url: str) -> FakeResponse:
            return FakeResponse(200)

        def put(self, url: str, json: dict) -> FakeResponse:
            if "/points" in url:
                all_points.extend(json["points"])
            return FakeResponse()

        def post(self, url: str, json: dict) -> FakeResponse:
            nonlocal active_embeds, max_active_embeds
            if url.endswith("/v1/embeddings"):
                active_embeds += 1
                max_active_embeds = max(max_active_embeds, active_embeds)
                try:
                    n = len(json["input"])
                    return FakeResponse(
                        body={
                            "data": [
                                {"embedding": [0.1] * DEFAULT_VECTOR_SIZE}
                                for _ in range(n)
                            ]
                        }
                    )
                finally:
                    active_embeds -= 1
            return FakeResponse()

    with tempfile.TemporaryDirectory() as tmp:
        file_path = os.path.join(tmp, "site_corpus.jsonl")
        records = [
            {
                "title": "Page One",
                "source_url": "https://example.com/one",
                "text": "First scraped page body.",
                "doc_id": "one",
                "usage": "reference-only",
            },
            {
                "title": "Page Two",
                "source_url": "https://example.com/two",
                "text": "Second scraped page body.",
            },
        ]
        with open(file_path, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")

        config = IngestConfig(
            zim_dir=tmp,
            upload_dir=tmp,
            embed_url="http://127.0.0.1:8089",
            qdrant_url="http://127.0.0.1:6333",
            qdrant_collection="test_collection",
            sparse_index_url="",
        )

        with patch("ingest.pipeline.httpx.Client", FakeHttpClient):
            count = process_file(file_path, config)

        assert count >= 2
        sources = {point["payload"]["source"] for point in all_points}
        titles = {point["payload"]["title"] for point in all_points}
        assert sources == {"https://example.com/one", "https://example.com/two"}
        assert titles == {"Page One", "Page Two"}
        one_points = [
            p for p in all_points if p["payload"]["source"] == "https://example.com/one"
        ]
        assert one_points[0]["payload"]["doc_id"] == "one"
        assert one_points[0]["payload"]["usage"] == "reference-only"


def test_delete_file_points_corpus_jsonl_deletes_each_source_url() -> None:
    deleted: list[str] = []

    def capture_delete(_url: str, _collection: str, source: str) -> None:
        deleted.append(source)

    with tempfile.TemporaryDirectory() as tmp:
        file_path = os.path.join(tmp, "wiki_corpus.jsonl")
        with open(file_path, "w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "title": "A",
                        "source_url": "https://example.com/a",
                        "text": "alpha",
                    }
                )
                + "\n"
            )
            handle.write(
                json.dumps(
                    {
                        "title": "B",
                        "source_url": "https://example.com/b",
                        "text": "beta",
                    }
                )
                + "\n"
            )

        config = IngestConfig(
            zim_dir=tmp,
            upload_dir=tmp,
            embed_url="http://127.0.0.1:8089",
            qdrant_url="http://127.0.0.1:6333",
            qdrant_collection="test_collection",
            sparse_index_url="",
        )

        with patch("ingest.worker.delete_by_source", side_effect=capture_delete):
            _delete_file_points(file_path, config)

    assert sorted(deleted) == [
        "https://example.com/a",
        "https://example.com/b",
    ]

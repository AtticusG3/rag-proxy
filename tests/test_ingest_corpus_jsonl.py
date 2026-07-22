"""Corpus JSONL ingest (rag-scrape-pipeline output)."""

from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import patch

from ingest.qdrant_writer import DEFAULT_VECTOR_SIZE
from ingest.scanner import scan_storage
from ingest.types import determine_file_type
from ingest.worker import IngestConfig, _delete_file_points, process_file


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

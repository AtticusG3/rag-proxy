"""Tests for process_file embed + Qdrant upsert payload conventions."""

from __future__ import annotations

import os
import tempfile
from unittest.mock import patch

from ingest.qdrant_writer import DEFAULT_VECTOR_SIZE
from ingest.worker import IngestConfig, process_file


def test_process_file_upserts_text_source_title_payload() -> None:
    """Qdrant points must carry text, source, and title for RAG lookup."""
    captured_points: list[dict] = []

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

        def __enter__(self) -> FakeHttpClient:
            return self

        def __exit__(self, *args) -> None:
            return None

        def get(self, url: str) -> FakeResponse:
            assert "/collections/" in url
            return FakeResponse(200)

        def put(self, url: str, json: dict) -> FakeResponse:
            if "/points" in url:
                captured_points.extend(json["points"])
            return FakeResponse()

        def post(self, url: str, json: dict) -> FakeResponse:
            if url.endswith("/v1/embeddings"):
                n = len(json["input"])
                return FakeResponse(
                    body={
                        "data": [
                            {"embedding": [0.1] * DEFAULT_VECTOR_SIZE}
                            for _ in range(n)
                        ]
                    }
                )
            return FakeResponse()

    with tempfile.TemporaryDirectory() as tmp:
        file_path = os.path.join(tmp, "notes.txt")
        with open(file_path, "w", encoding="utf-8") as handle:
            handle.write("First paragraph.\n\nSecond paragraph.")

        config = IngestConfig(
            zim_dir=tmp,
            upload_dir=tmp,
            embed_url="http://127.0.0.1:8089",
            qdrant_url="http://127.0.0.1:6333",
            qdrant_collection="test_collection",
            sparse_index_url="",
            batch_size=32,
        )

        with (
            patch("ingest.qdrant_writer.httpx.Client", FakeHttpClient),
            patch("ingest.embedder.httpx.Client", FakeHttpClient),
        ):
            count = process_file(file_path, config)

        assert count >= 1
        assert captured_points, "expected at least one upserted point"
        for point in captured_points:
            payload = point["payload"]
            assert "text" in payload
            assert payload["source"] == file_path
            assert payload["title"] == "Notes"
            assert isinstance(payload["text"], str)
            assert payload["text"].strip()

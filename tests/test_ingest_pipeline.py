"""Tests for pipelined concurrent bulk ingest."""

from __future__ import annotations

import os
import tempfile
import threading
import time
from unittest.mock import patch

from ingest.qdrant_writer import DEFAULT_VECTOR_SIZE
from ingest.pipeline import run_ingest_pipeline
from ingest.worker import IngestConfig, process_file


def _make_fake_http_client(*, embed_delay: float = 0.0) -> type:
    active_embeds = 0
    max_active_embeds = 0
    lock = threading.Lock()
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
        instances: list[FakeHttpClient] = []

        def __init__(self, *args, **kwargs) -> None:
            self.captured_points: list[dict] = []
            FakeHttpClient.instances.append(self)

        def close(self) -> None:
            return None

        def get(self, url: str) -> FakeResponse:
            return FakeResponse(200)

        def put(self, url: str, json: dict) -> FakeResponse:
            if "/points" in url:
                self.captured_points.extend(json["points"])
                all_points.extend(json["points"])
            return FakeResponse()

        def post(self, url: str, json: dict) -> FakeResponse:
            nonlocal active_embeds, max_active_embeds
            if url.endswith("/v1/embeddings"):
                with lock:
                    active_embeds += 1
                    max_active_embeds = max(max_active_embeds, active_embeds)
                try:
                    if embed_delay:
                        time.sleep(embed_delay)
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
                    with lock:
                        active_embeds -= 1
            return FakeResponse()

    FakeHttpClient.instances = []
    FakeHttpClient.max_active_embeds = lambda: max_active_embeds  # type: ignore[attr-defined]
    FakeHttpClient.all_points = lambda: list(all_points)  # type: ignore[attr-defined]
    return FakeHttpClient


def test_pipeline_preserves_chunk_idx_order() -> None:
    """Upserts must stay in chunk order even when embed batches complete out of order."""
    FakeHttpClient = _make_fake_http_client(embed_delay=0.02)

    def chunks():
        for i in range(6):
            yield ("Title", "/tmp/doc.txt", f"chunk-{i}")

    with patch("ingest.pipeline.httpx.Client", FakeHttpClient):
        total = run_ingest_pipeline(
            chunks(),
            embed_url="http://127.0.0.1:8089",
            qdrant_url="http://127.0.0.1:6333",
            qdrant_collection="test_collection",
            batch_size=2,
            embed_max_chars=2000,
            embed_concurrency=3,
        )

    assert total == 6
    points = FakeHttpClient.all_points()
    assert [p["payload"]["chunk_idx"] for p in points] == list(range(6))


def test_pipeline_overlaps_embed_requests_when_concurrency_gt_one() -> None:
    """Concurrent embed workers should overlap HTTP calls to the embed server."""
    FakeHttpClient = _make_fake_http_client(embed_delay=0.05)

    def chunks():
        for i in range(8):
            yield ("Title", "/tmp/doc.txt", f"chunk-{i}")

    with patch("ingest.pipeline.httpx.Client", FakeHttpClient):
        run_ingest_pipeline(
            chunks(),
            embed_url="http://127.0.0.1:8089",
            qdrant_url="http://127.0.0.1:6333",
            qdrant_collection="test_collection",
            batch_size=1,
            embed_max_chars=2000,
            embed_concurrency=4,
        )

    assert FakeHttpClient.max_active_embeds() >= 2


def test_pipeline_reports_progress_before_concurrency_window_fills() -> None:
    """Progress callbacks should fire as soon as ordered batches finish embedding."""
    FakeHttpClient = _make_fake_http_client()
    progress: list[int] = []

    def chunks():
        for i in range(4):
            yield ("Title", "/tmp/doc.txt", f"chunk-{i}")

    with patch("ingest.pipeline.httpx.Client", FakeHttpClient):
        run_ingest_pipeline(
            chunks(),
            embed_url="http://127.0.0.1:8089",
            qdrant_url="http://127.0.0.1:6333",
            qdrant_collection="test_collection",
            batch_size=1,
            embed_max_chars=2000,
            embed_concurrency=512,
            on_progress=lambda **kwargs: progress.append(int(kwargs["chunks_embedded"])),
        )

    assert progress == [1, 2, 3, 4]


def test_process_file_uses_pipeline_clients() -> None:
    """process_file must still write text/source/title payload fields via the pipeline."""
    FakeHttpClient = _make_fake_http_client()

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
            embed_concurrency=2,
        )

        with patch("ingest.pipeline.httpx.Client", FakeHttpClient):
            count = process_file(file_path, config)

        assert count >= 1
        captured_points = FakeHttpClient.all_points()
        assert captured_points
        for point in captured_points:
            payload = point["payload"]
            assert payload["source"] == file_path
            assert payload["title"] == "Notes"
            assert payload["text"].strip()

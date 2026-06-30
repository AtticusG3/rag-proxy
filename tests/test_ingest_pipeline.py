"""Tests for pipelined concurrent bulk ingest."""

from __future__ import annotations

import os
import tempfile
import threading
import time
from unittest.mock import patch

from ingest.qdrant_writer import DEFAULT_VECTOR_SIZE
from ingest.pipeline import make_embed_semaphore, run_ingest_pipeline
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


def test_pipeline_reuses_single_embed_client() -> None:
    """Embed batches should share one httpx.Client passed into embed_texts."""
    FakeHttpClient = _make_fake_http_client()
    seen_clients: list[object] = []

    def capture_embed(texts, **kwargs):
        client = kwargs.get("client")
        assert client is not None
        seen_clients.append(client)
        n = len(texts)
        return [[0.1] * DEFAULT_VECTOR_SIZE for _ in range(n)]

    def chunks():
        for i in range(6):
            yield ("Title", "/tmp/doc.txt", f"chunk-{i}")

    with patch("ingest.pipeline.httpx.Client", FakeHttpClient):
        with patch("ingest.pipeline.embed_texts", side_effect=capture_embed) as embed_mock:
            run_ingest_pipeline(
                chunks(),
                embed_url="http://127.0.0.1:8089",
                qdrant_url="http://127.0.0.1:6333",
                qdrant_collection="test_collection",
                batch_size=2,
                embed_max_chars=2000,
                embed_concurrency=3,
            )

    assert embed_mock.call_count == 3
    assert len(seen_clients) == 3
    assert len(set(id(client) for client in seen_clients)) == 1


def test_shared_embed_limiter_caps_across_concurrent_pipelines() -> None:
    """Two file pipelines must not exceed shared embed_limiter capacity."""
    FakeHttpClient = _make_fake_http_client(embed_delay=0.08)
    limiter = make_embed_semaphore(2)
    barrier = threading.Barrier(2)
    errors: list[Exception] = []

    def chunks(prefix: str):
        for i in range(6):
            yield ("Title", f"/tmp/{prefix}.txt", f"{prefix}-chunk-{i}")

    def run_pipeline(prefix: str) -> None:
        try:
            barrier.wait(timeout=5.0)
            run_ingest_pipeline(
                chunks(prefix),
                embed_url="http://127.0.0.1:8089",
                qdrant_url="http://127.0.0.1:6333",
                qdrant_collection="test_collection",
                batch_size=1,
                embed_max_chars=2000,
                embed_concurrency=4,
                embed_limiter=limiter,
            )
        except Exception as exc:
            errors.append(exc)

    with patch("ingest.pipeline.httpx.Client", FakeHttpClient):
        threads = [
            threading.Thread(target=run_pipeline, args=(f"doc-{index}",))
            for index in range(2)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30.0)

    assert not errors
    assert FakeHttpClient.max_active_embeds() <= 2

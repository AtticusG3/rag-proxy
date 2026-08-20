"""Endpoint health must pause ingest, not wipe already-embedded chunks."""

from __future__ import annotations

import os
import tempfile
import time
from unittest.mock import patch

import httpx
from ingest.db import IngestDatabase
from ingest.endpoint_health import (
    ProbeResult,
    check_ingest_endpoints,
    probe_embeds,
    probe_qdrant,
)
from ingest.types import EndpointUnavailable
from ingest.worker import IngestConfig, IngestWorker


def _clear_health_cache() -> None:
    import ingest.endpoint_health as module

    module._CACHE = None


def _stop_worker(worker: IngestWorker) -> None:
    worker.stop()
    for thread, _ in worker._workers:
        thread.join(timeout=5.0)


def _worker(db: IngestDatabase) -> IngestWorker:
    config = IngestConfig(
        zim_dir="/tmp",
        upload_dir="/tmp",
        embed_url="http://127.0.0.1:1",
        qdrant_url="http://127.0.0.1:1",
        qdrant_collection="test",
        sparse_index_url="",
        stall_seconds=900,
    )
    return IngestWorker(config, db)


def test_probe_embeds_ok_when_any_url_produces_a_vector() -> None:
    """A down pool member is not a failed file if another embed URL still works."""
    with patch("ingest.endpoint_health.probe_embed_url", side_effect=[False, True]):
        result = probe_embeds(["http://down:1", "http://up:2"])
    assert result.ok is True
    assert result.url == "http://up:2"


def test_probe_embeds_down_when_none_accept_embeddings() -> None:
    with patch("ingest.endpoint_health.probe_embed_url", return_value=False):
        result = probe_embeds(["http://down:1"])
    assert result.ok is False
    assert result.name == "embed"


def test_probe_qdrant_ok_on_readyz() -> None:
    class _Client:
        def __init__(self, *args, **kwargs) -> None:
            return None

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

        def get(self, url: str):
            response = httpx.Response(200, text="ok")
            return response

    with patch("ingest.endpoint_health.httpx.Client", _Client):
        result = probe_qdrant("http://qdrant:6333")
    assert result.ok is True


def test_check_ingest_endpoints_names_the_down_service() -> None:
    """Jobs UI and worker logs must say which endpoint to wait on."""
    _clear_health_cache()
    down_embed = ProbeResult("embed", "http://embed", False, "connection refused")
    up_qdrant = ProbeResult("qdrant", "http://qdrant", True)
    with patch("ingest.endpoint_health.probe_embeds", return_value=down_embed):
        with patch("ingest.endpoint_health.probe_qdrant", return_value=up_qdrant):
            health = check_ingest_endpoints(
                embed_urls=["http://embed"],
                qdrant_url="http://qdrant",
                use_cache=False,
            )
    assert health.ok is False
    assert health.waiting_on == "embed"


def test_worker_does_not_claim_or_reset_chunks_while_endpoints_are_down() -> None:
    """Embed outage is infrastructure: leave the 14000-chunk file pending."""
    with tempfile.TemporaryDirectory() as tmp:
        db = IngestDatabase(os.path.join(tmp, "admin.sqlite"))
        path = os.path.join(tmp, "big.zim")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("x")
        db.upsert_file_state(path, status="pending", chunks_embedded=14000)
        worker = _worker(db)
        with patch.object(worker, "_endpoints_ready", return_value=False):
            with patch("ingest.worker.process_file") as process:
                worker.start()
                time.sleep(0.4)
                _stop_worker(worker)
        process.assert_not_called()
        row = db.get_file_state(path)
        assert row is not None
        assert row["status"] == "pending"
        assert row["chunks_embedded"] == 14000


def test_endpoint_unavailable_requeues_pending_without_wiping_chunks() -> None:
    """A dead embed mid-file must not mark the job failed or delete progress."""

    def boom(*args, **kwargs):
        raise EndpointUnavailable("embed batch failed after 3 attempts: connection refused")

    with tempfile.TemporaryDirectory() as tmp:
        db = IngestDatabase(os.path.join(tmp, "admin.sqlite"))
        path = os.path.join(tmp, "big.txt")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("hello")
        db.upsert_file_state(path, status="pending", chunks_embedded=14000)
        worker = _worker(db)
        with patch.object(worker, "_endpoints_ready", return_value=True):
            with patch("ingest.worker.process_file", side_effect=boom):
                worker.start()
                deadline = time.time() + 5.0
                row = db.get_file_state(path)
                while time.time() < deadline:
                    row = db.get_file_state(path)
                    if row and "embed batch failed" in (row.get("last_error") or ""):
                        break
                    time.sleep(0.05)
                _stop_worker(worker)
        assert row is not None
        assert row["status"] == "pending"
        assert row["chunks_embedded"] == 14000
        assert "failed" not in (row.get("status") or "")


def test_process_one_resumes_from_chunks_already_embedded() -> None:
    """Retry after an outage must continue at chunk 14000, not re-embed from zero."""
    with tempfile.TemporaryDirectory() as tmp:
        db = IngestDatabase(os.path.join(tmp, "admin.sqlite"))
        path = os.path.join(tmp, "big.txt")
        db.upsert_file_state(path, status="running", chunks_embedded=14000)
        worker = _worker(db)
        with patch("ingest.worker.ensure_embed_urls"):
            with patch("ingest.worker.process_file", return_value=14002) as process:
                worker._process_one(path)
        assert process.call_args.kwargs["start_chunk_idx"] == 14000
        row = db.get_file_state(path)
        assert row is not None
        assert row["status"] == "indexed"
        assert row["chunks_embedded"] == 14002

"""Tests that ingest worker lock scope does not block process_file I/O."""

from __future__ import annotations

import os
import tempfile
import threading
from unittest.mock import patch

from ingest.db import IngestDatabase
from ingest.worker import IngestConfig, IngestWorker


def _worker(db: IngestDatabase) -> IngestWorker:
    config = IngestConfig(
        zim_dir="/tmp",
        upload_dir="/tmp",
        embed_url="http://127.0.0.1:1",
        qdrant_url="http://127.0.0.1:1",
        qdrant_collection="test",
        sparse_index_url="",
    )
    return IngestWorker(config, db)


def test_process_one_releases_lock_during_process_file() -> None:
    """Stall recovery and other DB work must proceed while embedding runs."""
    process_started = threading.Event()
    release_process = threading.Event()
    lock_acquired_during_process = threading.Event()

    def slow_process_file(file_path: str, config: IngestConfig, *, on_progress=None) -> int:
        process_started.set()
        release_process.wait(timeout=5.0)
        return 3

    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "admin.sqlite")
        db = IngestDatabase(db_path)
        file_path = "/tmp/sample.txt"
        db.upsert_file_state(file_path, status="pending", file_type="text")

        worker = _worker(db)

        def run_process_one() -> None:
            with patch("ingest.worker.process_file", side_effect=slow_process_file):
                worker._process_one(file_path)

        worker_thread = threading.Thread(target=run_process_one)
        worker_thread.start()

        assert process_started.wait(timeout=5.0), "process_file did not start"
        acquired = worker._lock.acquire(timeout=0.5)
        assert acquired, "worker lock still held during process_file network I/O"
        worker._lock.release()
        lock_acquired_during_process.set()
        release_process.set()
        worker_thread.join(timeout=5.0)

        row = db.get_file_state(file_path)
        assert row is not None
        assert row["status"] == "indexed"
        assert row["chunks_embedded"] == 3
        assert lock_acquired_during_process.is_set()

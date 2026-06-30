"""Tests for multi-file ingest worker concurrency."""

from __future__ import annotations

import os
import tempfile
import threading
import time
from unittest.mock import patch

from ingest.db import IngestDatabase
from ingest.worker import IngestConfig, IngestWorker, resolve_file_concurrency


def _worker(db: IngestDatabase, *, file_concurrency: int = 2) -> IngestWorker:
    config = IngestConfig(
        zim_dir="/tmp",
        upload_dir="/tmp",
        embed_url="http://127.0.0.1:1",
        qdrant_url="http://127.0.0.1:1",
        qdrant_collection="test",
        sparse_index_url="",
        file_concurrency=file_concurrency,
    )
    return IngestWorker(config, db)


def test_resolve_file_concurrency_defaults_to_embed_pool_cap() -> None:
    assert resolve_file_concurrency(["http://a", "http://b", "http://c", "http://d", "http://e"]) == 4
    assert resolve_file_concurrency(["http://a"]) == 1
    assert resolve_file_concurrency([], explicit=3) == 3


def test_worker_processes_multiple_files_in_parallel() -> None:
    active = 0
    peak_active = 0
    lock = threading.Lock()
    release = threading.Event()

    def slow_process_file(
        file_path: str,
        config: IngestConfig,
        *,
        on_progress=None,
        embed_limiter=None,
    ) -> int:
        nonlocal active, peak_active
        with lock:
            active += 1
            peak_active = max(peak_active, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        release.wait(timeout=5.0)
        return 1

    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "admin.sqlite")
        db = IngestDatabase(db_path)
        file_paths: list[str] = []
        for index in range(4):
            file_path = os.path.join(tmp, f"doc-{index}.txt")
            with open(file_path, "w", encoding="utf-8") as handle:
                handle.write(f"content {index}")
            file_paths.append(file_path)
            db.upsert_file_state(file_path, status="pending", file_type="text")

        worker = _worker(db, file_concurrency=2)
        with patch("ingest.worker.process_file", side_effect=slow_process_file):
            worker.start()
            deadline = time.time() + 5.0
            while time.time() < deadline:
                running = db.list_running_files()
                if len(running) >= 2:
                    break
                time.sleep(0.01)
            release.set()
            time.sleep(0.3)
            worker.stop()

        assert peak_active >= 2

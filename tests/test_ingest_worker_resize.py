"""Tests for dynamic file worker resizing."""

from __future__ import annotations

import os
import tempfile
import time
from dataclasses import replace

from ingest.db import IngestDatabase
from ingest.worker import IngestConfig, IngestWorker


def _config(file_concurrency: int) -> IngestConfig:
    return IngestConfig(
        zim_dir="/tmp",
        upload_dir="/tmp",
        embed_url="http://127.0.0.1:1",
        qdrant_url="http://127.0.0.1:1",
        qdrant_collection="test",
        sparse_index_url="",
        file_concurrency=file_concurrency,
    )


def _alive_count(worker: IngestWorker) -> int:
    return len([t for t, _ in worker._workers if t.is_alive()])


def _wait_for(predicate, timeout: float = 5.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def _stop_and_join(worker: IngestWorker) -> None:
    worker.stop()
    for thread, _ in worker._workers:
        thread.join(timeout=5.0)


def test_update_config_grows_and_shrinks_running_workers() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db = IngestDatabase(os.path.join(tmp, "ingest.sqlite"))
        worker = IngestWorker(_config(file_concurrency=1), db)
        worker.start()
        try:
            assert _wait_for(lambda: _alive_count(worker) == 1)

            worker.update_config(replace(worker.config, file_concurrency=3))
            assert _wait_for(lambda: _alive_count(worker) == 3)
            spawned = [thread for thread, _ in worker._workers]

            worker.update_config(replace(worker.config, file_concurrency=1))
            # Idle excess threads notice their stop event within the 1s poll sleep
            # and actually terminate, not just drop off the tracked list.
            assert _wait_for(
                lambda: sum(thread.is_alive() for thread in spawned) == 1
            )
        finally:
            _stop_and_join(worker)


def test_start_when_running_applies_new_worker_count() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db = IngestDatabase(os.path.join(tmp, "ingest.sqlite"))
        worker = IngestWorker(_config(file_concurrency=2), db)
        worker.start()
        try:
            assert _wait_for(lambda: _alive_count(worker) == 2)
            # start() on a running worker resizes instead of being a no-op.
            worker.config = replace(worker.config, file_concurrency=4)
            worker.start()
            assert _wait_for(lambda: _alive_count(worker) == 4)
        finally:
            _stop_and_join(worker)


def test_update_config_before_start_does_not_spawn_threads() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db = IngestDatabase(os.path.join(tmp, "ingest.sqlite"))
        worker = IngestWorker(_config(file_concurrency=2), db)
        worker.update_config(replace(worker.config, file_concurrency=4))
        assert _alive_count(worker) == 0

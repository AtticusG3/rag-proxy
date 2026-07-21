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
        should_abort=None,
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


def test_pause_aborts_in_flight_file_and_requeues() -> None:
    started = threading.Event()

    def blocking_process_file(
        file_path: str,
        config: IngestConfig,
        *,
        on_progress=None,
        embed_limiter=None,
        should_abort=None,
    ) -> int:
        started.set()
        while should_abort is None or not should_abort():
            time.sleep(0.02)
        from ingest.types import IngestAborted

        raise IngestAborted("paused")

    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "admin.sqlite")
        db = IngestDatabase(db_path)
        file_path = os.path.join(tmp, "doc.txt")
        with open(file_path, "w", encoding="utf-8") as handle:
            handle.write("content")
        db.upsert_file_state(file_path, status="pending", file_type="text")

        worker = _worker(db, file_concurrency=1)
        with patch("ingest.worker.process_file", side_effect=blocking_process_file):
            worker.start()
            assert started.wait(timeout=5.0)
            worker.set_paused(True)
            deadline = time.time() + 5.0
            while time.time() < deadline:
                row = db.get_file_state(file_path)
                if row and row["status"] == "pending":
                    break
                time.sleep(0.05)
            worker.stop()

        row = db.get_file_state(file_path)
        assert row is not None
        assert row["status"] == "pending"
        assert "paused" in (row.get("last_error") or "").lower()


def test_preempt_switches_worker_to_top_of_queue() -> None:
    """Preempt exists so a high-priority file does not wait hours behind a big ingest:
    the running file must yield (back to pending, not failed) and the worker must
    pick up the best pending file per priority order next."""
    started_paths: list[str] = []
    started = threading.Event()
    release = threading.Event()

    def blocking_process_file(
        file_path: str,
        config: IngestConfig,
        *,
        on_progress=None,
        embed_limiter=None,
        should_abort=None,
    ) -> int:
        started_paths.append(file_path)
        started.set()
        while not release.is_set():
            if should_abort and should_abort():
                from ingest.types import IngestAborted

                raise IngestAborted("preempted")
            time.sleep(0.02)
        return 1

    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "admin.sqlite")
        db = IngestDatabase(db_path)
        slow_path = os.path.join(tmp, "slow.txt")
        urgent_path = os.path.join(tmp, "urgent.txt")
        for path in (slow_path, urgent_path):
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("content")
        db.upsert_file_state(slow_path, status="pending", file_type="text")

        worker = _worker(db, file_concurrency=1)
        assert worker.preempt_running() == 0  # nothing running yet -> no-op

        with patch("ingest.worker.process_file", side_effect=blocking_process_file):
            worker.start()
            assert started.wait(timeout=5.0)

            db.upsert_file_state(urgent_path, status="pending", file_type="text")
            db.set_file_priority(urgent_path, "high")
            assert worker.preempt_running() == 1

            deadline = time.time() + 5.0
            while time.time() < deadline:
                if urgent_path in started_paths:
                    break
                time.sleep(0.05)
            release.set()
            worker.stop()

        assert started_paths[0] == slow_path
        assert urgent_path in started_paths
        slow_row = db.get_file_state(slow_path)
        assert slow_row is not None
        assert slow_row["status"] == "pending"
        assert "preempt" in (slow_row.get("last_error") or "").lower()


def test_stop_skips_sparse_flush_by_default() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "admin.sqlite")
        db = IngestDatabase(db_path)
        config = IngestConfig(
            zim_dir="/tmp",
            upload_dir="/tmp",
            embed_url="http://127.0.0.1:1",
            qdrant_url="http://127.0.0.1:1",
            qdrant_collection="test",
            sparse_index_url="http://127.0.0.1:1",
            sparse_reindex_mode="idle",
        )
        worker = IngestWorker(config, db)
        worker._sparse._dirty = True
        with patch("ingest.worker.trigger_sparse_reindex") as trigger:
            worker.stop()
            trigger.assert_not_called()
            worker.stop(flush_sparse=True)
            trigger.assert_called_once()

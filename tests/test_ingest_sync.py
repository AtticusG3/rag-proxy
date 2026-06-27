"""Tests for safe storage scan (no full re-embed)."""

from __future__ import annotations

import os
import tempfile
from unittest.mock import patch

from ingest.db import IngestDatabase
from ingest.types import determine_file_type
from ingest.worker import IngestConfig, IngestWorker, SparseReindexScheduler


def _worker_with_dirs(db: IngestDatabase, zim_dir: str, upload_dir: str) -> IngestWorker:
    config = IngestConfig(
        zim_dir=zim_dir,
        upload_dir=upload_dir,
        embed_url="http://127.0.0.1:1",
        qdrant_url="http://127.0.0.1:1",
        qdrant_collection="test",
        sparse_index_url="",
    )
    return IngestWorker(config, db)


def test_sparse_scheduler_idle_defers_until_flush() -> None:
    config = IngestConfig(
        zim_dir="/tmp",
        upload_dir="/tmp",
        embed_url="http://127.0.0.1:1",
        qdrant_url="http://127.0.0.1:1",
        qdrant_collection="test",
        sparse_index_url="http://127.0.0.1:1",
        sparse_reindex_mode="idle",
    )
    scheduler = SparseReindexScheduler(config)
    with patch("ingest.worker.trigger_sparse_reindex") as trigger:
        scheduler.after_file()
        trigger.assert_not_called()
        scheduler.flush()
        trigger.assert_called_once()


def test_sparse_scheduler_each_reindexes_immediately() -> None:
    config = IngestConfig(
        zim_dir="/tmp",
        upload_dir="/tmp",
        embed_url="http://127.0.0.1:1",
        qdrant_url="http://127.0.0.1:1",
        qdrant_collection="test",
        sparse_index_url="http://127.0.0.1:1",
        sparse_reindex_mode="each",
    )
    scheduler = SparseReindexScheduler(config)
    with patch("ingest.worker.trigger_sparse_reindex") as trigger:
        scheduler.after_file()
        trigger.assert_called_once()


def test_enqueue_sync_skips_indexed_files() -> None:
    with tempfile.TemporaryDirectory() as zim_dir:
        upload_dir = tempfile.mkdtemp()
        db_path = os.path.join(zim_dir, "admin.sqlite")
        db = IngestDatabase(db_path)
        zim_path = os.path.join(zim_dir, "sample.txt")
        with open(zim_path, "w", encoding="utf-8") as handle:
            handle.write("hello")

        db.upsert_file_state(
            zim_path,
            status="indexed",
            file_type=determine_file_type(zim_path),
            chunks_embedded=3,
        )

        worker = _worker_with_dirs(db, zim_dir, upload_dir)
        worker.enqueue_sync()

        row = db.get_file_state(zim_path)
        assert row is not None
        assert row["status"] == "indexed"
        assert row["chunks_embedded"] == 3


def test_enqueue_sync_retries_failed_only() -> None:
    with tempfile.TemporaryDirectory() as zim_dir:
        upload_dir = tempfile.mkdtemp()
        db_path = os.path.join(zim_dir, "admin.sqlite")
        db = IngestDatabase(db_path)
        ok_path = os.path.join(zim_dir, "ok.txt")
        bad_path = os.path.join(zim_dir, "bad.txt")
        for path, text in ((ok_path, "ok"), (bad_path, "bad")):
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(text)

        db.upsert_file_state(ok_path, status="indexed", chunks_embedded=1)
        db.upsert_file_state(
            bad_path,
            status="failed",
            last_error="boom",
            chunks_embedded=0,
        )

        worker = _worker_with_dirs(db, zim_dir, upload_dir)
        worker.enqueue_sync()

        assert db.get_file_state(ok_path)["status"] == "indexed"
        bad = db.get_file_state(bad_path)
        assert bad["status"] == "pending"
        assert bad["last_error"] is None


def test_prune_missing_files_removes_orphaned_rows() -> None:
    with tempfile.TemporaryDirectory() as zim_dir:
        upload_dir = tempfile.mkdtemp()
        db_path = os.path.join(zim_dir, "admin.sqlite")
        db = IngestDatabase(db_path)
        ghost_path = os.path.join(upload_dir, "gone.md")
        db.upsert_file_state(
            ghost_path,
            status="failed",
            last_error="[Errno 2] No such file or directory",
        )

        worker = _worker_with_dirs(db, zim_dir, upload_dir)
        with patch("ingest.worker.delete_by_source") as delete_mock:
            removed = worker.prune_missing_files()

        assert removed == [ghost_path]
        assert db.get_file_state(ghost_path) is None
        delete_mock.assert_called_once_with(
            worker.config.qdrant_url,
            worker.config.qdrant_collection,
            ghost_path,
        )


def test_enqueue_sync_registers_new_files() -> None:
    with tempfile.TemporaryDirectory() as zim_dir:
        upload_dir = tempfile.mkdtemp()
        db_path = os.path.join(zim_dir, "admin.sqlite")
        db = IngestDatabase(db_path)
        new_path = os.path.join(zim_dir, "fresh.txt")
        with open(new_path, "w", encoding="utf-8") as handle:
            handle.write("new")

        worker = _worker_with_dirs(db, zim_dir, upload_dir)
        job_id = worker.enqueue_sync()

        row = db.get_file_state(new_path)
        assert row is not None
        assert row["status"] == "pending"
        jobs = db.list_jobs(limit=5)
        assert any(j["id"] == job_id for j in jobs)

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


def _sparse_config(
    mode: str, *, sparse_index_url: str = "http://127.0.0.1:1"
) -> IngestConfig:
    return IngestConfig(
        zim_dir="/tmp",
        upload_dir="/tmp",
        embed_url="http://127.0.0.1:1",
        qdrant_url="http://127.0.0.1:1",
        qdrant_collection="test",
        sparse_index_url=sparse_index_url,
        sparse_reindex_mode=mode,
    )


def test_sparse_scheduler_idle_defers_until_flush() -> None:
    scheduler = SparseReindexScheduler(_sparse_config("idle"))
    with patch("ingest.worker.trigger_sparse_reindex") as trigger:
        scheduler.after_file()
        trigger.assert_not_called()
        scheduler.flush()
        trigger.assert_called_once()


def test_sparse_scheduler_each_reindexes_immediately() -> None:
    scheduler = SparseReindexScheduler(_sparse_config("each"))
    with patch("ingest.worker.trigger_sparse_reindex", return_value=42) as trigger:
        scheduler.after_file()
        trigger.assert_called_once()
    status = scheduler.status()
    assert status["reindexing"] is False
    assert status["dirty"] is False
    assert status["last_docs"] == 42


def test_sparse_scheduler_status_tracks_dirty_then_flush() -> None:
    scheduler = SparseReindexScheduler(_sparse_config("idle"))
    with patch("ingest.worker.trigger_sparse_reindex", return_value=7):
        with patch.object(scheduler, "_ensure_sidecar"):
            scheduler.after_file()
            assert scheduler.status()["dirty"] is True
            assert scheduler.status()["reindexing"] is False
            scheduler.flush()
    status = scheduler.status()
    assert status["dirty"] is False
    assert status["reindexing"] is False
    assert status["last_docs"] == 7
    assert status["last_finished_at"]

def test_sparse_scheduler_stays_dirty_when_flush_fails() -> None:
    """A failed flush must retry later, not silently drop the pending reindex.

    Clearing the dirty bit up front leaves BM25 permanently behind Qdrant after
    one transient sidecar error, with no signal until someone reads the logs.
    """
    config = _sparse_config("idle")
    scheduler = SparseReindexScheduler(config)
    with patch("ingest.worker.trigger_sparse_reindex", side_effect=RuntimeError("boom")):
        with patch.object(scheduler, "_ensure_sidecar"):
            scheduler.after_file()
            scheduler.flush()
            assert scheduler.status()["dirty"] is True
            assert scheduler.status()["last_error"] == "boom"

    with patch("ingest.worker.trigger_sparse_reindex", return_value=9) as trigger:
        with patch.object(scheduler, "_ensure_sidecar"):
            scheduler.flush()
            trigger.assert_called_once()
    assert scheduler.status()["dirty"] is False


def test_sparse_scheduler_stays_dirty_when_trigger_swallows_error() -> None:
    """trigger_sparse_reindex is fail-open and returns None on HTTP errors.

    The scheduler must read that as a failure, otherwise every real sidecar
    outage clears the dirty bit and BM25 never catches up.
    """
    scheduler = SparseReindexScheduler(_sparse_config("idle"))
    with patch("ingest.worker.trigger_sparse_reindex", return_value=None):
        with patch.object(scheduler, "_ensure_sidecar"):
            scheduler.after_file()
            scheduler.flush()
    status = scheduler.status()
    assert status["dirty"] is True
    assert status["last_error"]


def test_sparse_scheduler_each_mode_defers_failed_reindex_to_flush() -> None:
    """A per-file reindex that fails should be picked up by the idle flush."""
    config = _sparse_config("each")
    scheduler = SparseReindexScheduler(config)
    with patch("ingest.worker.trigger_sparse_reindex", side_effect=RuntimeError("boom")):
        scheduler.after_file()
    assert scheduler.status()["dirty"] is True

    with patch("ingest.worker.trigger_sparse_reindex", return_value=3) as trigger:
        with patch.object(scheduler, "_ensure_sidecar"):
            scheduler.flush()
            trigger.assert_called_once()
    assert scheduler.status()["dirty"] is False


def test_sparse_scheduler_inactive_without_sidecar_url() -> None:
    """No SPARSE_INDEX_URL means no reindex attempts at all."""
    scheduler = SparseReindexScheduler(_sparse_config("each", sparse_index_url=""))
    with patch("ingest.worker.trigger_sparse_reindex") as trigger:
        scheduler.after_file()
        scheduler.flush()
        trigger.assert_not_called()
    assert scheduler.status()["active"] is False


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
        with patch("ingest.worker.delete_source_points") as delete_mock:
            removed = worker.prune_missing_files()

        assert removed == [ghost_path]
        assert db.get_file_state(ghost_path) is None
        delete_mock.assert_called_once_with(
            worker.config.qdrant_url,
            worker.config.qdrant_collection,
            ghost_path,
            turbovec_url=worker.config.turbovec_url or None,
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

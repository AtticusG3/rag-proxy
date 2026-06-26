"""Tests for ingest stall detection and recovery."""

from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from ingest.db import IngestDatabase
from ingest.stall import is_stalled, stall_error_message
from ingest.worker import IngestConfig, IngestWorker


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


class TestStallDetection(unittest.TestCase):
    def test_is_stalled_when_old(self) -> None:
        old = (datetime.now(timezone.utc) - timedelta(minutes=20)).isoformat()
        self.assertTrue(is_stalled(old, 900))

    def test_is_not_stalled_when_recent(self) -> None:
        recent = datetime.now(timezone.utc).isoformat()
        self.assertFalse(is_stalled(recent, 900))

    def test_stall_message(self) -> None:
        msg = stall_error_message(stall_seconds=900, chunks_embedded=42)
        self.assertIn("42 chunks", msg)
        self.assertIn("15+", msg)


class TestStallRecovery(unittest.TestCase):
    def test_recover_interrupted_running_on_start(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "admin.sqlite")
            db = IngestDatabase(db_path)
            db.upsert_file_state("/zim/stuck.zim", status="running", chunks_embedded=99)
            worker = _worker(db)

            worker._recover_interrupted_running()

            row = db.get_file_state("/zim/stuck.zim")
            assert row is not None
            self.assertEqual(row["status"], "failed")
            self.assertIn("interrupted", row["last_error"] or "")
            self.assertIn("99 chunks", row["last_error"] or "")

    def test_fail_stalled_running(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "admin.sqlite")
            db = IngestDatabase(db_path)
            db.upsert_file_state("/zim/stuck.zim", status="running", chunks_embedded=12)
            old = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
            conn = sqlite3.connect(db_path)
            conn.execute(
                "UPDATE kb_ingest_state SET updated_at = ? WHERE file_path = ?",
                (old, "/zim/stuck.zim"),
            )
            conn.commit()
            conn.close()

            worker = _worker(db)
            worker._fail_stalled_running()

            row = db.get_file_state("/zim/stuck.zim")
            assert row is not None
            self.assertEqual(row["status"], "failed")
            self.assertIn("stalled", row["last_error"] or "")

    def test_restart_stalled_requeues_and_clears_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "admin.sqlite")
            db = IngestDatabase(db_path)
            db.upsert_file_state("/zim/stuck.zim", status="running", chunks_embedded=12)
            old = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
            conn = sqlite3.connect(db_path)
            conn.execute(
                "UPDATE kb_ingest_state SET updated_at = ? WHERE file_path = ?",
                (old, "/zim/stuck.zim"),
            )
            conn.commit()
            conn.close()

            worker = _worker(db)
            with patch("ingest.worker.delete_by_source"):
                worker.restart_stalled_files()

            row = db.get_file_state("/zim/stuck.zim")
            assert row is not None
            self.assertEqual(row["status"], "pending")
            self.assertEqual(row["chunks_embedded"], 0)


if __name__ == "__main__":
    unittest.main()

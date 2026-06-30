"""Tests for atomic SQLite pending-file claims."""

from __future__ import annotations

import os
import sqlite3
import tempfile
import threading
from unittest.mock import patch

from ingest.db import IngestDatabase, _sqlite_supports_returning


def _db() -> tuple[IngestDatabase, str]:
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "ingest.sqlite")
    return IngestDatabase(path), tmp


def test_claim_pending_file_marks_running() -> None:
    db, _tmp = _db()
    db.upsert_file_state("/data/a.txt", status="pending", file_type="text")
    claimed = db.claim_pending_file()
    assert claimed is not None
    assert claimed["file_path"] == "/data/a.txt"
    row = db.get_file_state("/data/a.txt")
    assert row is not None
    assert row["status"] == "running"
    assert row["started_at"]


def test_claim_pending_file_returns_none_when_queue_empty() -> None:
    db, _tmp = _db()
    assert db.claim_pending_file() is None


def test_claim_pending_file_is_exclusive_across_threads() -> None:
    db, _tmp = _db()
    paths = [f"/data/file-{index}.txt" for index in range(8)]
    for path in paths:
        db.upsert_file_state(path, status="pending", file_type="text")

    claimed_paths: list[str] = []
    lock = threading.Lock()
    barrier = threading.Barrier(4)

    def claim_once() -> None:
        barrier.wait(timeout=5.0)
        row = db.claim_pending_file()
        if row is not None:
            with lock:
                claimed_paths.append(str(row["file_path"]))

    threads = [threading.Thread(target=claim_once) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10.0)

    assert len(claimed_paths) == 4
    assert len(set(claimed_paths)) == 4
    for path in claimed_paths:
        row = db.get_file_state(path)
        assert row is not None
        assert row["status"] == "running"


def test_sqlite_returning_support_matches_runtime_version() -> None:
    parts = sqlite3.sqlite_version.split(".")
    major = int(parts[0])
    minor = int(parts[1]) if len(parts) > 1 else 0
    assert _sqlite_supports_returning() == ((major, minor) >= (3, 35))


def test_claim_pending_file_legacy_path_when_returning_unsupported() -> None:
    db, _tmp = _db()
    db.upsert_file_state("/data/legacy.txt", status="pending", file_type="text")
    with patch("ingest.db._sqlite_supports_returning", return_value=False):
        claimed = db.claim_pending_file()
    assert claimed is not None
    assert claimed["file_path"] == "/data/legacy.txt"
    row = db.get_file_state("/data/legacy.txt")
    assert row is not None
    assert row["status"] == "running"

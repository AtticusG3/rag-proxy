"""Tests for bench_ingest_host.py helpers."""

from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOST_PY = ROOT / "scripts" / "bench_ingest_host.py"


def _load_host():
    spec = importlib.util.spec_from_file_location("bench_ingest_host", HOST_PY)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _make_admin_db(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE admin_settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


def test_prepare_pause_sets_flag_when_not_paused(tmp_path: Path) -> None:
    host = _load_host()
    db = tmp_path / "admin.sqlite"
    _make_admin_db(db)
    assert host._read_paused(db) is False
    host._set_paused(db, False)
    host._set_paused(db, True)
    assert host._read_paused(db) is True


def test_restore_pause_restores_prior_state(tmp_path: Path) -> None:
    host = _load_host()
    db = tmp_path / "admin.sqlite"
    _make_admin_db(db)
    host._set_paused(db, False)
    host._set_paused(db, True)
    host._set_paused(db, False)
    assert host._read_paused(db) is False

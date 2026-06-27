"""Tests for persistent admin settings store."""

from __future__ import annotations

from pathlib import Path

from ingest.worker import IngestWorker
from rag_admin.db import AdminDatabase
from rag_admin.settings_store import SettingsStore


def test_save_group_writes_admin_env_and_hot_applies_worker(tmp_path: Path) -> None:
    db_path = tmp_path / "admin.sqlite"
    admin_env = tmp_path / "rag-admin.env"
    proxy_env = tmp_path / "rag-proxy.env"
    admin_env.write_text("INGEST_BATCH_SIZE=64\n", encoding="utf-8")

    db = AdminDatabase(str(db_path))
    store = SettingsStore(
        db,
        admin_env_path=str(admin_env),
        proxy_env_path=str(proxy_env),
    )
    worker = IngestWorker(
        store.build_ingest_config(zim_dir=str(tmp_path), upload_dir=str(tmp_path)),
        db.ingest,
    )
    assert worker.config.batch_size == 64

    store.save_group("ingest", {"INGEST_BATCH_SIZE": "96"})
    store.apply_to_worker(worker, zim_dir=str(tmp_path), upload_dir=str(tmp_path))
    assert worker.config.batch_size == 96
    assert read_env_batch_size(admin_env) == "96"


def test_ingest_pause_persists_in_sqlite(tmp_path: Path) -> None:
    db_path = tmp_path / "admin.sqlite"
    db = AdminDatabase(str(db_path))
    store = SettingsStore(
        db,
        admin_env_path=str(tmp_path / "admin.env"),
        proxy_env_path=str(tmp_path / "proxy.env"),
    )
    store.set_ingest_paused(True)
    store2 = SettingsStore(
        db,
        admin_env_path=str(tmp_path / "admin.env"),
        proxy_env_path=str(tmp_path / "proxy.env"),
    )
    assert store2.ingest_paused() is True


def read_env_batch_size(path: Path) -> str:
    from rag_admin.env_file import read_env_file

    return read_env_file(str(path))["INGEST_BATCH_SIZE"]

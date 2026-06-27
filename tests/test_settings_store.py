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


def test_ingest_save_mirrors_shared_keys_to_proxy_env(tmp_path: Path) -> None:
    from rag_admin.env_file import read_env_file

    db = AdminDatabase(str(tmp_path / "admin.sqlite"))
    admin_env = tmp_path / "rag-admin.env"
    proxy_env = tmp_path / "rag-proxy.env"
    store = SettingsStore(
        db,
        admin_env_path=str(admin_env),
        proxy_env_path=str(proxy_env),
    )
    store.save_group(
        "ingest",
        {
            "EMBED_URL": "http://127.0.0.1:18099",
            "QDRANT_URL": "http://qdrant:6333",
            "QDRANT_COLLECTION": "test_collection",
            "SPARSE_INDEX_URL": "http://127.0.0.1:18096",
        },
    )
    proxy = read_env_file(str(proxy_env))
    assert proxy["EMBED_URL"] == "http://127.0.0.1:18099"
    assert proxy["QDRANT_URL"] == "http://qdrant:6333"
    assert proxy["QDRANT_COLLECTION"] == "test_collection"
    assert proxy["SPARSE_INDEX_URL"] == "http://127.0.0.1:18096"


def test_proxy_save_writes_llama_swap_url(tmp_path: Path) -> None:
    from rag_admin.env_file import read_env_file

    db = AdminDatabase(str(tmp_path / "admin.sqlite"))
    proxy_env = tmp_path / "rag-proxy.env"
    store = SettingsStore(
        db,
        admin_env_path=str(tmp_path / "rag-admin.env"),
        proxy_env_path=str(proxy_env),
    )
    result = store.save_group("proxy_rag", {"LLAMA_SWAP_URL": "http://127.0.0.1:8787"})
    assert "LLAMA_SWAP_URL" in result.updated
    assert read_env_file(str(proxy_env))["LLAMA_SWAP_URL"] == "http://127.0.0.1:8787"
    assert result.restart_proxy is True


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

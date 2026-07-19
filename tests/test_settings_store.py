"""Tests for persistent admin settings store."""

from __future__ import annotations

from pathlib import Path

from ingest.worker import IngestWorker
from rag_admin.db import AdminDatabase
from rag_admin.env_file import read_env_file
from rag_admin.settings_schema import SETTING_FIELDS
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


def test_memgraph_build_params_use_schema_defaults(tmp_path: Path) -> None:
    db = AdminDatabase(str(tmp_path / "admin.sqlite"))
    store = SettingsStore(
        db,
        admin_env_path=str(tmp_path / "admin.env"),
        proxy_env_path=str(tmp_path / "proxy.env"),
    )
    params = store.memgraph_build_params()
    assert params["llm_url"] == "http://192.168.1.202:8081/v1"
    assert params["llm_model"] == "qwen3.5-9b-turbo"
    assert "8787" not in params["llm_url"]


def test_clear_override_reverts_to_schema_default(tmp_path: Path) -> None:
    admin_env = tmp_path / "rag-admin.env"
    admin_env.write_text("INGEST_BATCH_SIZE=96\n", encoding="utf-8")
    store = SettingsStore(
        AdminDatabase(str(tmp_path / "admin.sqlite")),
        admin_env_path=str(admin_env),
        proxy_env_path=str(tmp_path / "rag-proxy.env"),
        pool_env_path=str(tmp_path / "pool.env"),
        pool_scale_env_path=str(tmp_path / "pool-scale.env"),
    )
    field = next(f for f in SETTING_FIELDS if f.key == "INGEST_BATCH_SIZE")
    assert store.get_override_value(field.key, target=field.target) == "96"
    store.save_group("ingest", {"INGEST_BATCH_SIZE": ""})
    assert store.get_value("INGEST_BATCH_SIZE") == "64"
    assert store.get_override_value(field.key, target=field.target) is None
    assert "INGEST_BATCH_SIZE" not in read_env_file(str(admin_env))


def test_save_ingest_file_concurrency(tmp_path: Path) -> None:
    store = SettingsStore(
        AdminDatabase(str(tmp_path / "admin.sqlite")),
        admin_env_path=str(tmp_path / "admin.env"),
        proxy_env_path=str(tmp_path / "proxy.env"),
        pool_env_path=str(tmp_path / "pool.env"),
        pool_scale_env_path=str(tmp_path / "pool-scale.env"),
    )
    store.save_group("ingest", {"INGEST_FILE_CONCURRENCY": "3"})
    config = store.build_ingest_config(zim_dir=str(tmp_path), upload_dir=str(tmp_path))
    assert config.file_concurrency == 3
    store.save_group("ingest", {"INGEST_FILE_CONCURRENCY": ""})
    config = store.build_ingest_config(zim_dir=str(tmp_path), upload_dir=str(tmp_path))
    assert config.file_concurrency is None


def test_save_pool_scale_fields_writes_scale_env(tmp_path: Path) -> None:
    scale_env = tmp_path / "nomic-embed-scale.env"
    store = SettingsStore(
        AdminDatabase(str(tmp_path / "admin.sqlite")),
        admin_env_path=str(tmp_path / "admin.env"),
        proxy_env_path=str(tmp_path / "proxy.env"),
        pool_scale_env_path=str(scale_env),
        pool_env_path=str(tmp_path / "pool.env"),
    )
    store.save_group("ingest", {"NOMIC_POOL_MAX_INSTANCES": "8", "NOMIC_POOL_PARALLEL": "12"})
    written = read_env_file(str(scale_env))
    assert written["NOMIC_POOL_MAX_INSTANCES"] == "8"
    assert written["NOMIC_POOL_PARALLEL"] == "12"


def test_save_ingest_chunk_settings_updates_config(tmp_path: Path) -> None:
    db = AdminDatabase(str(tmp_path / "admin.sqlite"))
    store = SettingsStore(
        db,
        admin_env_path=str(tmp_path / "admin.env"),
        proxy_env_path=str(tmp_path / "proxy.env"),
    )
    store.save_group(
        "ingest",
        {
            "INGEST_CHUNK_SIZE_TOKENS": "256",
            "INGEST_CHUNK_OVERLAP_TOKENS": "32",
            "INGEST_CHUNK_TOKENIZER": "gpt2",
            "INGEST_CHUNK_SEMANTIC": "false",
            "INGEST_CHUNK_SEMANTIC_MODEL": "minishlab/potion-base-32M",
            "INGEST_CHUNK_MIN_TOKENS": "50",
        },
    )
    config = store.build_ingest_config(zim_dir=str(tmp_path), upload_dir=str(tmp_path))
    assert config.chunk_config.chunk_size == 256
    assert config.chunk_config.chunk_overlap == 32
    assert config.chunk_config.tokenizer == "gpt2"
    assert config.chunk_config.semantic_enabled is False
    assert config.chunk_config.min_chunk_tokens == 50


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

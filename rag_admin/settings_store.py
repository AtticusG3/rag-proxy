"""Persistent settings store: SQLite + env files."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from ingest.chunking import chunk_config_from_values
from ingest.embed_urls import parse_ingest_embed_urls
from ingest.worker import IngestConfig, IngestWorker
from rag_admin.db import AdminDatabase
from rag_admin.env_file import read_env_file, remove_env_file_keys, write_env_file
from rag_admin.settings_schema import (
    INGEST_MIRROR_TO_PROXY,
    INGEST_PAUSED_KEY,
    SETTING_FIELDS,
    SettingField,
)

# Keys the capacity planner writes to the pool env; synced into admin env on scale.
POOL_OUTPUT_KEYS: frozenset[str] = frozenset(
    {
        "INGEST_EMBED_URLS",
        "INGEST_EMBED_CONCURRENCY",
        "INGEST_FILE_CONCURRENCY",
        "INGEST_BATCH_SIZE",
        "INGEST_CHUNK_CONCURRENCY",
        "INGEST_CHUNK_SEMANTIC",
        "INGEST_SPARSE_REINDEX",
    }
)
from rag_proxy.env_parse import parse_bool


@dataclass
class SaveResult:
    group: str
    updated: list[str]
    restart_proxy: bool
    restart_admin: bool
    pool_scale_updated: bool = False


def _field_by_key(key: str) -> SettingField | None:
    for field in SETTING_FIELDS:
        if field.key == key:
            return field
    return None


def _coerce_value(field: SettingField, raw: str) -> str:
    value = raw.strip()
    if field.field_type == "bool":
        return "true" if value.lower() in ("1", "true", "yes", "on") else "false"
    if field.field_type == "int":
        return str(int(value))
    if field.field_type == "float":
        return str(float(value))
    if field.field_type == "select" and field.options and value not in field.options:
        raise ValueError(f"{field.key} must be one of {', '.join(field.options)}")
    return value


def _parse_bool(raw: str | None) -> bool:
    return parse_bool(raw, False)


class SettingsStore:
    """Merge env files, SQLite overrides, and hot-apply ingest settings."""

    def __init__(
        self,
        db: AdminDatabase,
        *,
        admin_env_path: str,
        proxy_env_path: str,
        pool_scale_env_path: str = "/opt/ai/config/nomic-embed-scale.env",
        pool_env_path: str = "/opt/ai/config/nomic-embed-pool.env",
        env_example_path: str = "",
    ) -> None:
        self.db = db
        self.admin_env_path = admin_env_path
        self.proxy_env_path = proxy_env_path
        self.pool_scale_env_path = pool_scale_env_path
        self.pool_env_path = pool_env_path
        self.env_example_path = env_example_path

    def _pool_scale_env(self) -> dict[str, str]:
        return read_env_file(self.pool_scale_env_path)

    def get_override_value(self, key: str, *, target: str) -> str | None:
        """Return an explicit override from SQLite or env files, or None if unset."""
        stored = self.db.get_setting(key)
        if stored is not None:
            return stored
        if target == "pool_scale":
            scale_env = self._pool_scale_env()
            if key in scale_env:
                return scale_env[key]
            return None
        if target in ("admin", "sqlite"):
            admin_env = read_env_file(self.admin_env_path)
            if key in admin_env:
                return admin_env[key]
        if target in ("admin", "proxy"):
            proxy_env = read_env_file(self.proxy_env_path)
            if key in proxy_env:
                return proxy_env[key]
        example = self._example_env()
        if key in example:
            return example[key]
        return None

    def _example_env(self) -> dict[str, str]:
        if not self.env_example_path:
            return {}
        cache = getattr(self, "_example_env_cache", None)
        if cache is None:
            cache = read_env_file(self.env_example_path)
            self._example_env_cache = cache
        return cache

    def has_override(self, field: SettingField) -> bool:
        return self.get_override_value(field.key, target=field.target) is not None

    def get_value(self, key: str, default: str = "") -> str:
        stored = self.db.get_setting(key)
        if stored is not None:
            return stored
        admin_env = read_env_file(self.admin_env_path)
        if key in admin_env:
            return admin_env[key]
        proxy_env = read_env_file(self.proxy_env_path)
        if key in proxy_env:
            return proxy_env[key]
        if key in POOL_OUTPUT_KEYS:
            pool_env = read_env_file(self.pool_env_path)
            if key in pool_env:
                return pool_env[key]
        field = _field_by_key(key)
        if field is not None and field.target == "pool_scale":
            scale_env = self._pool_scale_env()
            if key in scale_env:
                return scale_env[key]
        env_val = os.getenv(key)
        if env_val is not None and env_val != "":
            return env_val
        example = self._example_env()
        if key in example:
            return example[key]
        field = _field_by_key(key)
        if field is not None:
            return field.default
        return os.getenv(key, default)

    def get_group_values(self, group: str) -> dict[str, str]:
        return {
            field.key: self.get_value(field.key, field.default)
            for field in SETTING_FIELDS
            if field.group == group
        }

    def all_field_values(self) -> dict[str, str]:
        return {field.key: self.get_value(field.key, field.default) for field in SETTING_FIELDS}

    def ingest_paused(self) -> bool:
        return _parse_bool(self.db.get_setting(INGEST_PAUSED_KEY))

    def set_ingest_paused(self, paused: bool) -> None:
        self.db.set_setting(INGEST_PAUSED_KEY, "true" if paused else "false")

    def build_ingest_config(self, *, zim_dir: str, upload_dir: str) -> IngestConfig:
        values = self.all_field_values()
        embed_url = values.get("EMBED_URL", "http://127.0.0.1:8089")
        ingest_urls_raw = values.get("INGEST_EMBED_URLS", "").strip()
        return IngestConfig(
            zim_dir=zim_dir,
            upload_dir=upload_dir,
            embed_url=embed_url,
            embed_urls=parse_ingest_embed_urls(
                embed_url=embed_url,
                ingest_embed_urls=ingest_urls_raw or None,
            ),
            qdrant_url=values.get("QDRANT_URL", "http://127.0.0.1:6333"),
            qdrant_collection=values.get("QDRANT_COLLECTION", "nomad_knowledge_base"),
            sparse_index_url=values.get("SPARSE_INDEX_URL", ""),
            batch_size=int(values.get("INGEST_BATCH_SIZE", "64")),
            embed_concurrency=int(values.get("INGEST_EMBED_CONCURRENCY", "4")),
            max_articles=int(values.get("INGEST_MAX_ARTICLES", "0")),
            embed_max_chars=int(values.get("EMBED_MAX_CHARS", "2000")),
            sparse_reindex_mode=values.get("INGEST_SPARSE_REINDEX", "idle").lower(),
            stall_seconds=int(values.get("INGEST_STALL_MINUTES", "15")) * 60,
            file_concurrency=int(values["INGEST_FILE_CONCURRENCY"])
            if values.get("INGEST_FILE_CONCURRENCY", "").strip()
            else None,
            chunk_concurrency=int(values["INGEST_CHUNK_CONCURRENCY"])
            if values.get("INGEST_CHUNK_CONCURRENCY", "").strip()
            else None,
            chunk_config=chunk_config_from_values(values),
        )

    def apply_to_worker(self, worker: IngestWorker, *, zim_dir: str, upload_dir: str) -> None:
        worker.update_config(self.build_ingest_config(zim_dir=zim_dir, upload_dir=upload_dir))
        worker.set_paused(self.ingest_paused())

    def save_group(self, group: str, form: dict[str, str]) -> SaveResult:
        fields = [field for field in SETTING_FIELDS if field.group == group]
        if not fields:
            raise ValueError(f"unknown settings group: {group}")

        admin_updates: dict[str, str] = {}
        proxy_updates: dict[str, str] = {}
        sqlite_updates: dict[str, str] = {}
        pool_scale_updates: dict[str, str] = {}
        unset_admin: set[str] = set()
        unset_proxy: set[str] = set()
        unset_pool_scale: set[str] = set()
        updated_keys: list[str] = []

        clearable_types = frozenset({"int", "float", "str", "text"})

        for field in fields:
            if field.key not in form:
                continue
            raw = str(form[field.key]).strip()
            if raw == "" and field.field_type in clearable_types:
                updated_keys.append(field.key)
                self.db.delete_setting(field.key)
                if field.target == "admin":
                    unset_admin.add(field.key)
                elif field.target == "proxy":
                    unset_proxy.add(field.key)
                elif field.target == "pool_scale":
                    unset_pool_scale.add(field.key)
                continue
            value = _coerce_value(field, raw)
            updated_keys.append(field.key)
            if field.target == "admin":
                admin_updates[field.key] = value
            elif field.target == "proxy":
                proxy_updates[field.key] = value
            elif field.target == "pool_scale":
                pool_scale_updates[field.key] = value
            else:
                sqlite_updates[field.key] = value

        if unset_pool_scale:
            remove_env_file_keys(self.pool_scale_env_path, unset_pool_scale)
        if pool_scale_updates:
            write_env_file(self.pool_scale_env_path, pool_scale_updates)
        if unset_admin:
            remove_env_file_keys(self.admin_env_path, unset_admin)
            mirror_unset = unset_admin & set(INGEST_MIRROR_TO_PROXY)
            if mirror_unset:
                remove_env_file_keys(self.proxy_env_path, mirror_unset)
                unset_proxy |= mirror_unset
        if unset_proxy:
            remove_env_file_keys(self.proxy_env_path, unset_proxy)

        if admin_updates:
            write_env_file(self.admin_env_path, admin_updates)
            mirror = {
                key: value
                for key, value in admin_updates.items()
                if key in INGEST_MIRROR_TO_PROXY
            }
            if mirror:
                write_env_file(self.proxy_env_path, mirror)
                proxy_updates = {**proxy_updates, **mirror}
        if proxy_updates:
            write_env_file(self.proxy_env_path, proxy_updates)
        for key, value in {**admin_updates, **proxy_updates, **sqlite_updates}.items():
            self.db.set_setting(key, value)

        restart_proxy = bool(proxy_updates)
        restart_admin = any(
            field.key in admin_updates and not field.hot for field in fields
        )

        return SaveResult(
            group=group,
            updated=updated_keys,
            restart_proxy=restart_proxy,
            restart_admin=restart_admin,
            pool_scale_updated=bool(pool_scale_updates or unset_pool_scale),
        )

    def pool_env_snapshot(self) -> dict[str, Any]:
        env = read_env_file(self.pool_env_path)
        return {
            "path": self.pool_env_path,
            "scale_path": self.pool_scale_env_path,
            "exists": os.path.isfile(self.pool_env_path),
            "instance_count": env.get("NOMIC_POOL_INSTANCE_COUNT", ""),
            "ports": env.get("NOMIC_POOL_PORTS", ""),
            "embed_urls": env.get("INGEST_EMBED_URLS", ""),
            "embed_concurrency": env.get("INGEST_EMBED_CONCURRENCY", ""),
            "gpu_free_mib": env.get("NOMIC_POOL_GPU_FREE_MIB", ""),
            "file_concurrency": env.get("INGEST_FILE_CONCURRENCY", ""),
            "batch_size": env.get("INGEST_BATCH_SIZE", ""),
            "chunk_concurrency": env.get("INGEST_CHUNK_CONCURRENCY", ""),
            "chunk_semantic": env.get("INGEST_CHUNK_SEMANTIC", ""),
            "sparse_reindex": env.get("INGEST_SPARSE_REINDEX", ""),
            "cpu_cores": env.get("CAPACITY_CPU_CORES", ""),
            "cpu_model": env.get("CAPACITY_CPU_MODEL", ""),
            "ram_available_mib": env.get("CAPACITY_RAM_AVAILABLE_MIB", ""),
            "gpu_name": env.get("CAPACITY_GPU_NAME", ""),
            "probed_at": env.get("CAPACITY_PROBED_AT", ""),
        }

    def sync_pool_ingest_from_pool_env(self) -> list[str]:
        """Persist pool planner outputs into admin env for hot reload."""
        pool_env = read_env_file(self.pool_env_path)
        updates = {
            key: pool_env[key]
            for key in POOL_OUTPUT_KEYS
            if pool_env.get(key, "").strip()
        }
        if not updates:
            return []
        write_env_file(self.admin_env_path, updates)
        for key, value in updates.items():
            self.db.set_setting(key, value)
        return list(updates.keys())

    def embed_pool_scale_params(self) -> dict[str, str]:
        return {
            "pool_env_path": self.pool_env_path,
            "scale_env_path": self.pool_scale_env_path,
            "semantic_requested": self.get_value("INGEST_CHUNK_SEMANTIC", "true"),
        }

    def memgraph_build_params(self) -> dict[str, Any]:
        values = self.get_group_values("memgraphrag")
        build = self.get_group_values("memgraph_build")
        output = values.get("MEMGRAPHRAG_DB_PATH", "/var/lib/rag_proxy/memgraphrag.sqlite")
        embed = build.get("MEMGRAPH_BUILD_EMBED_URL") or self.get_value("EMBED_URL", "")
        return {
            "source": "qdrant",
            "qdrant_url": self.get_value("QDRANT_URL", "http://127.0.0.1:6333"),
            "collection": self.get_value("QDRANT_COLLECTION", "nomad_knowledge_base"),
            "output": output,
            "llm_url": self.get_value("MEMGRAPH_BUILD_LLM_URL"),
            "llm_model": self.get_value("MEMGRAPH_BUILD_LLM_MODEL"),
            "max_chunks": int(build.get("MEMGRAPH_BUILD_MAX_CHUNKS", "1000")),
            "concurrency": int(build.get("MEMGRAPH_BUILD_CONCURRENCY", "3")),
            "embed_url": embed,
            "skip_relations": _parse_bool(build.get("MEMGRAPH_BUILD_SKIP_RELATIONS", "false")),
        }

    def service_snapshot(self) -> dict[str, Any]:
        return {
            "admin_env_path": self.admin_env_path,
            "proxy_env_path": self.proxy_env_path,
            "ingest_paused": self.ingest_paused(),
            "memgraph_db_path": self.get_value(
                "MEMGRAPHRAG_DB_PATH",
                "/var/lib/rag_proxy/memgraphrag.sqlite",
            ),
            "rag_proxy_url": self.get_value("RAG_PROXY_URL", "http://127.0.0.1:8081"),
        }

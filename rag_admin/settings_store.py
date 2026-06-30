"""Persistent settings store: SQLite + env files."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from ingest.chunk_config import chunk_config_from_values
from ingest.embed_urls import parse_ingest_embed_urls
from ingest.worker import IngestConfig, IngestWorker
from rag_admin.db import AdminDatabase
from rag_admin.env_file import read_env_file, write_env_file
from rag_admin.settings_schema import (
    INGEST_MIRROR_TO_PROXY,
    INGEST_PAUSED_KEY,
    SETTING_FIELDS,
    SettingField,
)


@dataclass
class SaveResult:
    group: str
    updated: list[str]
    restart_proxy: bool
    restart_admin: bool


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
    return (raw or "").strip().lower() in ("1", "true", "yes", "on")


class SettingsStore:
    """Merge env files, SQLite overrides, and hot-apply ingest settings."""

    def __init__(
        self,
        db: AdminDatabase,
        *,
        admin_env_path: str,
        proxy_env_path: str,
    ) -> None:
        self.db = db
        self.admin_env_path = admin_env_path
        self.proxy_env_path = proxy_env_path

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
        env_val = os.getenv(key)
        if env_val is not None and env_val != "":
            return env_val
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
        updated_keys: list[str] = []

        for field in fields:
            if field.key not in form:
                continue
            value = _coerce_value(field, form[field.key])
            updated_keys.append(field.key)
            if field.target == "admin":
                admin_updates[field.key] = value
            elif field.target == "proxy":
                proxy_updates[field.key] = value
            else:
                sqlite_updates[field.key] = value

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
        )

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

"""RAG admin configuration from environment."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return int(raw)


@dataclass(frozen=True)
class AdminSettings:
    host: str
    port: int
    db_path: str
    zim_dir: str
    upload_dir: str
    embed_url: str
    qdrant_url: str
    qdrant_collection: str
    sparse_index_url: str
    batch_size: int
    max_articles: int
    embed_max_chars: int
    sparse_reindex_mode: str
    stall_seconds: int
    session_secret: str
    password: str
    rag_proxy_url: str

    @classmethod
    def from_env(cls) -> AdminSettings:
        return cls(
            host=os.getenv("ADMIN_HOST", "127.0.0.1"),
            port=_env_int("ADMIN_PORT", 8087),
            db_path=os.getenv("ADMIN_DB_PATH", "/opt/ai/rag/admin.sqlite"),
            zim_dir=os.getenv("ZIM_DIR", "/opt/ai/rag/zim"),
            upload_dir=os.getenv("UPLOAD_DIR", "/opt/ai/rag/uploads"),
            embed_url=os.getenv("EMBED_URL", "http://127.0.0.1:18089"),
            qdrant_url=os.getenv("QDRANT_URL", "http://127.0.0.1:6333"),
            qdrant_collection=os.getenv("QDRANT_COLLECTION", "nomad_knowledge_base"),
            sparse_index_url=os.getenv("SPARSE_INDEX_URL", "http://127.0.0.1:18096"),
            batch_size=_env_int("INGEST_BATCH_SIZE", 64),
            max_articles=_env_int("INGEST_MAX_ARTICLES", 0),
            embed_max_chars=_env_int("EMBED_MAX_CHARS", 2000),
            sparse_reindex_mode=os.getenv("INGEST_SPARSE_REINDEX", "idle").strip().lower(),
            stall_seconds=_env_int("INGEST_STALL_MINUTES", 15) * 60,
            session_secret=os.getenv("ADMIN_SESSION_SECRET", "change-me-in-production"),
            password=os.getenv("ADMIN_PASSWORD", "changeme"),
            rag_proxy_url=os.getenv("RAG_PROXY_URL", "http://127.0.0.1:8081"),
        )


settings = AdminSettings.from_env()

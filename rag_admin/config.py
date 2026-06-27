"""RAG admin configuration from environment."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_SESSION_SECRET = "change-me-in-production"
DEFAULT_PASSWORD = "changeme"


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
    ingest_embed_urls: str
    qdrant_url: str
    qdrant_collection: str
    sparse_index_url: str
    batch_size: int
    embed_concurrency: int
    max_articles: int
    embed_max_chars: int
    sparse_reindex_mode: str
    stall_seconds: int
    session_secret: str
    password: str
    rag_proxy_url: str
    admin_env_path: str
    proxy_env_path: str
    repo_root: str
    job_log_dir: str

    @classmethod
    def from_env(cls) -> AdminSettings:
        return cls(
            host=os.getenv("ADMIN_HOST", "127.0.0.1"),
            port=_env_int("ADMIN_PORT", 8087),
            db_path=os.getenv("ADMIN_DB_PATH", "/opt/ai/rag/admin.sqlite"),
            zim_dir=os.getenv("ZIM_DIR", "/opt/ai/rag/zim"),
            upload_dir=os.getenv("UPLOAD_DIR", "/opt/ai/rag/uploads"),
            embed_url=os.getenv("EMBED_URL", "http://127.0.0.1:18089"),
            ingest_embed_urls=os.getenv("INGEST_EMBED_URLS", "").strip(),
            qdrant_url=os.getenv("QDRANT_URL", "http://127.0.0.1:6333"),
            qdrant_collection=os.getenv("QDRANT_COLLECTION", "nomad_knowledge_base"),
            sparse_index_url=os.getenv("SPARSE_INDEX_URL", "http://127.0.0.1:18096"),
            batch_size=_env_int("INGEST_BATCH_SIZE", 64),
            embed_concurrency=_env_int("INGEST_EMBED_CONCURRENCY", 4),
            max_articles=_env_int("INGEST_MAX_ARTICLES", 0),
            embed_max_chars=_env_int("EMBED_MAX_CHARS", 2000),
            sparse_reindex_mode=os.getenv("INGEST_SPARSE_REINDEX", "idle").strip().lower(),
            stall_seconds=_env_int("INGEST_STALL_MINUTES", 15) * 60,
            session_secret=os.getenv("ADMIN_SESSION_SECRET", DEFAULT_SESSION_SECRET),
            password=os.getenv("ADMIN_PASSWORD", DEFAULT_PASSWORD),
            rag_proxy_url=os.getenv("RAG_PROXY_URL", "http://127.0.0.1:8081"),
            admin_env_path=os.getenv(
                "RAG_ADMIN_ENV_FILE",
                "/opt/ai/config/rag-admin.env",
            ),
            proxy_env_path=os.getenv(
                "RAG_PROXY_ENV_FILE",
                "/opt/ai/config/rag-proxy.env",
            ),
            repo_root=os.getenv("RAG_REPO_ROOT", str(Path(__file__).resolve().parents[1])),
            job_log_dir=os.getenv(
                "RAG_ADMIN_JOB_LOG_DIR",
                "/var/lib/rag_proxy/admin_jobs",
            ),
        )


def _path_is_under_root(path: Path, root: Path) -> bool:
    try:
        common = os.path.commonpath([str(path), str(root)])
    except ValueError:
        return False
    return common == str(root)


def resolve_ingest_path(file_path: str, *, zim_dir: str, upload_dir: str) -> Path:
    """Resolve file_path and ensure it lies under zim_dir or upload_dir."""
    try:
        resolved = Path(file_path).resolve()
    except OSError as exc:
        raise ValueError(f"Invalid file path: {file_path}") from exc
    roots = (Path(zim_dir).resolve(), Path(upload_dir).resolve())
    for root in roots:
        if _path_is_under_root(resolved, root):
            return resolved
    raise ValueError(f"file_path must be under {zim_dir} or {upload_dir}")


def validate_settings(s: AdminSettings) -> None:
    """Refuse insecure default secrets unless explicitly allowed for local dev."""
    allow = os.getenv("ADMIN_ALLOW_INSECURE_DEFAULTS", "").strip().lower()
    if allow in ("true", "1", "yes"):
        return
    problems: list[str] = []
    if s.session_secret == DEFAULT_SESSION_SECRET:
        problems.append("ADMIN_SESSION_SECRET is still the default placeholder")
    if s.password == DEFAULT_PASSWORD:
        problems.append("ADMIN_PASSWORD is still the default placeholder")
    if problems:
        raise RuntimeError(
            "rag-admin refused to start: "
            + "; ".join(problems)
            + ". Set secure values or ADMIN_ALLOW_INSECURE_DEFAULTS=true for local dev only."
        )


settings = AdminSettings.from_env()

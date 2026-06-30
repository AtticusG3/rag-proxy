#!/usr/bin/env python3
"""Re-queue all ingest files (clears Qdrant source points, resets job state)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ENV_FILE = os.getenv("RAG_ADMIN_ENV_FILE", "/opt/ai/config/rag-admin.env")


def _load_env_file(path: str) -> None:
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


def main() -> int:
    _load_env_file(ENV_FILE)

    from ingest.db import IngestDatabase
    from ingest.embed_urls import parse_ingest_embed_urls
    from ingest.worker import IngestConfig, IngestWorker

    db_path = os.getenv("ADMIN_DB_PATH", "/opt/ai/rag/admin.sqlite")
    embed_url = os.getenv("EMBED_URL", "http://127.0.0.1:18089")
    config = IngestConfig(
        zim_dir=os.getenv("ZIM_DIR", "/opt/ai/rag/zim"),
        upload_dir=os.getenv("UPLOAD_DIR", "/opt/ai/rag/uploads"),
        embed_url=embed_url,
        qdrant_url=os.getenv("QDRANT_URL", "http://127.0.0.1:6333"),
        qdrant_collection=os.getenv("QDRANT_COLLECTION", "nomad_knowledge_base"),
        sparse_index_url=os.getenv("SPARSE_INDEX_URL", ""),
        batch_size=int(os.getenv("INGEST_BATCH_SIZE", "64")),
        embed_concurrency=int(os.getenv("INGEST_EMBED_CONCURRENCY", "4")),
        max_articles=int(os.getenv("INGEST_MAX_ARTICLES", "0")),
        embed_max_chars=int(os.getenv("EMBED_MAX_CHARS", "2000")),
        sparse_reindex_mode=os.getenv("INGEST_SPARSE_REINDEX", "idle").lower(),
        stall_seconds=int(os.getenv("INGEST_STALL_MINUTES", "15")) * 60,
        embed_urls=parse_ingest_embed_urls(embed_url=embed_url),
    )
    worker = IngestWorker(config, IngestDatabase(db_path))
    job_id = worker.requeue_all_files()
    print(f"[ok] requeue_all job={job_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

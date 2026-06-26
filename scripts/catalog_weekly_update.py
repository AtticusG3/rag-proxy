#!/usr/bin/env python3
"""Weekly catalog update: check subscriptions and download updates."""

from __future__ import annotations

import os
import sys
from pathlib import Path


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
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    _load_env_file(os.getenv("RAG_ADMIN_ENV", "/opt/ai/config/rag-admin.env"))

    from ingest.worker import IngestConfig, IngestWorker
    from rag_admin.catalog.download_manager import CatalogDownloadManager
    from rag_admin.config import AdminSettings
    from rag_admin.db import AdminDatabase

    settings = AdminSettings.from_env()
    db = AdminDatabase(settings.db_path)
    config = IngestConfig(
        zim_dir=settings.zim_dir,
        upload_dir=settings.upload_dir,
        embed_url=settings.embed_url,
        qdrant_url=settings.qdrant_url,
        qdrant_collection=settings.qdrant_collection,
        sparse_index_url=settings.sparse_index_url,
        batch_size=settings.batch_size,
        max_articles=settings.max_articles,
        embed_max_chars=settings.embed_max_chars,
        sparse_reindex_mode=settings.sparse_reindex_mode,
    )
    worker = IngestWorker(config, db.ingest)
    worker.start()
    catalog = CatalogDownloadManager(db, settings.zim_dir, settings.upload_dir, worker)

    queued = catalog.check_updates()
    print(f"[catalog] update check queued {len(queued)} package(s)")

    downloaded = catalog.process_pending_downloads(max_items=20)
    print(f"[catalog] processed {downloaded} download(s)")

    worker.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

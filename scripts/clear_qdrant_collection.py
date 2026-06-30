#!/usr/bin/env python3
"""Drop and recreate the ingest Qdrant collection (clears all dense vectors)."""

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

    from ingest.qdrant_writer import clear_collection, get_collection_count

    qdrant_url = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
    collection = os.getenv("QDRANT_COLLECTION", "nomad_knowledge_base")

    before = 0
    try:
        before = get_collection_count(qdrant_url, collection)
    except Exception:
        before = 0

    removed = clear_collection(qdrant_url, collection)
    after = get_collection_count(qdrant_url, collection)

    print(
        f"[ok] cleared Qdrant collection={collection} url={qdrant_url} "
        f"removed={removed} (count_before={before}) points_now={after}"
    )
    print("[note] re-run scripts/requeue_all_ingest.py if ingest queue is not pending")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

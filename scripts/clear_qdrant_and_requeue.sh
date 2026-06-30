#!/usr/bin/env bash
# Clear all dense vectors from the ingest Qdrant collection and re-queue ingest.
set -euo pipefail

REPO="${REPO_ROOT:-/opt/ai/repo/rag_proxy}"
ENV_FILE="${RAG_ADMIN_ENV_FILE:-/opt/ai/config/rag-admin.env}"
PYTHON=/opt/ai/venv/bin/python

echo "[stop] rag-admin ingest worker"
pkill -f "python -m rag_admin" 2>/dev/null || true
sleep 2

echo "[clear] Qdrant + requeue via Python (safe env load)"
PYTHONPATH="$REPO" "$PYTHON" <<PY
import os
import sys
from pathlib import Path

repo = Path("$REPO")
sys.path.insert(0, str(repo))
env_file = os.environ.get("RAG_ADMIN_ENV_FILE", "$ENV_FILE")
if os.path.isfile(env_file):
    with open(env_file, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

import httpx

base = os.environ.get("QDRANT_URL", "http://127.0.0.1:6333").rstrip("/")
col = os.environ.get("QDRANT_COLLECTION", "nomad_knowledge_base")
with httpx.Client(timeout=120.0) as client:
    info = client.get(f"{base}/collections/{col}")
    prior = int(info.json()["result"]["points_count"]) if info.status_code == 200 else 0
    client.delete(f"{base}/collections/{col}")
    client.put(
        f"{base}/collections/{col}",
        json={"vectors": {"size": 768, "distance": "Cosine"}},
    )
    after = int(client.get(f"{base}/collections/{col}").json()["result"]["points_count"])
print(f"[ok] cleared {col} removed={prior} points_now={after}")

from ingest.db import IngestDatabase
from ingest.embed_urls import parse_ingest_embed_urls
from ingest.worker import IngestConfig, IngestWorker

embed_url = os.environ.get("EMBED_URL", "http://127.0.0.1:18089")
config = IngestConfig(
    zim_dir=os.environ.get("ZIM_DIR", "/opt/ai/rag/zim"),
    upload_dir=os.environ.get("UPLOAD_DIR", "/opt/ai/rag/uploads"),
    embed_url=embed_url,
    qdrant_url=base,
    qdrant_collection=col,
    sparse_index_url=os.environ.get("SPARSE_INDEX_URL", ""),
    batch_size=int(os.environ.get("INGEST_BATCH_SIZE", "64")),
    embed_concurrency=int(os.environ.get("INGEST_EMBED_CONCURRENCY", "4")),
    max_articles=int(os.environ.get("INGEST_MAX_ARTICLES", "0")),
    embed_max_chars=int(os.environ.get("EMBED_MAX_CHARS", "2000")),
    sparse_reindex_mode=os.environ.get("INGEST_SPARSE_REINDEX", "idle").lower(),
    stall_seconds=int(os.environ.get("INGEST_STALL_MINUTES", "15")) * 60,
    embed_urls=parse_ingest_embed_urls(embed_url=embed_url),
)
worker = IngestWorker(config, IngestDatabase(os.environ.get("ADMIN_DB_PATH", "/opt/ai/rag/admin.sqlite")))
job_id = worker.requeue_all_files()
print(f"[ok] requeue_all job={job_id}")
PY

echo "[start] rag-admin"
bash "$REPO/scripts/start_rag_admin.sh" 2>/dev/null || bash /tmp/start_rag_admin.sh

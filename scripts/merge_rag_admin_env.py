#!/usr/bin/env python3
"""Merge recommended keys into /opt/ai/config/rag-admin.env (run on buster)."""
from __future__ import annotations

from pathlib import Path

TARGET = Path("/opt/ai/config/rag-admin.env")
UPDATES = {
    "INGEST_BATCH_SIZE": "32",
    "INGEST_EMBED_CONCURRENCY": "16",
    "INGEST_FILE_CONCURRENCY": "1",
    "QDRANT_UPSERT_TIMEOUT_SEC": "300",
    "QDRANT_UPSERT_RETRIES": "6",
    "QDRANT_UPSERT_BACKOFF_SEC": "2",
    "INGEST_CAPACITY_QDRANT_RAM_BUDGET_MIB": "8192",
    "INGEST_CAPACITY_QDRANT_LARGE_COLLECTION_POINTS": "500000",
    "INGEST_CAPACITY_QDRANT_HUGE_COLLECTION_POINTS": "2000000",
    "SIDECAR_ON_DEMAND": "true",
    "SIDECAR_STARTUP_TIMEOUT_SEC": "900",
    "SIDECAR_RERANK_STARTUP_TIMEOUT_SEC": "180",
    "SIDECAR_LIFECYCLE_POLL_SEC": "20",
    "SPARSE_SIDECAR_UNIT": "sparse-sidecar.service",
    "RERANK_SIDECAR_UNIT": "rerank-sidecar.service",
}


def main() -> None:
    lines = TARGET.read_text(encoding="utf-8").splitlines() if TARGET.is_file() else []
    existing = {}
    for line in lines:
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        existing[key.strip()] = value.strip()
    existing.update(UPDATES)
    out = [f"{key}={value}" for key, value in sorted(existing.items())]
    TARGET.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"updated {TARGET} ({len(UPDATES)} keys merged)")


if __name__ == "__main__":
    main()

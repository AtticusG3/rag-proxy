#!/usr/bin/env python3
"""Post-rollout inspection for buster (run on host)."""
from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import urllib.request
from pathlib import Path

ADMIN_DB = "/opt/ai/rag/admin.sqlite"
QDRANT = "http://127.0.0.1:6333"
COLLECTION = "nomad_knowledge_base"
ADMIN_ENV = Path("/opt/ai/config/rag-admin.env")


def _get(url: str, timeout: float = 10.0) -> tuple[int, str]:
    try:
        r = urllib.request.urlopen(url, timeout=timeout)
        return r.status, r.read().decode("utf-8", errors="replace")
    except Exception as exc:
        return 0, str(exc)


def main() -> int:
    ok = True
    print("=== systemd ===")
    for unit in ("rag-admin", "rag-proxy", "sparse-sidecar", "rerank-sidecar"):
        out = subprocess.run(
            ["systemctl", "is-active", unit],
            capture_output=True,
            text=True,
        )
        status = out.stdout.strip() or out.stderr.strip()
        print(f"  {unit}: {status}")
        if unit in ("rag-admin", "rag-proxy") and status != "active":
            ok = False

    print("\n=== docker ===")
    out = subprocess.run(
        ["docker", "compose", "-f", "/opt/ai/rag/docker-compose.yaml", "ps", "--format", "json"],
        capture_output=True,
        text=True,
    )
    if out.stdout.strip():
        for line in out.stdout.strip().splitlines():
            row = json.loads(line)
            name = row.get("Name", "?")
            health = row.get("Health", "n/a")
            state = row.get("State", "?")
            print(f"  {name}: state={state} health={health}")
    else:
        print("  (compose ps empty)", out.stderr.strip())

    print("\n=== qdrant ===")
    code, body = _get(f"{QDRANT}/healthz")
    print(f"  healthz: {code} {body[:80]}")
    code, body = _get(f"{QDRANT}/collections/{COLLECTION}")
    if code == 200:
        result = json.loads(body).get("result", {})
        print(
            f"  collection: points={result.get('points_count')} "
            f"status={result.get('status')} optimizer={result.get('optimizer_status')}"
        )
    else:
        print(f"  collection: FAIL {body}")
        ok = False

    print("\n=== sidecars ===")
    for label, url in (
        ("sparse", "http://127.0.0.1:18096/health"),
        ("rerank", "http://127.0.0.1:18095/health"),
        ("embed:8089", "http://127.0.0.1:8089/health"),
    ):
        code, body = _get(url, timeout=3.0)
        print(f"  {label}: {code} {body[:120] if body else ''}")

    print("\n=== ingest queue ===")
    try:
        c = sqlite3.connect(ADMIN_DB)
        for row in c.execute(
            "SELECT status, count(1) FROM kb_ingest_state GROUP BY status ORDER BY status"
        ):
            print(f"  {row[0]}: {row[1]}")
        print("  failed/running:")
        for row in c.execute(
            """
            SELECT file_path, status, chunks_embedded, substr(COALESCE(last_error,''),1,80)
            FROM kb_ingest_state
            WHERE status IN ('failed','running')
            ORDER BY updated_at DESC LIMIT 8
            """
        ):
            print(f"    {row}")
    except Exception as exc:
        print(f"  sqlite error: {exc}")
        ok = False

    print("\n=== rag-admin env (key knobs) ===")
    keys = (
        "INGEST_BATCH_SIZE",
        "INGEST_EMBED_CONCURRENCY",
        "INGEST_FILE_CONCURRENCY",
        "SIDECAR_ON_DEMAND",
        "QDRANT_UPSERT_TIMEOUT_SEC",
    )
    if ADMIN_ENV.is_file():
        for line in ADMIN_ENV.read_text(encoding="utf-8").splitlines():
            if any(line.startswith(k + "=") for k in keys):
                print(f"  {line}")

    print("\n=== summary ===")
    print("  OK" if ok else "  ISSUES DETECTED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

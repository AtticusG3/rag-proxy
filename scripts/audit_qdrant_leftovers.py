#!/usr/bin/env python3
"""Identify leftover Qdrant sources after requeue."""

from __future__ import annotations

import collections
import json
import os
import sqlite3
import urllib.request


def load_env(path: str) -> dict[str, str]:
    env: dict[str, str] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            env[key.strip()] = value.strip()
    return env


def http_json(method: str, url: str, body: dict | None = None):
    data = None if body is None else json.dumps(body).encode()
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


def main() -> int:
    env = load_env("/opt/ai/config/rag-admin.env")
    base = env["QDRANT_URL"].rstrip("/")
    coll = env["QDRANT_COLLECTION"]
    conn = sqlite3.connect(env.get("ADMIN_DB_PATH", "/opt/ai/rag/admin.sqlite"))
    known = {
        r[0]
        for r in conn.execute("SELECT file_path FROM kb_ingest_state").fetchall()
    }

    # pause flags
    for key in sorted(env):
        if "PAUSE" in key.upper() or key.upper() == "INGEST_PAUSED":
            print(f"env {key}={env[key]}")

    sources: collections.Counter[str] = collections.Counter()
    no_source = 0
    samples = []
    offset = None
    while True:
        body: dict = {"limit": 64, "with_payload": True, "with_vector": False}
        if offset is not None:
            body["offset"] = offset
        result = http_json(
            "POST", f"{base}/collections/{coll}/points/scroll", body
        )["result"]
        pts = result.get("points") or []
        if not pts:
            break
        for point in pts:
            payload = point.get("payload") or {}
            src = str(payload.get("source") or "")
            if not src:
                no_source += 1
            sources[src] += 1
            if len(samples) < 5:
                samples.append(
                    {
                        "id": point.get("id"),
                        "source": src,
                        "title": str(payload.get("title") or "")[:80],
                        "preview": str(payload.get("text") or "")[:120].replace(
                            "\n", " "
                        ),
                        "in_admin": src in known,
                    }
                )
        offset = result.get("next_page_offset")
        if offset is None:
            break

    print(f"leftover_points={sum(sources.values()) + no_source}")
    print(f"no_source_payload={no_source}")
    print(f"unique_sources={len([s for s in sources if s])}")
    for src, count in sources.most_common():
        print(
            f"  {count:4d}  in_admin={src in known}  {src or '(empty)'}"
        )
    print("SAMPLES:")
    for sample in samples:
        print(json.dumps(sample, ensure_ascii=False))

    # Why not running?
    rows = conn.execute(
        "SELECT status, COUNT(*) FROM kb_ingest_state GROUP BY status"
    ).fetchall()
    print(f"ingest_status={dict(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

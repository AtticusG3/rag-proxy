#!/usr/bin/env python3
"""Diagnose NHS dirty points vs live sanitiser; fetch one raw payload."""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import urllib.request

sys.path.insert(0, "/home/kevyn/rag-proxy")


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

    # Confirm import path
    import ingest.zim_reader as zr
    import ingest.zim_sanitize as zs

    print(f"zim_reader_file={zr.__file__}")
    print(f"zim_sanitize_file={zs.__file__}")
    print(f"has_sanitize_import={'sanitize_zim_html' in dir(zr)}")

    # Find NHS source path from admin db
    conn = sqlite3.connect(env.get("ADMIN_DB_PATH", "/opt/ai/rag/admin.sqlite"))
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT file_path, status, chunks_embedded, updated_at FROM kb_ingest_state "
        "WHERE file_name LIKE '%nhs.uk_en_medicines%' LIMIT 1"
    ).fetchone()
    print(f"nhs_row={dict(row) if row else None}")
    if not row:
        return 1
    source = row["file_path"]

    # Scroll one dirty NHS point
    result = http_json(
        "POST",
        f"{base}/collections/{coll}/points/scroll",
        {
            "filter": {"must": [{"key": "source", "match": {"value": source}}]},
            "limit": 5,
            "with_payload": True,
            "with_vector": False,
        },
    )["result"]
    dirty = None
    cleanish = None
    for point in result.get("points") or []:
        text = str((point.get("payload") or {}).get("text") or "")
        title = str((point.get("payload") or {}).get("title") or "")
        if "wombat" in text.lower() and dirty is None:
            dirty = {"title": title, "text": text[:500], "id": point.get("id")}
        if "wombat" not in text.lower() and cleanish is None and len(text) > 100:
            cleanish = {"title": title, "text": text[:500], "id": point.get("id")}
    print("DIRTY_POINT=" + json.dumps(dirty, ensure_ascii=False))
    print("CLEANISH_POINT=" + json.dumps(cleanish, ensure_ascii=False))

    # Count NHS points vs filter
    # Use scroll count
    count = 0
    wombat = 0
    offset = None
    while True:
        body = {
            "filter": {"must": [{"key": "source", "match": {"value": source}}]},
            "limit": 128,
            "with_payload": ["text"],
            "with_vector": False,
        }
        if offset is not None:
            body["offset"] = offset
        page = http_json(
            "POST", f"{base}/collections/{coll}/points/scroll", body
        )["result"]
        pts = page.get("points") or []
        if not pts:
            break
        for point in pts:
            count += 1
            text = str((point.get("payload") or {}).get("text") or "")
            if "wombat" in text.lower():
                wombat += 1
        offset = page.get("next_page_offset")
        if offset is None:
            break
    print(f"nhs_points={count} nhs_wombat_points={wombat}")

    # Chemistry status anomaly
    chem = conn.execute(
        "SELECT file_path, status, chunks_embedded FROM kb_ingest_state "
        "WHERE file_name LIKE '%chemistry.stackexchange%' LIMIT 1"
    ).fetchone()
    print(f"chemistry_row={dict(chem) if chem else None}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

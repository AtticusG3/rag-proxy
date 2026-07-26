#!/usr/bin/env python3
"""Audit Qdrant payload quality and orphan sources on the ingest host."""

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
    env = load_env(os.getenv("RAG_ADMIN_ENV_FILE", "/opt/ai/config/rag-admin.env"))
    base = env["QDRANT_URL"].rstrip("/")
    coll = env["QDRANT_COLLECTION"]
    db_path = env.get("ADMIN_DB_PATH", "/opt/ai/rag/admin.sqlite")

    info = http_json("GET", f"{base}/collections/{coll}")["result"]
    points_count = info.get("points_count")
    print(f"collection={coll}")
    print(f"points_count={points_count}")
    print(f"indexed_vectors={info.get('indexed_vectors_count')}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT file_path, status, chunks_embedded, file_name FROM kb_ingest_state"
    ).fetchall()
    by_status = collections.Counter(r["status"] for r in rows)
    known = {r["file_path"] for r in rows}
    on_disk = {r["file_path"] for r in rows if os.path.isfile(r["file_path"])}
    print(f"admin_files={len(rows)} status={dict(by_status)} on_disk={len(on_disk)}")

    sources: collections.Counter[str] = collections.Counter()
    smell_total: collections.Counter[str] = collections.Counter()
    samples: list[dict] = []
    bad_examples: list[dict] = []
    offset = None
    scanned = 0
    max_scan = int(os.getenv("QDRANT_AUDIT_MAX_SCAN", "8000"))

    while scanned < max_scan:
        body: dict = {
            "limit": 64,
            "with_payload": True,
            "with_vector": False,
        }
        if offset is not None:
            body["offset"] = offset
        result = http_json(
            "POST", f"{base}/collections/{coll}/points/scroll", body
        )["result"]
        pts = result.get("points") or []
        if not pts:
            break
        for point in pts:
            scanned += 1
            payload = point.get("payload") or {}
            src = str(payload.get("source") or "")
            title = str(payload.get("title") or "")
            text = str(payload.get("text") or "")
            sources[src] += 1

            low = text.lower()
            smells: list[str] = []
            if "wombat" in low or "_____wb$" in low:
                smells.append("wayback_js")
            if '"@context"' in text and "schema.org" in text:
                smells.append("json_ld")
            if text.count("{") > 15 and (
                "function" in low or "document." in low or "window." in low
            ):
                smells.append("js_heavy")
            for smell in smells:
                smell_total[smell] += 1
            if smells and len(bad_examples) < 6:
                bad_examples.append(
                    {
                        "source": os.path.basename(src),
                        "title": title[:80],
                        "smells": smells,
                        "preview": text[:180].replace("\n", " "),
                    }
                )

            if len(samples) < 8 and src.endswith(".zim"):
                samples.append(
                    {
                        "source": os.path.basename(src),
                        "title": title[:100],
                        "chunk_idx": payload.get("chunk_idx"),
                        "text_len": len(text),
                        "preview": text[:240].replace("\n", " / "),
                    }
                )
        offset = result.get("next_page_offset")
        if offset is None:
            break

    print(f"scanned_points={scanned} unique_sources={len(sources)}")
    print(f"smell_counts={dict(smell_total)}")

    orphan_sources = sorted(src for src in sources if src and src not in known)
    not_on_disk = sorted(src for src in sources if src and src not in on_disk)
    print(f"orphan_vs_admin_db={len(orphan_sources)}")
    print(f"source_present_but_file_missing={len(not_on_disk)}")
    for src in orphan_sources[:15]:
        print(f"  ORPHAN count={sources[src]} path={src}")
    for src in not_on_disk[:10]:
        if src in known:
            print(f"  MISSING_DISK count={sources[src]} path={src}")

    print("TOP_SOURCES:")
    status_by_path = {r["file_path"]: r for r in rows}
    for src, count in sources.most_common(15):
        row = status_by_path.get(src)
        status = (
            f"{row['status']}/chunks={row['chunks_embedded']}" if row else "NOT_IN_ADMIN"
        )
        print(f"  {count:6d}  {status:28s}  {os.path.basename(src) or src}")

    print("SAMPLE_PAYLOADS:")
    for sample in samples:
        print(json.dumps(sample, ensure_ascii=False))

    if bad_examples:
        print("BAD_EXAMPLES:")
        for example in bad_examples:
            print(json.dumps(example, ensure_ascii=False))
    else:
        print("BAD_EXAMPLES: none in scanned set")

    # Full facet of distinct sources via scroll without payload text if needed:
    # For orphan completeness beyond sample, scan source-only until exhausted or cap.
    if scanned < points_count and points_count:
        print(
            f"NOTE: scanned {scanned}/{points_count} points; orphan counts are lower bounds"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Full Qdrant orphan + payload-quality audit for ingest host."""

from __future__ import annotations

import collections
import json
import os
import sqlite3
import sys
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
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode())


def main() -> int:
    env_path = os.getenv("RAG_ADMIN_ENV_FILE", "/opt/ai/config/rag-admin.env")
    env = load_env(env_path)
    base = env["QDRANT_URL"].rstrip("/")
    coll = env["QDRANT_COLLECTION"]
    db_path = env.get("ADMIN_DB_PATH", "/opt/ai/rag/admin.sqlite")

    info = http_json("GET", f"{base}/collections/{coll}")["result"]
    points_count = int(info.get("points_count") or 0)
    print(f"collection={coll}")
    print(f"points_count={points_count}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = list(
        conn.execute(
            "SELECT file_path, status, chunks_embedded, file_name FROM kb_ingest_state"
        )
    )
    known = {r["file_path"] for r in rows}
    on_disk = {r["file_path"] for r in rows if os.path.isfile(r["file_path"])}
    status_map = {r["file_path"]: r for r in rows}
    print(
        "admin_status="
        + json.dumps(dict(collections.Counter(r["status"] for r in rows)))
    )

    sources: collections.Counter[str] = collections.Counter()
    smells: collections.Counter[str] = collections.Counter()
    smells_by_source: dict[str, collections.Counter[str]] = collections.defaultdict(
        collections.Counter
    )
    samples_bad: list[dict] = []
    samples_good: list[dict] = []
    offset = None
    scanned = 0

    while True:
        body: dict = {
            "limit": 128,
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
            found: list[str] = []
            if "wombat" in low or "_____wb$" in low:
                found.append("wayback_js")
            if "@context" in text and "schema.org" in text:
                found.append("json_ld")
            if "background-image" in low or "external-link" in low and "{" in text:
                if text.count("{") > 3:
                    found.append("css_leak")
            for smell in found:
                smells[smell] += 1
                smells_by_source[src][smell] += 1
            if found and len(samples_bad) < 5:
                samples_bad.append(
                    {
                        "source": os.path.basename(src),
                        "title": title[:90],
                        "smells": found,
                        "preview": text[:200].replace("\n", " "),
                    }
                )
            if (
                not found
                and src.endswith(".zim")
                and len(samples_good) < 5
                and len(text) > 200
            ):
                samples_good.append(
                    {
                        "source": os.path.basename(src),
                        "title": title[:90],
                        "preview": text[:200].replace("\n", " / "),
                    }
                )
        offset = result.get("next_page_offset")
        if offset is None:
            break

    orphans = sorted(src for src in sources if src and src not in known)
    missing_disk = sorted(src for src in sources if src and src not in on_disk)
    print(f"scanned_points={scanned}")
    print(f"unique_sources={len(sources)}")
    print(f"orphan_vs_admin_db={len(orphans)}")
    print(f"source_file_missing={len(missing_disk)}")
    print(f"smell_totals={dict(smells)}")
    if scanned:
        print(
            "wayback_js_pct="
            + str(round(100.0 * smells.get("wayback_js", 0) / scanned, 2))
        )

    for src in orphans[:20]:
        print(f"ORPHAN count={sources[src]} path={src}")
    for src in missing_disk[:10]:
        print(f"MISSING_DISK count={sources[src]} path={src}")

    print("TOP_SOURCES:")
    for src, count in sources.most_common(20):
        row = status_map.get(src)
        status = (
            f"{row['status']}/chunks={row['chunks_embedded']}"
            if row
            else "NOT_IN_ADMIN"
        )
        smell = dict(smells_by_source.get(src, {}))
        print(
            f"  {count:7d}  {status:30s}  smells={smell}  {os.path.basename(src) or src}"
        )

    print("GOOD_SAMPLES:")
    for sample in samples_good:
        print(json.dumps(sample, ensure_ascii=False))
    print("BAD_SAMPLES:")
    for sample in samples_bad:
        print(json.dumps(sample, ensure_ascii=False))

    # Live sanitiser check against one NHS point title if available
    try:
        sys.path.insert(0, "/home/kevyn/rag-proxy")
        from ingest.zim_sanitize import sanitize_zim_html
        from libzim.reader import Archive

        nhs = next((s for s in sources if "nhs.uk_en_medicines" in s), "")
        if nhs and os.path.isfile(nhs):
            archive = Archive(nhs)
            # find first html entry and compare
            checked = 0
            for index in range(min(int(archive.all_entry_count), 400)):
                try:
                    entry = archive._get_entry_by_id(index)
                    if entry.is_redirect:
                        continue
                    item = entry.get_item()
                    mime = (getattr(item, "mimetype", "") or "").lower()
                    if "html" not in mime:
                        continue
                    raw = bytes(item.content).decode("utf-8", errors="replace")
                    cleaned = sanitize_zim_html(
                        raw, title=entry.title or entry.path, url=entry.path
                    )
                    has_wombat = "wombat" in raw.lower()
                    clean_wombat = cleaned is not None and "wombat" in cleaned.lower()
                    if has_wombat:
                        print(
                            "SANITIZE_CHECK",
                            json.dumps(
                                {
                                    "title": (entry.title or "")[:80],
                                    "raw_has_wombat": has_wombat,
                                    "clean_is_none": cleaned is None,
                                    "clean_has_wombat": clean_wombat,
                                    "clean_len": 0 if cleaned is None else len(cleaned),
                                    "preview": ""
                                    if cleaned is None
                                    else cleaned[:160].replace("\n", " "),
                                },
                                ensure_ascii=False,
                            ),
                        )
                        checked += 1
                        if checked >= 3:
                            break
                except Exception:
                    continue
    except Exception as exc:
        print(f"SANITIZE_CHECK_ERROR={exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

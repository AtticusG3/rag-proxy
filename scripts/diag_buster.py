#!/usr/bin/env python3
"""One-shot buster health dump (run on host)."""
import sqlite3
import subprocess
import sys
import urllib.request

DB = "/opt/ai/rag/admin.sqlite"
QUERY_EMBED = "http://127.0.0.1:8089"


def main() -> None:
    fix = "--fix-embed-url" in sys.argv
    c = sqlite3.connect(DB)
    if fix:
        c.execute(
            "update admin_settings set value=? where key=?",
            (QUERY_EMBED, "EMBED_URL"),
        )
        c.commit()
        print("fixed EMBED_URL in admin_settings ->", QUERY_EMBED)
    print("admin_settings EMBED*:", c.execute(
        "select key, value from admin_settings where key like '%EMBED%'"
    ).fetchall())
    print("ingest state:", c.execute(
        "select status, count(1) from kb_ingest_state group by status"
    ).fetchall())
    try:
        r = urllib.request.urlopen(
            "http://127.0.0.1:6333/collections/nomad_knowledge_base", timeout=10
        )
        print("qdrant:", r.status, r.read()[:120])
    except Exception as exc:
        print("qdrant:", exc)
    try:
        r = urllib.request.urlopen("http://127.0.0.1:8089/health", timeout=5)
        print("embed health:", r.status)
    except Exception as exc:
        print("embed health:", exc)
    out = subprocess.run(
        ["docker", "logs", "rag_qdrant", "--tail", "2"],
        capture_output=True,
        text=True,
    )
    print("qdrant log tail:", out.stdout or out.stderr)


if __name__ == "__main__":
    main()

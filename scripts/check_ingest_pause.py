#!/usr/bin/env python3
import sqlite3

conn = sqlite3.connect("/opt/ai/rag/admin.sqlite")
print("tables=", [r[0] for r in conn.execute(
    "SELECT name FROM sqlite_master WHERE type='table'"
)])
try:
    print("admin_settings_pause=", conn.execute(
        "SELECT key, value FROM admin_settings WHERE key LIKE '%PAUSE%'"
    ).fetchall())
except Exception as exc:
    print("admin_settings_err=", exc)
print("status=", conn.execute(
    "SELECT status, COUNT(*) FROM kb_ingest_state GROUP BY status"
).fetchall())
print("nonzero_chunks=")
for row in conn.execute(
    "SELECT file_name, status, chunks_embedded, updated_at FROM kb_ingest_state "
    "WHERE chunks_embedded > 0 ORDER BY updated_at DESC LIMIT 12"
):
    print(" ", row)

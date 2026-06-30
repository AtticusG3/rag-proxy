#!/usr/bin/env python3
import sqlite3
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/rag/admin.sqlite"
conn = sqlite3.connect(path)
rows = conn.execute(
    "SELECT status, COUNT(*) FROM kb_ingest_state GROUP BY status ORDER BY status"
).fetchall()
print("ingest_state:", dict(rows))
running = conn.execute(
    "SELECT file_name, chunks_embedded FROM kb_ingest_state WHERE status='running' LIMIT 5"
).fetchall()
if running:
    print("running:", running)

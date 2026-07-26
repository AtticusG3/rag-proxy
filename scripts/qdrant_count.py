#!/usr/bin/env python3
import json
import urllib.request

env = {}
with open("/opt/ai/config/rag-admin.env", encoding="utf-8") as handle:
    for line in handle:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip()

base = env["QDRANT_URL"].rstrip("/")
coll = env["QDRANT_COLLECTION"]
info = json.load(urllib.request.urlopen(f"{base}/collections/{coll}"))["result"]
print(f"points_count={info.get('points_count')}")

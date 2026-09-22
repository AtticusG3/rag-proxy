---
name: rag-proxy-debug
description: >-
  Diagnoses why RAG context is missing or wrong in rag_proxy — embedding,
  Qdrant search, threshold, injection, or upstream passthrough. Use when RAG
  does not inject, scores are low, streaming breaks, or homelab pipeline debug.
---

# rag_proxy — Debug RAG Pipeline

## Success criteria

```
- [ ] Identified which stage failed: query extract | embed | qdrant | inject | forward
- [ ] Evidence: log line or HTTP response, not guesswork
- [ ] Fix or documented config change with verification step
```

## Decision flow

```
POST chat/completions?
  no  -> expected: passthrough only
  yes -> query extracted? (LOG: no user text)
         no  -> check messages / multimodal content
         yes -> embedding OK? (LOG: skipped embedding returned None)
                no  -> curl EMBED_URL /v1/embeddings (model = EMBED_MODEL, e.g. embed-pool)
                yes -> Qdrant hits? (LOG: no chunks above threshold)
                       no  -> lower SIMILARITY_THRESHOLD or fix QDRANT_URL/collection/vector width
                       yes -> injection OK? (LOG: injected N chunks)
                              check client sees system message growth
```

## Read logs

`LOG_LEVEL=DEBUG` for threshold skips. Look for:

- `RAG: injected N chunk(s)` — success
- `RAG: no chunks above threshold` — search ran, nothing passed score
- `RAG: skipped (embedding returned None)` — embed failure
- `RAG augmentation error (passing through unmodified)` — exception in RAG block

## Manual checks (ASCII-friendly)

```bash
# Embed server — match EMBED_MODEL (embed-pool or legacy nomic-embed-text-v1.5)
curl -s -X POST "$EMBED_URL/v1/embeddings" -H "Content-Type: application/json" -d "{\"model\":\"embed-pool\",\"input\":\"test\"}"

# Qdrant (replace collection)
curl -s "$QDRANT_URL/collections/$QDRANT_COLLECTION"

# Proxy health — non-RAG passthrough
curl -s "http://127.0.0.1:${PROXY_PORT:-8088}/v1/models"
```

## Common causes

| Symptom | Likely cause |
|---------|----------------|
| Never injects | `QDRANT_URL` placeholder `CHANGE_ME`, wrong collection |
| Sometimes injects | `SIMILARITY_THRESHOLD` too high for your vectors |
| Always passthrough | Wrong path (not in `CHAT_PATHS`), not POST |
| Empty chunks | Payload keys don't match `extract_chunk_text` order |
| Stream hangs | client disconnect or abandoned stream — check `relay_upstream`; janitor log `closed N idle upstream stream(s)` means `UPSTREAM_STREAM_ABANDON_SEC` elapsed with no bytes relayed |
| Docker: embed fails | `EMBED_URL` uses `127.0.0.1` inside container — use host LAN/Tailscale URL or `host.docker.internal` |

## Rules

- Do not add permanent debug endpoints without user request (Rule 2).
- Fix root cause; don't disable fail-open unless user explicitly wants fail-closed.

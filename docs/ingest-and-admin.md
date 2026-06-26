# Ingest and admin

Optional content management stack for indexing ZIM archives, PDFs, and text into Qdrant. Separate from the core proxy — you can run `rag_proxy` against an existing Qdrant collection without ever running admin/ingest.

## Deployment topology

| Component | Typical placement |
| --- | --- |
| `rag_proxy` | Any host with network access to Qdrant, embed, and llama-swap |
| `rag_admin` + ingest | Optional; same host or a dedicated indexing machine |

Admin upserts vectors to Qdrant; the proxy reads the same `QDRANT_URL` / `QDRANT_COLLECTION`. Nothing requires co-location — only reachable URLs and paths matter.

This repository ships `rag-proxy.service` and `nomic-embed.service` examples only. Provide your own systemd unit (or process manager) for `python -m rag_admin` if needed. `scripts/catalog_weekly_update.py` accepts `RAG_ADMIN_ENV` (default `/opt/ai/config/rag-admin.env`) for cron on whichever host runs admin.

## Components

| Component | Path | Role |
| --- | --- | --- |
| RAG admin UI | `rag_admin/` | Web UI: catalog, uploads, job queue, content explorer |
| Ingest worker | `ingest/` | Background worker: chunk, embed, upsert to Qdrant |
| Offline MemGraphRAG index | `scripts/build_memgraphrag_index.py` | Build MemGraphRAG SQLite from chunks |

Admin embeds the ingest worker in-process (`IngestWorker` started from `rag_admin/app.py` lifespan).

## RAG admin UI

### Run

```bash
# From repo root with shared .env
python -m rag_admin
```

Uses uvicorn; bind and paths from environment (see below).

### Features

- Dashboard and ingest job status
- ZIM catalog subscriptions and downloads
- PDF/text upload queue
- Content explorer for indexed material
- arXiv and archive catalog providers (`rag_admin/catalog/`)

### Security

Startup **refuses** default `ADMIN_SESSION_SECRET` and `ADMIN_PASSWORD` unless `ADMIN_ALLOW_INSECURE_DEFAULTS=true` (local dev only). Set strong secrets before exposing beyond localhost.

Default bind: `127.0.0.1:8087`. `rag_admin/config.py` defaults `EMBED_URL` to `http://127.0.0.1:18089` when unset — set `EMBED_URL` explicitly if your embed server uses a different port.

## Environment variables

Proxy and admin share many vars (`EMBED_URL`, `QDRANT_URL`, `QDRANT_COLLECTION`). Admin-specific (from `.env.example` comments and `rag_admin/config.py`):

| Variable | Default | Purpose |
| --- | --- | --- |
| `ADMIN_HOST` | `127.0.0.1` | UI bind address |
| `ADMIN_PORT` | `8087` | UI port |
| `ADMIN_DB_PATH` | `/opt/ai/rag/admin.sqlite` | Job/catalog SQLite |
| `ZIM_DIR` | `/opt/ai/rag/zim` | Downloaded ZIM files |
| `UPLOAD_DIR` | `/opt/ai/rag/uploads` | Uploaded PDFs/text |
| `ADMIN_SESSION_SECRET` | *(required)* | Session cookie signing |
| `ADMIN_PASSWORD` | *(required)* | Login password |
| `ADMIN_ALLOW_INSECURE_DEFAULTS` | — | `true` for local dev only |
| `INGEST_BATCH_SIZE` | `64` | Texts per embed HTTP request / Qdrant upsert batch |
| `INGEST_EMBED_CONCURRENCY` | `4` | Concurrent in-flight embed batches (auto-set when using VRAM pool) |
| `INGEST_EMBED_URLS` | — | Comma-separated embed endpoints for ingest round-robin (generated pool file) |
| `INGEST_MAX_ARTICLES` | `0` | ZIM article limit (`0` = unlimited) |
| `INGEST_SPARSE_REINDEX` | `idle` | When to trigger sparse sidecar reindex |
| `INGEST_STALL_MINUTES` | `15` | Mark jobs stalled after no progress |
| `RAG_PROXY_URL` | `http://127.0.0.1:8081` | Optional proxy URL for admin smoke hooks |

Full list: [Configuration — RAG admin and ingest](configuration.md#rag-admin-and-ingest-optional).

## Ingest pipeline

1. **Queue** — jobs created from UI (ZIM path, upload, catalog subscription).
2. **Read** — ZIM (`ingest/zim_reader.py`), PDF (`ingest/pdf_reader.py`), or plain text.
3. **Chunk** — `ingest/chunking.py` splits text for embedding.
4. **Embed** — `ingest/embedder.py` calls `EMBED_URL` (same nomic-embed as proxy).
5. **Write** — `ingest/qdrant_writer.py` upserts to `QDRANT_COLLECTION`.
6. **Sparse reindex** — optional POST to `SPARSE_INDEX_URL` when `INGEST_SPARSE_REINDEX` triggers (hybrid cognitive mode).

Bulk ZIM ingest uses `ingest/pipeline.py`: multiple embed batches run concurrently (`INGEST_EMBED_CONCURRENCY`) while Qdrant upserts stay in chunk order. Set `llama-server --parallel` on the embed endpoint to at least the same value (e.g. `16` on a dedicated nomic-embed GPU). Smaller `INGEST_BATCH_SIZE` (e.g. `32`) with higher concurrency often beats one huge batch per request.

Chunking defaults to 400 characters so each input stays under the per-slot token limit when context is divided by `--parallel` (e.g. `-c 8096 --parallel 16` -> ~512 tokens per input). The embedder bisects batches on `exceed_context_size` 400 responses.

### Multi-instance embed pool (VRAM auto-scale)

For bulk ingest on a GPU host, run several `llama-server` embed instances (systemd template `nomic-embed@PORT.service`) and round-robin across them via `INGEST_EMBED_URLS`.

`scripts/scale_nomic_embed_pool.py` sizes the pool from free VRAM:

```text
instances = clamp((gpu_free_mib - NOMIC_POOL_VRAM_RESERVE_MIB) / NOMIC_POOL_VRAM_PER_INSTANCE_MIB)
```

It writes `/opt/ai/config/nomic-embed-pool.env` with `INGEST_EMBED_URLS` and `INGEST_EMBED_CONCURRENCY` (`instances * --parallel` per unit). Tune via `/opt/ai/config/nomic-embed-scale.env` (see `nomic-embed-scale.env.example` in infra).

```bash
# Dry-run plan
python scripts/scale_nomic_embed_pool.py

# Apply via systemd (recommended on boot)
systemctl start nomic-embed-scale.service
systemctl restart rag-admin.service
```

Without `nvidia-smi`, the planner falls back to a single port (`NOMIC_POOL_PORT_BASE`). The embedder fails over to alternate pool URLs on HTTP 404/5xx.

Payload fields written for proxy retrieval: `text`, `content`, `chunk`, `document`, `page_content` (proxy checks in that order).

### Stall detection

`ingest/stall.py` marks long-idle jobs using `INGEST_STALL_MINUTES`.

## Qdrant ownership

If you do not control the Qdrant collection schema, coordinate index params and payload fields with the collection owner. rag_proxy and ingest can upsert points but cannot change upstream schema.

## MemGraphRAG

MemGraphRAG builds a separate SQLite graph index from your Qdrant chunks (or text files). It is not part of the ingest worker — run the offline build after content is in Qdrant.

Full operator guide: [MemGraphRAG](memgraphrag.md).

## Catalog weekly updates

Cron helper for subscription update checks:

```bash
python scripts/catalog_weekly_update.py
```

## MCP retrieval tools

`sidecars/mcp_rag/` exposes MCP tools (e.g. `search_knowledge_base`) over the hybrid stack for IDE integration — separate from the HTTP proxy path.

## Related docs

- Proxy retrieval behavior: [Architecture](architecture.md)
- Hybrid/sparse sidecar: [docker/README.md](../docker/README.md)
- Verify indexed content reaches chat: [Getting started — Verify the stack](getting-started.md#verify-the-stack)

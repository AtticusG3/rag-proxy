# Ingest Codemap

**Last Updated:** 2026-09-22
**Entry Points:** `ingest/worker.py` (`IngestWorker`), started from `rag_admin` lifespan / jobs

## Architecture

```text
Admin queue (SQLite)
        |
        v
  IngestWorker file threads
        |
        +-- read/extract (text / ZIM / PDF / corpus JSONL)
        +-- chunking (+ strategy selection)
        +-- embed HTTP (pool URLs / concurrency)
        +-- Qdrant upsert
        +-- optional TurboVec dual-write
        +-- optional sparse / TurboVec reindex schedulers
```

## Key Modules

| Module | Purpose |
| --- | --- |
| `ingest/worker.py` | Queue consumer, file concurrency, reindex schedulers |
| `ingest/endpoint_health.py` | Probe embed + Qdrant; wait/resume instead of failing files |
| `ingest/pipeline.py` | Per-file embed + upsert orchestration |
| `ingest/chunking.py` | Chonkie execution, `INGEST_CHUNK_*` |
| `ingest/chunking_strategy.py` | Per-document strategy selection |
| `ingest/embedder.py` | HTTP embed batches |
| `ingest/qdrant_writer.py` | Upsert with retries/backoff |
| `ingest/turbovec_client.py` | Dual-write / reindex client |
| `ingest/dual_write.py` | Qdrant + TurboVec write path |
| `ingest/scanner.py` | Directory scan / enqueue |
| `ingest/zim_reader.py` / `pdf_reader.py` | Format extractors |

## Data Flow

1. Files land under `ZIM_DIR` / `UPLOAD_DIR` (or catalog downloads).
2. Worker dequeues, chunks, embeds via `INGEST_EMBED_URLS` (or `EMBED_URL`).
3. Points upsert to Qdrant; when `TURBOVEC_URL` is set, vectors dual-write.

## Operator scripts

| Script | Role |
| --- | --- |
| `scripts/requeue_all_ingest.py` | Re-chunk after `INGEST_CHUNK_*` changes |

## Related Areas

- [admin.md](admin.md) — UI / Settings that drive the worker
- [Ingest and admin](../ingest-and-admin.md)

# Ingest Codemap

**Last Updated:** 2026-07-27
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
| `ingest/pipeline.py` | Per-file embed + upsert orchestration |
| `ingest/chunking.py` | Chonkie execution, `INGEST_CHUNK_*` |
| `ingest/chunking_strategy.py` | Per-document strategy selection |
| `ingest/embedder.py` | HTTP embed batches |
| `ingest/embed_pool.py` | VRAM-aware nomic-embed pool plan |
| `ingest/capacity_planner.py` | Multi-resource caps (CPU/RAM/disk/GPU) |
| `ingest/host_profile.py` | Host probes (`nvidia-smi`, RAM, disk) |
| `ingest/qdrant_writer.py` | Upsert with retries/backoff |
| `ingest/turbovec_client.py` | Dual-write / reindex client |
| `ingest/dual_write.py` | Qdrant + TurboVec write path |
| `ingest/scanner.py` | Directory scan / enqueue |
| `ingest/zim_reader.py` / `pdf_reader.py` | Format extractors |

## Data Flow

1. Files land under `ZIM_DIR` / `UPLOAD_DIR` (or catalog downloads).
2. Worker dequeues, chunks, embeds via `INGEST_EMBED_URLS` (or `EMBED_URL`).
3. Points upsert to Qdrant; when `TURBOVEC_URL` is set, vectors dual-write.
4. Capacity planner (`scripts/scale_ingest_capacity.py`) writes pool + concurrency env.

## Operator scripts

| Script | Role |
| --- | --- |
| `scripts/scale_ingest_capacity.py` | Apply capacity plan |
| `scripts/bench_ingest_capacity.py` | Throughput benches |
| `scripts/requeue_all_ingest.py` | Re-chunk after `INGEST_CHUNK_*` changes |

## Related Areas

- [admin.md](admin.md) — UI / Settings that drive the worker
- [Ingest and admin](../ingest-and-admin.md)
- [Ingest capacity planning](../ingest-capacity-planning.md)

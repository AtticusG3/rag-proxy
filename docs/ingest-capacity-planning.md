# Ingest capacity planning

Audit reference for the multi-resource ingest scaling work. Documents how each ingest
stage binds to host resources, which knobs control throughput, what the current planner
covers, and the metrics used to judge whether a capacity plan actually helped.

Related guides: [Ingest and admin](ingest-and-admin.md), [Configuration](configuration.md),
[Deployment](deployment.md).

## Pipeline stages and resource binding

Per-file flow in `ingest/worker.py`:

```mermaid
flowchart LR
  scan[Scan and queue] --> read[Read and extract]
  read --> chunk[Chunk]
  chunk --> embed[Embed HTTP]
  embed --> upsert[Qdrant upsert]
  upsert --> sparse[Sparse BM25 reindex]
```

| Stage | Code | Primary binding | Notes |
| --- | --- | --- | --- |
| Scan / queue | `ingest/scanner.py`, `IngestWorker.enqueue_sync` | Disk metadata + SQLite | Not a throughput bottleneck |
| Read / extract (text) | `_read_text_file` in `ingest/worker.py` | RAM | Whole file read into memory |
| Read / extract (ZIM) | `ingest/zim_reader.py` | Disk I/O + CPU | Streaming iterator; HTML strip per article |
| Read / extract (PDF) | `ingest/pdf_reader.py` | CPU + disk | `pypdf` per-page text extraction |
| Chunk | `ingest/chunking.py` | CPU (RAM for semantic) | Chonkie strategies; semantic loads an embedding model |
| Embed | `ingest/pipeline.py`, `ingest/embedder.py` | GPU (remote) + network | HTTP to nomic-embed pool; usually the intended bottleneck |
| Qdrant upsert | `ingest/qdrant_writer.py` | Network + Qdrant disk | Serialized in chunk order per file |
| Sparse reindex | `SparseReindexScheduler` in `ingest/worker.py` | Sidecar CPU + disk | Full-collection rebuild; can dominate end-of-run latency |

### Concurrency model

Two tiers, both worker-global:

- **File workers** — one thread per file, count fixed at `IngestWorker.start()` from
  `resolve_file_concurrency()`: explicit `INGEST_FILE_CONCURRENCY`, else
  `max(1, min(4, pool URL count))`.
- **Embed batches** — per-file `ThreadPoolExecutor` gated by one shared
  `threading.Semaphore(INGEST_EMBED_CONCURRENCY)`. Total in-flight embed HTTP calls
  across all files never exceeds this value.

### Known scaling ceilings

| Ceiling | Location | Effect |
| --- | --- | --- |
| Chunk concurrency cap | `INGEST_CHUNK_CONCURRENCY` semaphore in `ingest/chunking.py` | Caps parallel Chonkie executions across files (per-thread runners) |
| Chunk profile changes need requeue | `scripts/requeue_all_ingest.py` | Changing `INGEST_CHUNK_*` has no effect on already-indexed files without a requeue |
| Manual pool env drift | `NOMIC_POOL_PARALLEL_PER_INSTANCE` in scale env vs written `NOMIC_POOL_PARALLEL` | Re-run the scale job after editing scale env; planner writes aligned values to the pool env |

## Knob inventory

### Hot-reload (applied by `apply_to_worker` on settings save)

| Variable | Default | Effect |
| --- | --- | --- |
| `INGEST_BATCH_SIZE` | `64` | Texts per embed HTTP request / Qdrant upsert batch |
| `INGEST_EMBED_CONCURRENCY` | `4` | Global cap on concurrent embed batches |
| `INGEST_FILE_CONCURRENCY` | auto | Worker thread count; hot-reloads via `IngestWorker.resize_file_workers()` when the worker is running |
| `INGEST_EMBED_URLS` | empty | Embed pool endpoints, round-robin |
| `INGEST_SPARSE_REINDEX` | `idle` | BM25 rebuild trigger (`off` / `each` / `idle`) |
| `INGEST_STALL_MINUTES` | `15` | Stall detection window |
| `INGEST_MAX_ARTICLES` | `0` | ZIM article cap |
| `EMBED_MAX_CHARS` | `2000` | Truncation before the embed API |

### Restart + requeue required

| Variable | Default | Effect |
| --- | --- | --- |
| `INGEST_CHUNK_SIZE_TOKENS` | `512` | Chunk size; drives chunk count and embed load |
| `INGEST_CHUNK_OVERLAP_TOKENS` | `64` | Overlap; more chunks per document |
| `INGEST_CHUNK_TOKENIZER` | `nomic-ai/nomic-embed-text-v1.5` | CPU cost at chunk time |
| `INGEST_CHUNK_SEMANTIC` | `true` | Enables the heaviest chunk path (model in RAM) |
| `INGEST_CHUNK_SEMANTIC_MODEL` | `minishlab/potion-base-32M` | Semantic boundary model |
| `INGEST_CHUNK_MIN_TOKENS` | `100` | Merge pass for undersized chunks |

### GPU pool planner (`nomic-embed-scale.env`)

| Variable | Default | Effect |
| --- | --- | --- |
| `NOMIC_POOL_VRAM_PER_INSTANCE_MIB` | `1024` | Instance sizing |
| `NOMIC_POOL_VRAM_RESERVE_MIB` | `2048` | Headroom for other GPU workloads |
| `NOMIC_POOL_MAX_INSTANCES` | `12` | Hard cap |
| `NOMIC_POOL_MIN_INSTANCES` | `1` | No-GPU floor |
| `NOMIC_POOL_PARALLEL_PER_INSTANCE` | `16` | Concurrency multiplier in the plan |
| `NOMIC_POOL_PARALLEL` | `16` | `llama-server --parallel` in systemd units |
| `NOMIC_POOL_PORT_BASE` | `18089` | First pool port |
| `NOMIC_POOL_GPU_INDEX` | `0` | `nvidia-smi` target |

## Capacity planner (current)

Entry point: `scripts/scale_ingest_capacity.py` (legacy wrapper `scripts/scale_nomic_embed_pool.py`; `nomic-embed-scale.service` runs it with `--apply`).

Modules:

- `ingest/host_profile.py` — CPU, RAM, disk, and GPU probes
- `ingest/embed_pool.py` — VRAM pool sizing (`instances = clamp((free - reserve) / per_instance, min, max)`)
- `ingest/capacity_planner.py` — merges pool plan with CPU/RAM/disk caps and GPU bandwidth tiering
- `ingest/bench_fit.py` — optional bench JSON overrides for chunk/embed concurrency and batch size

The VRAM-only embed pool planner still lives inside `embed_pool.py`; `scale_ingest_capacity.py` wraps it with multi-resource caps, rationale comments in the pool env file, and optional benchmark fit from the admin **Scale ingest capacity** job.

## Host signals used by the capacity planner

| Signal | Probe | Planner use |
| --- | --- | --- |
| `cpu_logical_cores` | `os.cpu_count()` | Cap file concurrency and chunk parallelism |
| `cpu_model` | `/proc/cpuinfo` (`platform.processor()` fallback) | Display / diagnostics |
| `ram_total_mib`, `ram_available_mib` | `/proc/meminfo` | Disable semantic chunking below floor; cap file concurrency |
| `disk_free_mib` per data path | `shutil.disk_usage` | Warn on low space |
| `disk_seq_read_mbps` | Cached one-shot read benchmark | Cap file concurrency on slow storage |
| `gpu_free_mib` | `nvidia-smi` | Embed instance count (existing formula) |
| `gpu_name` | `nvidia-smi` | Bandwidth tier lookup adjusting per-instance parallel |

## Planner outputs

`scripts/scale_ingest_capacity.py` (wrapped by the legacy `scale_nomic_embed_pool.py`
entry point) writes the full plan to the pool env file, and the admin scale job syncs
the `INGEST_*` keys into the admin env for hot reload:

| Key | Source |
| --- | --- |
| `INGEST_EMBED_URLS`, `INGEST_EMBED_CONCURRENCY` | VRAM pool plan x per-instance parallel |
| `INGEST_FILE_CONCURRENCY` | min() of pool size, CPU, RAM, disk, and configured caps |
| `INGEST_BATCH_SIZE` | Inverse of embed concurrency (32 at >=16, 64 at >=8, else 128) |
| `INGEST_CHUNK_CONCURRENCY` | min(file concurrency, cores / chunk share) |
| `INGEST_CHUNK_SEMANTIC` | Downgraded below RAM/CPU floors, never upgraded |
| `INGEST_SPARSE_REINDEX` | `off` during bulk (rebuild once at end) |
| `NOMIC_POOL_PARALLEL` | Single source for planner math and systemd `--parallel` |
| `CAPACITY_*` | Host snapshot for the admin UI (not synced) |

Rationale for each decision is printed to the job log and written as comments in the
pool env file.

Planner caps are tunable via `INGEST_CAPACITY_*` env vars (see `.env.example`).

## Calibration results (2026-07)

`scripts/bench_ingest_capacity.py` sweeps on the homelab GPU host
(i5-9600T 6 cores, 52 GiB RAM free, Tesla V100 32 GiB, 4 embed instances):

| Sweep | Result |
| --- | --- |
| Chunk concurrency 1 -> 2 | 6,505 -> 15,082 chunks/min (2.3x) |
| Chunk concurrency 2 -> 3 | Flat (~14,400) on 6 cores; cores/2 cap is right |
| Embed concurrency 4 -> 16 (4 instances) | 787 -> 2,250 chunks/min at batch 32 |
| Embed concurrency 16 -> 32 | Flat (~2,200); instances x parallel is the ceiling |
| Batch 32 vs 64 at concurrency >= 16 | Equivalent (2,250 vs 2,205); planner picks 32 |

These validate the shipped coefficients: `chunk_cpu_share=2`, embed concurrency
`instances x parallel`, and the batch-size step function. When bench JSON is
present during scale, measured `chunks_per_min` overrides chunk concurrency,
embed concurrency, and batch size (see `ingest/bench_fit.py`).

## Success metrics

| Metric | Source | Target |
| --- | --- | --- |
| Chunks per minute | Jobs page embed rate; `chunks_embedded` deltas in ingest SQLite | Higher after scale on multi-core hosts |
| Wall time per fixture set | `scripts/bench_ingest_capacity.py` report | Plan within 10% of best observed sweep result |
| Stall rate | `stalled` count in queue stats | No increase after scale |
| Embed pool health | Scale job log (health probe results) | All planned instances healthy |

## Fail-open defaults

When probes are unavailable the planner falls back conservatively, matching the spirit of
the existing no-GPU path:

- No `nvidia-smi`: single-port pool, no systemd changes.
- No `/proc/meminfo` (non-Linux dev host): RAM caps skipped, semantic chunking left as
  configured.
- No disk benchmark: disk cap skipped.
- Any probe exception: log a warning, continue with remaining signals.

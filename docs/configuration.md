# Configuration

All settings load from environment variables (`.env` or systemd `EnvironmentFile`). Defaults live in `rag_proxy/config.py`; [.env.example](../.env.example) is the operator template.

Restart the proxy after changing `.env`.

## Core proxy and retrieval

| Variable | Default | Purpose |
| --- | --- | --- |
| `LLAMA_SWAP_URL` | `http://127.0.0.1:8080` | Upstream OpenAI-compatible chat API base URL (historical name; not limited to llama-swap). Examples: llama-swap, llama-server, vLLM, OpenRouter, OpenAI |
| `EMBED_URL` | `http://127.0.0.1:8089` | nomic-embed llama-server |
| `QDRANT_URL` | `http://192.168.1.36:6333` | Qdrant HTTP API |
| `QDRANT_COLLECTION` | `nomad_knowledge_base` | Collection name |
| `TOP_K` | `5` | Max chunks to retrieve |
| `SIMILARITY_THRESHOLD` | `0.65` | Minimum cosine score to inject |
| `PROXY_HOST` | `0.0.0.0` | Bind address |
| `PROXY_PORT` | `8088` | Listen port (use another port for a side-by-side second instance) |
| `LOG_LEVEL` | `INFO` | Python log level |
| `EMBED_MAX_CHARS` | `2000` | Truncate query text before embed |
| `EMBED_RETRIES` | `2` | Embed request retries |

## Upstream HTTP pool

Shared `httpx.AsyncClient` for all upstream traffic (chat API relay and embed/Qdrant calls).

| Variable | Default | Purpose |
| --- | --- | --- |
| `UPSTREAM_TIMEOUT_SEC` | `600` | Request timeout |
| `UPSTREAM_MAX_CONNECTIONS` | `50` | Connection pool cap |
| `UPSTREAM_MAX_KEEPALIVE` | `0` | Keepalive sockets (`0` = close after one-shot polls) |
| `UPSTREAM_KEEPALIVE_EXPIRY_SEC` | `15` | Keepalive socket expiry |
| `UPSTREAM_IDLE_SWEEP_SEC` | `30` | Janitor interval for abandoned streams |
| `UPSTREAM_STREAM_ABANDON_SEC` | `120` | Close upstream SSE when no bytes relayed for this long |

## Proxy access control (optional)

| Variable | Default | Purpose |
| --- | --- | --- |
| `PROXY_INTERNAL_TOKEN` | *(empty)* | When set, require `X-Internal-Token` header on all proxy routes and `GET /metrics` |

Default empty = no change (open proxy, same as before). Use when `PROXY_HOST=0.0.0.0` exposes the proxy on LAN or Tailscale — see [Deployment — Trust boundary](deployment.md#trust-boundary).

## Cognitive pipeline master and stages

| Variable | Default | Purpose |
| --- | --- | --- |
| `ENABLE_COGNITIVE_PIPELINE` | `false` | Master switch; `false` = legacy always-retrieve |
| `ENABLE_TIER0_HEURISTICS` | `false` | Regex fast path; skip embed/Qdrant for simple queries |
| `ENABLE_INTENT_ROUTER` | `false` | Rules + optional `INTENT_MODEL` |
| `ENABLE_RETRIEVAL_GATING` | `false` | Skip retrieval when not needed |
| `GATING_LOG_ONLY` | `false` | Log gating decisions without skipping (bake-in) |
| `ENABLE_QUERY_REWRITE` | `false` | Deterministic query normalization |
| `ENABLE_QUERY_REWRITE_LLM` | `false` | LLM query rewrite (requires rewrite enabled) |
| `ENABLE_HYBRID_RETRIEVAL` | `false` | Dense + sparse RRF merge |
| `ENABLE_RERANKER` | `false` | HTTP rerank sidecar |
| `ENABLE_SEMANTIC_DEDUPE` | `false` | Embedding-similarity dedupe before inject |
| `ENABLE_GRAPH_LOOKUP` | `false` | SQLite infra graph |
| `ENABLE_MEMGRAPHRAG` | `false` | Three-layer memory + PPR retrieval |
| `ENABLE_MODEL_ROUTING` | `false` | Model override by intent |
| `MODEL_ROUTING_MODE` | `suggest` | `suggest` or `force` |
| `ENABLE_TOOLS` | `false` | Whitelisted file reads |
| `ENABLE_ROLLING_MEMORY` | `false` | Session summaries in SQLite |
| `ENABLE_EMBED_CACHE` | `false` | In-request embed cache (respects `X-No-Cache`) |
| `ENABLE_TOKENIZER_ESTIMATE` | `false` | Use tiktoken (`cl100k_base`) for context budget and injection truncation |

Full rollout guidance: [Cognitive pipeline](cognitive-pipeline.md) and [COGNITIVE_RAG_PLAN.md](COGNITIVE_RAG_PLAN.md).

## Latency budgets

| Variable | Default | Purpose |
| --- | --- | --- |
| `COGNITIVE_LATENCY_BUDGET_MS` | `800` | Global budget; stages skip when remaining ms is too low |
| `STAGE_BUDGET_ROUTING_MS` | `0` | Min ms for model routing |
| `STAGE_BUDGET_REWRITE_MS` | `20` | Min ms for query rewrite |
| `STAGE_BUDGET_RETRIEVE_MS` | `50` | Min ms for embed + Qdrant/hybrid |
| `STAGE_BUDGET_GRAPH_MS` | `100` | Min ms for graph lookup |
| `STAGE_BUDGET_MEMGRAPHRAG_MS` | `200` | Min ms for MemGraphRAG stage |
| `STAGE_EXEC_TIMEOUT_MS` | `30000` | Max runtime for stages without a dedicated budget (also caps tier0/intent/memory/context) |
| `RETRIEVAL_CANDIDATE_K` | `20` | Candidate pool before rerank/top-k trim |
| `CONTEXT_BUDGET_RATIO` | `0.25` | Fraction of context window for RAG chunks |
| `CONTEXT_FALLBACK_CHARS` | `8000` | Char fallback when tokenizer estimate off |
| `DEFAULT_COMPLETION_RESERVE` | `1024` | Reserved tokens for model reply |
| `TIER0_MAX_CHARS` | `80` | Max query length for tier0 heuristic bypass |

Rerank and tools use `RERANK_TIMEOUT_MS` and `TOOL_BUDGET_MS` as their stage minimums.

## Intent

| Variable | Default | Purpose |
| --- | --- | --- |
| `INTENT_MODEL` | *(empty)* | Optional model id on the upstream API for intent classification |
| `INTENT_CONFIDENCE_THRESHOLD` | `0.55` | Below this, intent may not gate retrieval |
| `INTENT_TIMEOUT_MS` | `150` | Intent call timeout |

## Hybrid retrieval and rerank

| Variable | Default | Purpose |
| --- | --- | --- |
| `HYBRID_DENSE_WEIGHT` | `0.7` | Dense weight in RRF merge |
| `SPARSE_INDEX_URL` | *(empty)* | BM25 sidecar base URL |
| `RECENCY_WEIGHT` | `0.1` | Recency boost on hits |
| `RERANKER_URL` | `http://127.0.0.1:8095` | Cross-encoder sidecar |
| `RERANK_TOP_K` | `5` | Chunks after rerank |
| `RERANK_TIMEOUT_MS` | `200` | Rerank stage budget/timeout |

## Graph, tools, memory, MemGraphRAG

| Variable | Default | Purpose |
| --- | --- | --- |
| `GRAPH_DB_PATH` | `/var/lib/rag_proxy/graph.sqlite` | Infra graph SQLite |
| `GRAPH_MAX_DEPTH` | `2` | Graph traversal depth |
| `TOOL_ALLOWED_ROOTS` | *(empty)* | Comma-separated absolute paths for file tools |
| `TOOL_TIMEOUT_SEC` | `5` | Per-tool call timeout |
| `TOOL_BUDGET_MS` | `300` | Tools stage min budget |
| `TOOL_MAX_OUTPUT_CHARS` | `4000` | Max chars read per tool |
| `MEMORY_DB_PATH` | `/var/lib/rag_proxy/memory.sqlite` | Rolling memory SQLite |
| `MEMORY_TTL_HOURS` | `72` | Session memory expiry |
| `MEMORY_REFRESH_TURNS` | `8` | Turns before memory refresh |
| `MEMGRAPHRAG_DB_PATH` | `/var/lib/rag_proxy/memgraphrag.sqlite` | MemGraphRAG index |
| `MEMGRAPHRAG_FACT_TOP_K` | `20` | Fact candidates for PPR |
| `MEMGRAPHRAG_PPR_DAMPING` | `0.85` | PageRank damping |
| `MEMGRAPHRAG_PPR_ITERATIONS` | `20` | PPR iterations |
| `MEMGRAPHRAG_PASSAGE_NODE_WEIGHT` | `0.5` | Passage node weight in graph |
| `MEMGRAPH_BUILD_LLM_URL` | `http://127.0.0.1:8080/v1` | LLM for offline entity/relation extraction |
| `MEMGRAPH_BUILD_LLM_MODEL` | `qwen3.5-9b-turbo` | Model name for build script |
| `MEMGRAPH_BUILD_MAX_CHUNKS` | `1000` | Qdrant sample size / chunk cap |
| `MEMGRAPH_BUILD_CONCURRENCY` | `3` | Parallel LLM calls during build |
| `MEMGRAPH_BUILD_EMBED_URL` | *(EMBED_URL)* | Embed endpoint for fact vectors at build time |
| `MEMGRAPH_BUILD_SKIP_RELATIONS` | `false` | Entity-only extraction (faster) |

Env-only defaults above come from `rag_proxy/config.py`. Rag-admin **Settings → MemGraphRAG index build** defaults `MEMGRAPH_BUILD_LLM_URL` to `http://192.168.1.202:8081/v1` (homelab remote qwen); override in Settings or env.

Build and rollout: [MemGraphRAG operator guide](memgraphrag.md). Rag-admin **Settings → MemGraphRAG index build** uses the same `MEMGRAPH_BUILD_*` keys (stored in admin SQLite; env vars are the fallback).

| Variable | Default | Purpose |
| --- | --- | --- |
| `MODEL_REGISTRY_TTL_SEC` | `300` | Cache TTL for `/v1/models` |
| `MODEL_CAPABILITIES_JSON` | *(empty)* | JSON overrides for model capabilities |
| `MODEL_ROUTES_JSON` | *(empty)* | JSON intent → model map |

## Transcript capture

Optional JSONL capture for fine-tuning exports and RAG improvement review. This is separate from observability logs and is disabled by default.

| Variable | Default | Purpose |
| --- | --- | --- |
| `ENABLE_TRANSCRIPT_CAPTURE` | `false` | Master switch for transcript JSONL capture |
| `FINETUNE_LOG_PATH` | `/var/lib/rag_proxy/capture/finetune.jsonl` | Sanitized request + assistant completion stream |
| `RAG_IMPROVEMENT_LOG_PATH` | `/var/lib/rag_proxy/capture/rag_improvement.jsonl` | Query, retrieval metadata, hits, and Q&A stream |
| `TRANSCRIPT_STRIP_PROXY_ARTEFACTS` | `true` | Remove RAG and rolling-memory system prefixes from captured messages |
| `TRANSCRIPT_HEADER_OPT_IN` | `false` | Require `X-Capture-Log: true` per request when enabled |
| `TRANSCRIPT_SAMPLE_RATE` | `1.0` | Capture sample rate from `0.0` to `1.0` |
| `TRANSCRIPT_HIT_PREVIEW_CHARS` | `300` | Max characters of each hit stored in RAG improvement records |
| `ENABLE_RAG_CORPUS_AUTO_INGEST` | `false` | Promote eligible RAG Q&A pairs to Qdrant after JSONL append |
| `RAG_CORPUS_COLLECTION` | `nomad_conversation_derived` | Derived Q&A collection; separate from `QDRANT_COLLECTION` by default |
| `RAG_CORPUS_MIN_ANSWER_CHARS` | `100` | Minimum answer length for auto-ingest promotion |
| `RAG_CORPUS_REQUIRE_CHUNKS` | `false` | Require at least one injected chunk before promotion |

Offline helpers:

```bash
python scripts/export_finetune_dataset.py --input /var/lib/rag_proxy/capture/finetune.jsonl --output finetune_messages.jsonl
python scripts/promote_rag_corpus.py --input /var/lib/rag_proxy/capture/rag_improvement.jsonl --dry-run
```

## Observability

| Variable | Default | Purpose |
| --- | --- | --- |
| `ENABLE_REQUEST_TRACE` | `true` | Per-request pipeline summary logs |
| `ENABLE_JSON_LOGS` | `false` | JSON pipeline logs instead of text |
| `ENABLE_METRICS` | `false` | Expose `GET /metrics` on proxy port |

Details: [Observability](observability.md).

## RAG admin and ingest (optional)

Used by `rag_admin/` and `ingest/` — separate from the proxy. Not required for proxy-only deploys.

| Variable | Default | Purpose |
| --- | --- | --- |
| `ADMIN_HOST` | `127.0.0.1` | Admin UI bind |
| `ADMIN_PORT` | `8087` | Admin UI port |
| `ADMIN_DB_PATH` | `/opt/ai/rag/admin.sqlite` | Admin SQLite |
| `ZIM_DIR` | `/opt/ai/rag/zim` | ZIM archive directory |
| `UPLOAD_DIR` | `/opt/ai/rag/uploads` | PDF/text uploads |
| `ADMIN_SESSION_SECRET` | *(must change)* | Session signing key |
| `ADMIN_PASSWORD` | *(must change)* | Login password |
| `ADMIN_SESSION_TTL_SECONDS` | `604800` (7 days) | Server-side session lifetime; cookie `max-age` matches |
| `ADMIN_LOGIN_MAX_ATTEMPTS` | `5` | Failed logins per client IP before lockout |
| `ADMIN_LOGIN_LOCKOUT_MINUTES` | `15` | Lockout window for failed logins (HTTP 429) |
| `ADMIN_ALLOW_INSECURE_DEFAULTS` | — | Set `true` for local dev only |

**Session cookies (Track D):** Login creates a row in `admin_sessions` (same SQLite as `ADMIN_DB_PATH`). The cookie value is `session_id.exp.hmac` where `hmac` signs `session_id.exp` with `ADMIN_SESSION_SECRET`. Logout sets `revoked_at` on the row. Auth checks HMAC, expiry, and that the row exists and is not revoked. **Upgrading invalidates** old static `authenticated` cookies — users must log in again.

**Login rate limit:** Per client IP: `CF-Connecting-IP` when present (preferred behind Cloudflare tunnel/proxy), else first hop of `X-Forwarded-For`, else `request.client.host`. After `ADMIN_LOGIN_MAX_ATTEMPTS` failures within `ADMIN_LOGIN_LOCKOUT_MINUTES`, `POST /login` returns HTTP 429 until the window elapses. Successful login clears the counter for that IP.

| Variable | Default | Purpose |
| --- | --- | --- |
| `INGEST_BATCH_SIZE` | `64` | Texts per embed HTTP request / Qdrant upsert batch |
| `INGEST_EMBED_CONCURRENCY` | `4` | Concurrent in-flight embed batches (match `llama-server --parallel`) |
| `INGEST_EMBED_URLS` | *(empty)* | Comma-separated embed endpoints for ingest round-robin (often generated by pool planner) |
| `INGEST_MAX_ARTICLES` | `0` | ZIM article cap (`0` = unlimited) |
| `INGEST_SPARSE_REINDEX` | `idle` | Sparse sidecar reindex mode |
| `INGEST_STALL_MINUTES` | `15` | Stall detection for ingest jobs |
| `INGEST_CHUNK_SIZE_TOKENS` | `512` | Target chunk size in tokens (nomic-embed range) |
| `INGEST_CHUNK_OVERLAP_TOKENS` | `64` | Chunk overlap in tokens (~12.5%) |
| `INGEST_CHUNK_TOKENIZER` | `nomic-ai/nomic-embed-text-v1.5` | Tokenizer for chunk sizing |
| `INGEST_CHUNK_SEMANTIC` | `true` | Use semantic chunking when deps installed |
| `INGEST_CHUNK_SEMANTIC_MODEL` | `minishlab/potion-base-32M` | Embedding model for semantic boundaries |
| `INGEST_CHUNK_MIN_TOKENS` | `100` | Merge adjacent chunks below this token count before embed |
| `INGEST_FILE_CONCURRENCY` | auto | Parallel file worker threads (planner-set; hot-reloads via `resize_file_workers`) |
| `INGEST_CHUNK_CONCURRENCY` | `min(4, cores/2)` | Concurrent chunk executions across file workers |
| `RAG_PROXY_URL` | `http://127.0.0.1:8081` | Proxy URL for admin smoke hooks |
| `RAG_ADMIN_ENV_FILE` | `/opt/ai/config/rag-admin.env` | Admin/ingest env file (Settings UI writes here) |
| `RAG_PROXY_ENV_FILE` | `/opt/ai/config/rag-proxy.env` | Proxy env file (Settings UI writes cognitive/proxy groups here) |
| `RAG_REPO_ROOT` | repo root | Path for admin background jobs |
| `RAG_ADMIN_JOB_LOG_DIR` | `/var/lib/rag_proxy/admin_jobs` | MemGraph build job logs |
| `RAG_PROXY_RESTART_CMD` | `systemctl restart rag-proxy` | Optional restart hook from Settings |
| `RAG_ADMIN_RESTART_CMD` | `systemctl restart rag-admin` | Optional restart hook from Settings |
| `NOMIC_EMBED_SCALE_ENV_FILE` | `/opt/ai/config/nomic-embed-scale.env` | Planner tuning caps (`NOMIC_POOL_*`, `INGEST_CAPACITY_*`) |
| `NOMIC_EMBED_POOL_ENV_FILE` | `/opt/ai/config/nomic-embed-pool.env` | Written plan (`INGEST_EMBED_URLS`, `INGEST_*`, `NOMIC_POOL_PARALLEL`) |
| `ARXIV_USER_AGENT` | *(built-in default)* | User-Agent for arXiv catalog API |

### Ingest capacity planner (bulk ingest)

Used by `scripts/scale_ingest_capacity.py` (legacy entry point `scale_nomic_embed_pool.py`) and `nomic-embed-scale.service`. GPU pool defaults from `ingest/embed_pool.py`, capacity caps from `ingest/capacity_planner.py`.

| Variable | Default | Purpose |
| --- | --- | --- |
| `NOMIC_POOL_VRAM_PER_INSTANCE_MIB` | `1024` | Estimated VRAM per pool instance |
| `NOMIC_POOL_VRAM_RESERVE_MIB` | `2048` | VRAM left for other GPU workloads |
| `NOMIC_POOL_PORT_BASE` | `18089` | First pool port (`nomic-embed@PORT`) |
| `NOMIC_POOL_MAX_INSTANCES` | `12` | Hard cap on pool size |
| `NOMIC_POOL_MIN_INSTANCES` | `1` | Floor when GPU sizing unavailable |
| `NOMIC_POOL_PARALLEL_PER_INSTANCE` | `16` | `--parallel` per pool unit (capped by GPU tier) |
| `NOMIC_POOL_GPU_INDEX` | `0` | `nvidia-smi` GPU index |
| `INGEST_CAPACITY_RAM_RESERVE_MIB` | `4096` | RAM headroom before file concurrency caps apply |
| `INGEST_CAPACITY_RAM_PER_FILE_MIB` | `2048` | Assumed RAM per concurrent file |
| `INGEST_CAPACITY_SEMANTIC_RAM_FLOOR_MIB` | `8192` | Below this available RAM, semantic chunking is disabled |
| `INGEST_CAPACITY_SEMANTIC_CPU_FLOOR` | `4` | Below this core count, semantic chunking is disabled |
| `INGEST_CAPACITY_CHUNK_CPU_SHARE` | `2` | Cores per concurrent chunking file |
| `INGEST_CAPACITY_MAX_FILE_CONCURRENCY` | `8` | Hard cap on parallel files |
| `INGEST_CAPACITY_MIN_DISK_MBPS` | `100` | Below this sequential read speed, file concurrency is capped |
| `INGEST_CAPACITY_SLOW_DISK_FILE_CAP` | `2` | File cap applied on slow storage |
| `INGEST_CAPACITY_SPARSE_REINDEX` | `off` | BM25 reindex mode written by the planner |

The planner writes all `INGEST_*` throughput knobs plus `NOMIC_POOL_*` metadata to the pool env file; the admin scale job syncs the ingest keys into the admin env and hot-reloads the worker. `NOMIC_POOL_PARALLEL` is now written by the planner so systemd `nomic-embed@.service` and the concurrency math stay aligned.

Details: [Ingest and admin](ingest-and-admin.md) and [Ingest capacity planning](ingest-capacity-planning.md).

## Sidecar services (optional)

Not loaded by `rag_proxy/config.py`. Set in Docker compose or sidecar unit env.

| Service | Key vars | Default bind |
| --- | --- | --- |
| Rerank (`sidecars/rerank/`) | `RERANK_MODEL`, `RERANK_HOST`, `RERANK_PORT` | `8095` |
| Sparse BM25 (`sidecars/sparse/`) | `SPARSE_HOST`, `SPARSE_PORT`, `SPARSE_REFRESH_SEC`, `SPARSE_SCROLL_BATCH`, `SPARSE_MAX_POINTS` | `8096` |
| MCP RAG (`sidecars/mcp_rag/`) | `MCP_HOST`, `MCP_PORT`, `MCP_TRANSPORT`, `MCP_RAG_USER_AGENT` | `9001` |

Proxy references rerank/sparse via `RERANKER_URL` and `SPARSE_INDEX_URL` only.

Details: [docker/README.md](../docker/README.md).

## Runtime config: proxy vs rag_admin

**rag_proxy** has no in-process settings API. Change `.env` (or systemd `EnvironmentFile`) and restart the proxy. Per-request overrides use HTTP headers — [Headers and clients](headers-and-clients.md).

**rag_admin** persists operator knobs via `GET /settings`, `POST /settings/save/{group}`, and `GET /api/settings/status`. Saved values are written to `RAG_ADMIN_ENV_FILE` / `RAG_PROXY_ENV_FILE`; ingest worker hot-reloads some keys; proxy and cognitive flags require a proxy restart.

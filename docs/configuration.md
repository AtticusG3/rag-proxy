# Configuration

All settings load from environment variables (`.env` or systemd `EnvironmentFile`). Defaults live in `rag_proxy/config.py`; [.env.example](../.env.example) is the operator template.

Restart the proxy after changing `.env`.

## Core proxy and retrieval

| Variable | Default | Purpose |
| --- | --- | --- |
| `LLAMA_SWAP_URL` | `http://127.0.0.1:8080` | Upstream llama-swap base URL |
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

Shared `httpx.AsyncClient` for all upstream traffic (llama-swap relay and embed/Qdrant calls).

| Variable | Default | Purpose |
| --- | --- | --- |
| `UPSTREAM_TIMEOUT_SEC` | `600` | Request timeout |
| `UPSTREAM_MAX_CONNECTIONS` | `50` | Connection pool cap |
| `UPSTREAM_MAX_KEEPALIVE` | `0` | Keepalive sockets (`0` = close after one-shot polls) |
| `UPSTREAM_KEEPALIVE_EXPIRY_SEC` | `15` | Keepalive socket expiry |
| `UPSTREAM_IDLE_SWEEP_SEC` | `30` | Janitor interval for abandoned streams |
| `UPSTREAM_STREAM_ABANDON_SEC` | `120` | Close upstream SSE when no bytes relayed for this long |

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
| `ENABLE_TOKENIZER_ESTIMATE` | `false` | Token-budget estimation in context assembly |

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
| `RETRIEVAL_CANDIDATE_K` | `20` | Candidate pool before rerank/top-k trim |
| `CONTEXT_BUDGET_RATIO` | `0.25` | Fraction of context window for RAG chunks |
| `CONTEXT_FALLBACK_CHARS` | `8000` | Char fallback when tokenizer estimate off |
| `DEFAULT_COMPLETION_RESERVE` | `1024` | Reserved tokens for model reply |
| `TIER0_MAX_CHARS` | `80` | Max query length for tier0 heuristic bypass |

Rerank and tools use `RERANK_TIMEOUT_MS` and `TOOL_BUDGET_MS` as their stage minimums.

## Intent

| Variable | Default | Purpose |
| --- | --- | --- |
| `INTENT_MODEL` | *(empty)* | Optional llama-swap model for intent classification |
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

Build and rollout: [MemGraphRAG operator guide](memgraphrag.md). Rag-admin **Settings → MemGraphRAG index build** uses the same `MEMGRAPH_BUILD_*` keys (stored in admin SQLite; env vars are the fallback).

| Variable | Default | Purpose |
| --- | --- | --- |
| `MODEL_REGISTRY_TTL_SEC` | `300` | Cache TTL for `/v1/models` |
| `MODEL_REGISTRY_CONFIG_PATH` | *(empty)* | Optional static registry file |
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
| `METRICS_PORT` | `0` | Legacy: `>0` also enables metrics when `ENABLE_METRICS` is unset/false |

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
| `ADMIN_ALLOW_INSECURE_DEFAULTS` | — | Set `true` for local dev only |
| `INGEST_BATCH_SIZE` | `64` | Texts per embed HTTP request / Qdrant upsert batch |
| `INGEST_EMBED_CONCURRENCY` | `4` | Concurrent in-flight embed batches (match `llama-server --parallel`) |
| `INGEST_MAX_ARTICLES` | `0` | ZIM article cap (`0` = unlimited) |
| `INGEST_SPARSE_REINDEX` | `idle` | Sparse sidecar reindex mode |
| `INGEST_STALL_MINUTES` | `15` | Stall detection for ingest jobs |
| `RAG_PROXY_URL` | `http://127.0.0.1:8081` | Proxy URL for admin smoke hooks |

Details: [Ingest and admin](ingest-and-admin.md).

## No JSON settings API

There is no admin API for runtime config. Change `.env` (or systemd `EnvironmentFile`) and restart the proxy. Per-request overrides use HTTP headers — [Headers and clients](headers-and-clients.md).

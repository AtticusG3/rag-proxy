# Proxy Codemap

**Last Updated:** 2026-07-27
**Entry Points:** `rag_proxy.py`, `python -m rag_proxy`, `rag_proxy/app.py`

## Architecture

```text
Client POST /v1/chat/completions
        |
        v
  rag_proxy/app.py  ---- GET /metrics, GET /debug (native)
        |
        v
  orchestrator.run_pipeline
        |
        +-- build_legacy_pipeline_stages()   # ENABLE_COGNITIVE_PIPELINE=false
        |     retrieve -> context
        |
        +-- build_pipeline_stages()          # cognitive on
              tier0 -> intent -> gating -> routing -> rewrite
              -> retrieve -> rerank -> graph -> memgraphrag
              -> tools -> memory -> context
        |
        v
  upstream_client.relay_upstream -> LLAMA_SWAP_URL
```

## Key Modules

| Module | Purpose | Notable exports / roles |
| --- | --- | --- |
| `rag_proxy/app.py` | FastAPI app, lifespan, catch-all proxy | `proxy`, `/metrics`, `/debug` |
| `rag_proxy/orchestrator.py` | Budget-aware stage loop, header parsing | pipeline runner |
| `rag_proxy/pipeline_stages.py` | Stage registry | `build_pipeline_stages`, `build_legacy_pipeline_stages` |
| `rag_proxy/retrieval_policy.py` | Tier0 + gating decisions | skip/light/full |
| `rag_proxy/context.py` | Request state | `RequestContext`, enums |
| `rag_proxy/config.py` | Env-backed settings | `settings`, `CHAT_PATHS` |
| `rag_proxy/upstream_client.py` | Shared httpx pool + stream janitor | `relay_upstream`, `UPSTREAM_*` |
| `rag_proxy/clients/qdrant.py` | Async hybrid search for proxy | `hybrid_search` |
| `rag_proxy/clients/retrieval_core.py` | Shared payload/parse helpers | dense/sparse/TurboVec payloads |
| `rag_proxy/clients/retrieve_sync.py` | Sync retrieval (MCP / helpers) | `hybrid_retrieve` |
| `rag_proxy/legacy_rag.py` | Extract / inject helpers | query extract, context inject |
| `rag_proxy/observability.py` | Traces, metrics text | `GET /metrics` rendering |
| `rag_proxy/capture.py` | Transcript capture opt-in | JSONL finetune / RAG improvement |
| `rag_proxy/memgraphrag/` | Three-layer memory + PPR | `ThreeLayerMemory`, `MemGraphRetriever` |
| `rag_proxy/stages/*` | Per-stage implementations | tier0–tier3 |

## Data Flow

1. Chat `POST` only (`CHAT_PATHS`); other paths proxy through.
2. Pipeline fills `RequestContext.hits` (fail-open on stage errors).
3. Context stage injects chunk text into the system message.
4. Body forwarded via shared upstream pool; SSE relayed with abandon janitor.
5. Optional transcript capture runs after the pipeline (not a pipeline stage).

## External Dependencies

- Upstream OpenAI-compatible API (`LLAMA_SWAP_URL`)
- nomic-embed (`EMBED_URL`)
- Qdrant (`QDRANT_URL` / `QDRANT_COLLECTION`)
- Optional: TurboVec, sparse BM25, reranker, MemGraphRAG SQLite

## Related Areas

- [ingest.md](ingest.md) — indexing into Qdrant / TurboVec dual-write
- [sidecars.md](sidecars.md) — HTTP services the proxy calls
- [Architecture](../architecture.md) — operator narrative

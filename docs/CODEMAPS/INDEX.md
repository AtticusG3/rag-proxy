# Codemaps

**Last Updated:** 2026-09-22

Module maps for contributors and agents. Operator guides stay under [docs/README.md](../README.md). Path-level agent notes: [AGENTS.md](../../AGENTS.md).

## Areas

| Codemap | Scope |
| --- | --- |
| [proxy.md](proxy.md) | `rag_proxy/` FastAPI proxy, cognitive pipeline, retrieval clients |
| [ingest.md](ingest.md) | `ingest/` worker, chunking, embed, Qdrant write path |
| [admin.md](admin.md) | `rag_admin/` UI, settings, catalog, ingest queue |
| [sidecars.md](sidecars.md) | Rerank, sparse BM25, TurboVec, MCP RAG |

## System overview

```text
Clients / MCP
    |                    +-- EMBED_URL (Qwen embed-pool / OpenAI-compatible)
    v                    |
rag_proxy (:8088) -----> Qdrant (+ optional TurboVec / sparse / rerank)
    |                    |
    +---> LLAMA_SWAP_URL (OpenAI-compatible upstream; often remote)

rag_admin (:8087) ---> ingest worker ---> EMBED_URL + Qdrant (+ sidecars)

Docker default: rag-proxy container only; Qdrant/chat/embed via docker/.env URLs
Optional compose profiles: qdrant, cognitive, llama-swap (see docker/README.md)
```

## Related operator docs

- [Architecture](../architecture.md)
- [Configuration](../configuration.md)
- [Ingest and admin](../ingest-and-admin.md)
- [MemGraphRAG](../memgraphrag.md)

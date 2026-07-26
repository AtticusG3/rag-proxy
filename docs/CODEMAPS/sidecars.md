# Sidecars Codemap

**Last Updated:** 2026-07-27
**Entry Points:** each `sidecars/*/app.py` (Docker compose profiles / systemd units)

## Architecture

```text
rag_proxy / ingest / MCP
        |
        +-- RERANKER_URL ---------> sidecars/rerank   (:8095)
        +-- SPARSE_INDEX_URL -----> sidecars/sparse   (:8096)
        +-- TURBOVEC_URL ---------> sidecars/turbovec (:8097)
        +-- MCP HTTP/stdio -------> sidecars/mcp_rag  (:9001)
```

## Key Modules

| Service | Path | Purpose | Typical port |
| --- | --- | --- | --- |
| Rerank | `sidecars/rerank/` | Cross-encoder reorder | `8095` |
| Sparse BM25 | `sidecars/sparse/` | Sparse index + search; Qdrant scroll rebuild | `8096` |
| TurboVec | `sidecars/turbovec/` | TurboQuant dense ANN (`/search`, `/add`, `/reindex`, `/save`) | `8097` |
| MCP RAG | `sidecars/mcp_rag/` | MCP tools: KB search modes + personal `memory_*` | `9001` |

Each HTTP sidecar follows `app.py` + `core.py` (+ `Dockerfile` / `requirements.txt`).

## MCP tools (`sidecars/mcp_rag/`)

| Tool | Role |
| --- | --- |
| `search_knowledge_base` | Modes: `passages` (`hybrid` alias), `dense`, `sparse`, `facts` |
| `knowledge_base_status` | Health of Qdrant / sparse / embed / MemGraph |
| `memory_store` / `memory_recall` / `memory_forget` | Personal SQLite notes (not KB ingest) |

See [`sidecars/mcp_rag/README.md`](../../sidecars/mcp_rag/README.md).

## Data Flow

- Proxy retrieve/rerank stages call sidecars over HTTP; failures fail-open.
- Ingest dual-writes TurboVec when `TURBOVEC_URL` is set; sparse/TurboVec reindex modes are operator-controlled.
- Admin can start/stop sidecars on demand (Linux + systemd) around ingest work.

## Related Areas

- [proxy.md](proxy.md)
- [Configuration — Sidecar services](../configuration.md#sidecar-services-optional)
- [Configuration — TurboVec rollout](../configuration.md#turbovec-rollout-cut-qdrant-ram)
- [docker/README.md](../../docker/README.md)

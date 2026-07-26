# MCP RAG context server

Exposes curated KB retrieval and personal agent memory as MCP tools for Hermes, Cursor, OpenClaw, etc.

## Tools

| Tool | Purpose |
| --- | --- |
| `search_knowledge_base` | Read curated KB. Modes: `passages` (default), `dense`, `sparse`, `facts` (MemGraphRAG). |
| `knowledge_base_status` | Qdrant / sparse / embed / MemGraph health |
| `memory_store` | Upsert a personal note (separate SQLite; **not** Qdrant) |
| `memory_recall` | Lexical recall over personal notes only |
| `memory_forget` | Delete a personal note by key |

`min_score` on search applies to the **dense** similarity leg only; it does not filter BM25 or MemGraph fact scores.

## Run locally

```bash
pip install -r requirements.txt
export EMBED_URL=http://127.0.0.1:8089
export QDRANT_URL=http://127.0.0.1:6333
# optional hybrid:
# export SPARSE_INDEX_URL=http://127.0.0.1:8096
# export ENABLE_HYBRID_RETRIEVAL=true
# export RERANKER_URL=http://127.0.0.1:8095
# export ENABLE_RERANKER=true
python app.py
```

Default transport: streamable HTTP at `http://127.0.0.1:9001/mcp`. Set `MCP_TRANSPORT=stdio` for stdio mode.

Personal store path: `MCP_PERSONAL_STORE_PATH` (default `~/.local/share/rag_proxy/mcp_personal.sqlite`).

## Production (systemd)

Run under your process manager with the same env as rag-proxy (`EMBED_URL`, `QDRANT_URL`, optional `SPARSE_INDEX_URL` / `RERANKER_URL` / `MEMGRAPHRAG_DB_PATH`) plus `MCP_*` vars. Keep `MCP_HOST=127.0.0.1` unless you add auth in front.

## Hermes / OpenClaw

Point the agent MCP client at this server. Use `memory_*` for durable personal preferences; use `search_knowledge_base` for offline docs. Never treat memory tools as KB ingest.

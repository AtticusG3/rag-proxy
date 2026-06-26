# MCP RAG context server

Exposes hybrid retrieval (`search_knowledge_base`, `knowledge_base_status`) as MCP tools for agents (Hermes, Cursor, etc.).

## Run locally

```bash
pip install -r requirements.txt
export EMBED_URL=http://127.0.0.1:18089
export QDRANT_URL=http://127.0.0.1:6333
export SPARSE_INDEX_URL=http://127.0.0.1:18096
export RERANKER_URL=http://127.0.0.1:18095
python app.py
```

Default transport: streamable HTTP at `http://127.0.0.1:9001/mcp`. Set `MCP_TRANSPORT=stdio` for stdio mode.

## Production (systemd)

Run under your process manager with `EMBED_URL`, `QDRANT_URL`, and optional `SPARSE_INDEX_URL` / `RERANKER_URL` set in the unit environment or an `EnvironmentFile`.

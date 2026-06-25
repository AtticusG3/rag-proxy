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

## Homelab (buster)

See `local-ai-infra/nodes/buster/systemd/mcp-rag-context.service` in the homelab repo.

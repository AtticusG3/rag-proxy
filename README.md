# rag_proxy

Transparent RAG middleware in front of [llama-swap](https://github.com/mostlygeek/llama-swap). Clients point at this proxy instead of llama-swap directly; chat requests can be augmented with Qdrant context before they reach any model.

## Architecture

### Default (legacy) mode

`ENABLE_COGNITIVE_PIPELINE=false` (default): same as before — always embed, dense Qdrant search, inject.

```
Client
  -> rag_proxy :8088
        | embed last user message (nomic-embed :8089)
        | search Qdrant :6333
        | inject retrieved chunks as system context
        -> llama-swap :8080 -> llama-server models
```

### Cognitive pipeline (optional)

`ENABLE_COGNITIVE_PIPELINE=true` runs tiered stages; each subsystem has its own `ENABLE_*` flag (see [.env.example](.env.example)).

```
Client -> rag_proxy :8088
           |
     Tier 0  heuristics (bypass embed/Qdrant for simple queries)
           |
     Tier 1  intent + retrieval gating (+ optional tiny classifier)
           |
     Tier 2  rewrite -> hybrid retrieve -> rerank -> dedupe/budget -> inject
           |
     Tier 3  graph / tools / rolling memory (rare, off by default)
           |
           -> llama-swap :8080
```

| Tier | Typical added latency | Default |
|------|----------------------|---------|
| 0 Heuristics | 1-15 ms | `ENABLE_TIER0_HEURISTICS` |
| 1 Intent + gating | 20-100 ms | off |
| 2 Retrieval | 100-500 ms | dense path when not skipped |
| 3 Heavy | 1-3 s | off |

- **Auth**: unchanged. API keys are validated by llama-swap downstream; this proxy forwards headers as-is.
- **Passthrough**: non-chat routes and non-POST traffic are proxied without modification.
- **Failure mode**: cognitive or RAG errors never break the request; the original body is forwarded.

### Client headers (cognitive mode)

| Header | Values | Effect |
|--------|--------|--------|
| `X-RAG-Mode` | `off`, `auto`, `force` | Skip RAG, default pipeline, or force retrieval |
| `X-No-Cache` | `true` | Bypass embed/retrieval caches |
| `X-Conversation-Id` | string | Rolling memory session key |

## Prerequisites

| Service | Role | Default |
|---------|------|---------|
| llama-swap | Model router | `http://127.0.0.1:8080` |
| nomic-embed (llama-server `--embedding`) | Query vectors | `http://127.0.0.1:8089` |
| Qdrant | Vector store | set `QDRANT_URL` |
| Sparse index (optional) | BM25 sidecar | `SPARSE_INDEX_URL` |
| Reranker (optional) | Cross-encoder HTTP | `RERANKER_URL` |

`config.yaml` is an example llama-swap config (paths are host-specific).

## Quick start (Linux)

```bash
cd rag_proxy
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env -- at minimum set QDRANT_URL

python rag_proxy.py
# or: python -m rag_proxy
```

Point OpenAI-compatible clients at `http://<host>:8088/v1` (same paths as llama-swap).

## Environment

See [.env.example](.env.example) and `rag_proxy/config.py`. Core variables:

| Variable | Purpose |
|----------|---------|
| `QDRANT_URL` | Qdrant HTTP API base |
| `ENABLE_COGNITIVE_PIPELINE` | Use tiered pipeline vs legacy always-retrieve |
| `ENABLE_TIER0_HEURISTICS` | Fast-path bypass for simple queries |
| `ENABLE_RETRIEVAL_GATING` | Skip retrieval when not needed |
| `GATING_LOG_ONLY` | Log skip decisions without skipping (bake-in on nomad) |

Full operator guide: [docs/COGNITIVE_RAG_PLAN.md](docs/COGNITIVE_RAG_PLAN.md).

## systemd

```bash
sudo cp nomic-embed.service rag-proxy.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now nomic-embed rag-proxy
```

## Tests

```bash
pip install pytest
pytest tests/ -q
```

Offline unit tests only (no live Qdrant or embed server).

## Nomad rollout (recommended)

1. Deploy package extract; keep `ENABLE_COGNITIVE_PIPELINE=false` (no behavior change).
2. Set `ENABLE_COGNITIVE_PIPELINE=true`, `ENABLE_TIER0_HEURISTICS=true`, `GATING_LOG_ONLY=true` — observe logs.
3. Set `ENABLE_RETRIEVAL_GATING=true`, `GATING_LOG_ONLY=false`.
4. Enable one flag per week: intent, rewrite, hybrid, reranker, tier 3.

## RAG behavior (legacy path)

On `POST /v1/chat/completions` or `POST /api/chat`:

1. Last user message text is embedded via `nomic-embed-text-v1.5`.
2. Qdrant vector search returns up to `TOP_K` hits above `SIMILARITY_THRESHOLD`.
3. Chunk text from payload fields: `text`, `content`, `chunk`, `document`, `page_content`.
4. Chunks prepended to system message (or new system message inserted).

## Embedding size limits

Tail-truncate to `EMBED_MAX_CHARS` (default 2000). On batch overflow, retry at 1200 chars. See nomic-embed.service `-ub 2048` for large inputs.

## License

Private / homelab.

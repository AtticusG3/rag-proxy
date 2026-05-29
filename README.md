# rag_proxy

Transparent RAG middleware in front of [llama-swap](https://github.com/mostlygeek/llama-swap). Clients point at this proxy instead of llama-swap directly; chat requests can be augmented with Qdrant context before they reach any model.

**You do not change client API keys** — point the OpenAI-compatible base URL at the proxy; llama-swap still validates auth downstream.

## First-time setup

1. **Start dependencies** (in order): Qdrant with your collection, `nomic-embed` on `:8089`, llama-swap on `:8080`.
2. **Clone and install** (see [Quick start](#quick-start-linux) below).
3. **Copy and edit `.env`** — at minimum set `QDRANT_URL` and `QDRANT_COLLECTION` to your vector store. Defaults assume everything runs on the same host (`127.0.0.1`).
4. **Run the proxy**: `python rag_proxy.py` — startup logs list embed, Qdrant, and cognitive mode.
5. **Point your client** at `http://<host>:8088/v1` instead of llama-swap `:8080`.
6. **Verify** with the smoke tests in [Verify the stack](#verify-the-stack) below.

### Minimum `.env` checklist

| Variable | You must set | Typical value |
|----------|--------------|---------------|
| `QDRANT_URL` | Yes | `http://192.168.1.36:6333` |
| `QDRANT_COLLECTION` | If not `nomad_knowledge_base` | your collection name |
| `LLAMA_SWAP_URL` | If llama-swap is not local | `http://127.0.0.1:8080` |
| `EMBED_URL` | If embed server is not local | `http://127.0.0.1:8089` |

Leave `ENABLE_COGNITIVE_PIPELINE=false` until you have verified legacy RAG works. All other vars have safe defaults in `.env.example`.

## Point your client at the proxy

Use the **same paths and API key** as llama-swap; only the base URL changes.

| Client | Setting | Value |
|--------|---------|-------|
| Open WebUI | Settings → Connections → OpenAI API → URL | `http://<host>:8088/v1` |
| Continue / Cursor | OpenAI base URL override | `http://<host>:8088/v1` |
| `curl` / scripts | `-H "Authorization: Bearer …"` unchanged | POST to `http://<host>:8088/v1/chat/completions` |

RAG runs only on `POST /v1/chat/completions` and `POST /api/chat`. Everything else (models list, embeddings route on llama-swap, health) passes through unchanged.

## Verify the stack

Replace host/ports with your `.env` values. Expect JSON responses; errors usually mean a service is down or the URL is wrong.

```bash
# 1. Embed server (must return embedding vector)
curl -s -X POST "http://127.0.0.1:8089/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{"model":"nomic-embed-text-v1.5","input":"test query"}'

# 2. Qdrant collection exists
curl -s "http://127.0.0.1:6333/collections/nomad_knowledge_base"

# 3. Proxy forwards to llama-swap (no RAG required for this call)
curl -s "http://127.0.0.1:8088/v1/models"

# 4. RAG path — send a question that should match your knowledge base
curl -s -X POST "http://127.0.0.1:8088/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_KEY" \
  -d '{"model":"your-model","messages":[{"role":"user","content":"question about your indexed docs"}],"stream":false}'
```

**Success signals in proxy logs** (`LOG_LEVEL=INFO`):

- `RAG: injected N chunk(s) (scores: …)` — retrieval worked; context was added to the system message.
- `RAG: no chunks above threshold=…` — search ran but nothing scored high enough; try lowering `SIMILARITY_THRESHOLD` or rephrase the query.
- `QDRANT_URL still has placeholder` — fix `.env` before expecting retrieval.

For cognitive mode, look for `trace=… tier=… retrieval=… chunks=…` lines (see [docs/COGNITIVE_RAG_PLAN.md](docs/COGNITIVE_RAG_PLAN.md)).

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
     tier0 -> intent -> gating -> routing
           |
     rewrite -> retrieve -> rerank -> context (inject)
           |
     graph -> tools -> memory  (tier 3; off by default)
           |
           -> llama-swap :8080
```

Stages are registered in `pipeline_stages.py`; the orchestrator skips disabled stages or those below per-stage budget (`STAGE_BUDGET_*`). Retrieval policy (`retrieval_policy.py`) drives tier0 bypass and gating skip/light/full.

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
$EDITOR .env   # set QDRANT_URL and QDRANT_COLLECTION

python rag_proxy.py
# or: python -m rag_proxy
```

On Windows (local dev only): use `\.venv\Scripts\activate` and the same `pip` / `python rag_proxy.py` steps; production deploy targets Linux systemd below.

## Environment

See [.env.example](.env.example) and `rag_proxy/config.py`. Core variables:

| Variable | Purpose |
|----------|---------|
| `QDRANT_URL` | Qdrant HTTP API base |
| `ENABLE_COGNITIVE_PIPELINE` | Use tiered pipeline vs legacy always-retrieve |
| `ENABLE_TIER0_HEURISTICS` | Fast-path bypass for simple queries |
| `ENABLE_RETRIEVAL_GATING` | Skip retrieval when not needed |
| `GATING_LOG_ONLY` | Log skip decisions without skipping (bake-in on nomad) |
| `STAGE_BUDGET_*` | Min ms remaining before routing/rewrite/retrieve/graph stages run |
| `ENABLE_REQUEST_TRACE` | Per-request pipeline summary logs (default on) |
| `ENABLE_METRICS` | `GET /metrics` on proxy port (not a separate listener) |

Full operator guide: [docs/COGNITIVE_RAG_PLAN.md](docs/COGNITIVE_RAG_PLAN.md).

## systemd (production)

1. **Edit unit files** before install — paths in `rag-proxy.service` and `nomic-embed.service` are host-specific (`User`, `WorkingDirectory`, model path, `ExecStart`).
2. **Install units**:

```bash
sudo cp nomic-embed.service rag-proxy.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now nomic-embed rag-proxy
```

3. **Check status and logs**:

```bash
sudo systemctl status rag-proxy nomic-embed
journalctl -u rag-proxy -f          # follow proxy logs (RAG inject lines appear here)
journalctl -u rag-proxy --since today | grep -E 'RAG:|trace='
```

`rag-proxy.service` loads `/home/kevyn/rag_proxy/.env` via `EnvironmentFile`. After changing `.env`, run `sudo systemctl restart rag-proxy`.

**Common startup failures**

| Symptom | Fix |
|---------|-----|
| `status=203/EXEC` | `.venv` missing or wrong path in `ExecStart` — recreate venv at `WorkingDirectory` |
| Proxy up, never injects | `QDRANT_URL` wrong or embed down — run [Verify the stack](#verify-the-stack) |
| `Address already in use` | Another process on `PROXY_PORT` (8088) |

## Observability (optional)

- **Traces**: `ENABLE_REQUEST_TRACE=true` (default) logs per-request summaries with `trace_id`, stage latencies, and `stage_trace`. Set `ENABLE_JSON_LOGS=true` for structured JSON instead of text.
- **Metrics**: `ENABLE_METRICS=true` exposes `GET /metrics` on the main proxy (`http://<proxy_host>:<proxy_port>/metrics`). Counters: `rag_requests_total`, `rag_chunks_injected_total` (cognitive and legacy paths). Legacy: non-zero `METRICS_PORT` also enables metrics when `ENABLE_METRICS` is unset/false. Not a separate listener — lightweight homelab stub, not a full Prometheus client.

## Tests

```bash
pip install pytest
pytest tests/ -q
```

Offline unit tests only (no live Qdrant or embed server).

## Nomad rollout (recommended)

Do not enable cognitive flags until legacy RAG injects chunks reliably (see [Verify the stack](#verify-the-stack)).

1. Deploy package extract; keep `ENABLE_COGNITIVE_PIPELINE=false` (no behavior change).
2. Set `ENABLE_COGNITIVE_PIPELINE=true`, `ENABLE_TIER0_HEURISTICS=true`, `GATING_LOG_ONLY=true` — observe logs; gating still retrieves but logs what it *would* skip.
3. Set `ENABLE_RETRIEVAL_GATING=true`, `GATING_LOG_ONLY=false` — confirm simple greetings skip embed/Qdrant (`RAG: skipped retrieval` in logs).
4. Enable one flag per week: intent, rewrite, hybrid, reranker, tier 3. Full walkthrough: [docs/COGNITIVE_RAG_PLAN.md](docs/COGNITIVE_RAG_PLAN.md#enabling-cognitive-mode-step-by-step).

## Troubleshooting

| Problem | What to check |
|---------|----------------|
| Chat works but no KB context | `LOG_LEVEL=DEBUG`; look for `no chunks above threshold` — lower `SIMILARITY_THRESHOLD` (e.g. `0.55`) or confirm Qdrant has vectors for that topic |
| Never injects | `curl` Qdrant collection; confirm `QDRANT_URL` has no placeholder; confirm embed server responds |
| Only some messages get RAG | Open WebUI "follow-up" / `### Task:` prompts are skipped by design — use a normal user question |
| Streaming broken | Usually upstream llama-swap; proxy forwards SSE as-is — test same request against `:8080` directly |
| Cognitive mode feels random | Enable `ENABLE_REQUEST_TRACE=true`; read `trace=` lines for `retrieval=skip` vs `full` |
| Request still works when RAG fails | Expected — fail-open by design; fix logs warnings, do not expect 5xx from RAG errors |

Per-request overrides (cognitive mode): send header `X-RAG-Mode: force` to always retrieve, or `off` to skip RAG for one request.

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

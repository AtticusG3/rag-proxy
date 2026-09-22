# Docker stack (rag-proxy)

Clients use **rag-proxy** on port **8088** (`/v1`). Chat (`LLAMA_SWAP_URL`) and embeddings (`EMBED_URL`) are **external** — set them in `docker/.env` so the proxy can run on a different host from your LLMs.

Optional compose **profiles**:

| Profile | Services |
| --- | --- |
| *(default)* | `rag-proxy` only |
| `qdrant` | Bundled Qdrant |
| `cognitive` | Rerank, BM25 sparse, TurboVec sidecars |
| `llama-swap` | Bundled GPU llama-swap (single-box dev; needs `MODELS_DIR` + `docker/config.yaml`) |

Embeddings use an OpenAI-compatible API (`EMBED_MODEL=embed-pool` → Qwen3 Embedding 4B, 2560-d). There is no nomic-embed container.

## Prerequisites

- Docker Compose v2
- Reachable `QDRANT_URL`, `LLAMA_SWAP_URL`, and `EMBED_URL` from the rag-proxy container network
- For `--profile llama-swap`: NVIDIA Container Toolkit, `MODELS_DIR`, and `docker/config.yaml`
- For full cognitive: Qdrant collection with chunk payloads matching `QDRANT_VECTOR_SIZE` (2560 for Qwen3)

## Deploy modes

### Proxy on its own host (typical)

```bash
cp docker/.env.example docker/.env
# Set QDRANT_URL, LLAMA_SWAP_URL, EMBED_URL to your real hosts (not 127.0.0.1 unless host-networking)

docker compose up -d --build
```

### Proxy + bundled Qdrant + cognitive sidecars

```bash
cp docker/.env.example docker/.env
cp docker/.env.homelab.example docker/.env.homelab
# QDRANT_URL=http://qdrant:6333 in docker/.env; merge homelab flags from .env.homelab.example
docker compose --profile qdrant --profile cognitive up -d --build
```

### Single-box dev (optional bundled llama-swap)

```bash
cp docker/.env.example docker/.env
cp docker/config.yaml.example docker/config.yaml
# Uncomment/set MODELS_DIR and llama-swap profile vars in docker/.env
# LLAMA_SWAP_URL=http://llama-swap:8080
# EMBED_URL=http://llama-swap:8080

docker compose --profile llama-swap up -d --build
```

First **reranker** start downloads `BAAI/bge-reranker-base` (~400MB). **sparse-index** scrolls Qdrant on the first start (may take minutes on large collections), then writes an mmap snapshot under `SPARSE_INDEX_PATH` so later restarts skip the scroll.

## Smoke tests

```bash
# Passthrough (requires LLAMA_SWAP_URL reachable from container)
curl -s "http://127.0.0.1:8088/v1/models"

# Chat (legacy RAG when cognitive off)
curl -s -X POST "http://127.0.0.1:8088/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer no-key" \
  -d '{"model":"your-upstream-model","messages":[{"role":"user","content":"hello"}]}'

# Sidecars (cognitive profile)
curl -s "http://127.0.0.1:8095/health"
curl -s "http://127.0.0.1:8096/health"
curl -s -X POST "http://127.0.0.1:8096/reindex" -H "Content-Type: application/json" -d '{}'
```

Use the same API key and paths as your upstream; only the client base URL changes to `http://<proxy-host>:8088/v1`.

## Service map

| Service | Profile | Image | Port | Role |
|---------|---------|-------|------|------|
| `rag-proxy` | default | build `Dockerfile` | 8088 | RAG injection, client entry |
| `qdrant` | `qdrant` | `qdrant/qdrant` | 6333 | Vector DB (optional) |
| `llama-swap` | `llama-swap` | `llama-swap:cuda` | 8080 | Optional GPU chat + embed-pool |
| `reranker` | `cognitive` | `sidecars/rerank` | 8095 | Cross-encoder rerank |
| `sparse-index` | `cognitive` | `sidecars/sparse` | 8096 | BM25 hybrid retrieval |
| `turbovec-index` | `cognitive` | `sidecars/turbovec` | 8097 | TurboQuant dense ANN |

URLs:

| Env | Set in | Notes |
|-----|--------|-------|
| `LLAMA_SWAP_URL` | `docker/.env` | Remote chat API; required |
| `EMBED_URL` | `docker/.env` | Remote Qwen `/v1/embeddings`; required for RAG |
| `QDRANT_URL` | `docker/.env` | Remote or `http://qdrant:6333` with `--profile qdrant` |
| `RERANKER_URL` | compose | `http://reranker:8095` when cognitive profile on |
| `SPARSE_INDEX_URL` | compose | `http://sparse-index:8096` when cognitive profile on |
| `TURBOVEC_URL` | compose | `http://turbovec-index:8097` when cognitive profile on |

## Sidecar APIs

Contract matches `rag_proxy/stages/tier2_rerank.py` and `rag_proxy/clients/qdrant.py`.

**Reranker** — `POST /rerank`

```json
{"pairs": [{"query": "...", "document": "..."}], "top_k": 5}
-> {"indices": [2, 0, 1]}
```

**Sparse index** — `POST /search`

```json
{"query": "...", "limit": 20, "collection": "your_collection"}
-> {"results": [{"id": "...", "score": 1.2, "payload": {"text": "..."}}]}
```

**Sparse reindex** — `POST /reindex` (after Qdrant ingest)

```json
{"collection": "your_collection"}
```

Sparse `/search` uses the `collection` field to select the synced BM25 index. Reindex after ingest; unknown collections return empty results (fail-open). Zero-score BM25 hits are omitted (small collections may return fewer than `limit` results).

**TurboVec** — `POST /search` (set `DENSE_BACKEND=turbovec` on the proxy)

```json
{"vector": [0.1, ...], "limit": 5, "score_threshold": 0.0}
-> {"results": [{"id": "md5hex...", "score": 0.82}]}
```

Payloads are fetched from Qdrant by id in the proxy. Dual-write via ingest `POST /add`; rebuild with `POST /reindex`.

## Cognitive rollout

1. Start with legacy (`ENABLE_COGNITIVE_PIPELINE=false`) and verify injection.
2. Copy homelab flags from `docker/.env.homelab.example` into `docker/.env`.
3. `docker compose --profile qdrant --profile cognitive up -d --build`
4. Enable subsystems one at a time if debugging (see `docs/COGNITIVE_RAG_PLAN.md`).

Optional **intent** model: configure on your upstream llama-swap host (`docker/config.yaml` when using the llama-swap profile) and set `ENABLE_INTENT_ROUTER=true` + `INTENT_MODEL=<model-id>` in `.env`.

## External Qdrant

For an existing Qdrant host, set `QDRANT_URL` in `docker/.env` and omit `--profile qdrant`:

```bash
# docker/.env
QDRANT_URL=http://192.168.1.10:6333
```

```bash
docker compose --profile cognitive up -d --build
```

## Docker-in-Docker models (llama-swap profile)

If `docker/config.yaml` uses `docker run` in `cmd`, mount the host socket into llama-swap ([llama-swap wiki](https://github.com/mostlygeek/llama-swap/wiki/Docker-in-Docker-with-llama%E2%80%90swap-guide)):

```yaml
# llama-swap service volumes:
#   - /var/run/docker.sock:/var/run/docker.sock
#   - /usr/bin/docker:/usr/bin/docker
```

## Rebuild after code changes

```bash
docker compose build rag-proxy reranker sparse-index
docker compose --profile qdrant --profile cognitive up -d
```

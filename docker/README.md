# Docker stack (rag-proxy + llama-swap:cuda)

Clients use **rag-proxy** on port **8088** (`/v1`). Optional **cognitive** profile adds CPU rerank, BM25 sparse, and TurboVec dense sidecars.

Embeddings are an **OpenAI-compatible API** (`EMBED_URL` + `EMBED_MODEL=embed-pool` → Qwen3 Embedding 4B, 2560-d). Stock compose does **not** start a nomic-embed container. Default `docker/.env` points `EMBED_URL` at bundled llama-swap; set it to any Qwen embeddings API instead.

## Prerequisites

- Docker Compose v2
- NVIDIA Container Toolkit (for bundled `llama-swap:cuda`; skip if chat and embed URLs are external)
- Host directory with GGUF models if you run bundled llama-swap (chat model + Qwen3 embedding)
- For full cognitive: Qdrant collection populated with chunk payloads (`text`, `content`, etc.) matching `QDRANT_VECTOR_SIZE` (2560 for Qwen3)

## Deploy modes

### Legacy RAG (minimal)

```bash
cp docker/.env.example docker/.env
cp docker/config.yaml.example docker/config.yaml
# Edit MODELS_DIR, QDRANT_URL, chat/embed GGUF paths, or set EMBED_URL to an external Qwen API

docker compose up -d --build
```

### Full homelab (Qdrant + cognitive sidecars)

```bash
cp docker/.env.example docker/.env
cp docker/.env.homelab.example docker/.env.homelab
cp docker/config.yaml.example docker/config.yaml
# Copy homelab flags from docker/.env.homelab.example into docker/.env
docker compose --profile qdrant --profile cognitive up -d --build
```

First **reranker** start downloads `BAAI/bge-reranker-base` (~400MB). **sparse-index** scrolls Qdrant on the first start (may take minutes on large collections), then writes an mmap snapshot under `SPARSE_INDEX_PATH` so later restarts skip the scroll.

## Smoke tests

```bash
# Passthrough
curl -s "http://127.0.0.1:8088/v1/models"

# Chat (legacy RAG when cognitive off)
curl -s -X POST "http://127.0.0.1:8088/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer no-key" \
  -d '{"model":"chat-demo","messages":[{"role":"user","content":"hello"}]}'

# Sidecars (cognitive profile)
curl -s "http://127.0.0.1:8095/health"
curl -s "http://127.0.0.1:8096/health"
curl -s -X POST "http://127.0.0.1:8096/reindex" -H "Content-Type: application/json" -d '{}'
```

Use the same API key and paths as llama-swap; only the base URL changes to `http://<host>:8088/v1`.

## Service map

| Service | Profile | Image | Port | Role |
|---------|---------|-------|------|------|
| `rag-proxy` | default | build `Dockerfile` | 8088 | RAG injection, client entry |
| `llama-swap` | default | `llama-swap:cuda` | 8080 | GPU chat router + Qwen embeddings (`embed-pool`) unless URLs are overridden |
| `qdrant` | `qdrant` | `qdrant/qdrant` | 6333 | Vector DB (raised `nofile` ulimits in compose for large collections) |
| `reranker` | `cognitive` | `sidecars/rerank` | 8095 | Cross-encoder rerank |
| `sparse-index` | `cognitive` | `sidecars/sparse` | 8096 | BM25 hybrid retrieval |
| `turbovec-index` | `cognitive` | `sidecars/turbovec` | 8097 | TurboQuant dense ANN |

URLs:

| Env | Stock default (`docker/.env.example`) | Compose hard-wires? |
|-----|-------|------|
| `LLAMA_SWAP_URL` | `http://llama-swap:8080` | No — set in `docker/.env` (external chat API allowed) |
| `EMBED_URL` | `http://llama-swap:8080` | No — Qwen `/v1/embeddings` (bundled llama-swap or any host) |
| `EMBED_MODEL` | `embed-pool` | No — llama-swap id for Qwen3 Embedding 4B |
| `RERANKER_URL` | `http://reranker:8095` | Yes (cognitive profile) |
| `SPARSE_INDEX_URL` | `http://sparse-index:8096` | Yes (cognitive profile) |
| `TURBOVEC_URL` | `http://turbovec-index:8097` | Yes (cognitive profile) |

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

Optional **intent** model: add a small chat model to `docker/config.yaml` and set `ENABLE_INTENT_ROUTER=true` + `INTENT_MODEL=<model-id>` in `.env`. Intent runs via llama-swap, not a separate container.

## External Qdrant

Use bundled Qdrant only when you want an all-in-one stack. For an existing Qdrant host:

```bash
# docker/.env
QDRANT_URL=http://192.168.1.10:6333
```

```bash
docker compose --profile cognitive up -d --build
```

(`--profile qdrant` not needed.)

## Docker-in-Docker models

If `docker/config.yaml` uses `docker run` in `cmd`, mount the host socket into llama-swap ([llama-swap wiki](https://github.com/mostlygeek/llama-swap/wiki/Docker-in-Docker-with-llama%E2%80%90swap-guide)):

```yaml
# llama-swap.volumes:
#   - /var/run/docker.sock:/var/run/docker.sock
#   - /usr/bin/docker:/usr/bin/docker
```

## Rebuild after code changes

```bash
docker compose build rag-proxy reranker sparse-index
docker compose --profile qdrant --profile cognitive up -d
```

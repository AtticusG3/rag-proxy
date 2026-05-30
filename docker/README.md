# Docker stack (rag-proxy + llama-swap:cuda)

Clients use **rag-proxy** on port **8088** (`/v1`). Optional **cognitive** profile adds CPU rerank + BM25 sparse sidecars.

## Prerequisites

- Docker Compose v2
- NVIDIA Container Toolkit (for `llama-swap:cuda`)
- Host directory with GGUF models (chat model + nomic-embed)
- For full cognitive: Qdrant collection populated with chunk payloads (`text`, `content`, etc.)

## Deploy modes

### Legacy RAG (minimal)

```bash
cp docker/.env.example docker/.env
cp docker/config.yaml.example docker/config.yaml
# Edit MODELS_DIR, QDRANT_URL, chat model in docker/config.yaml

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

First **reranker** start downloads `BAAI/bge-reranker-base` (~400MB). **sparse-index** scrolls Qdrant on startup (may take a minute on large collections).

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
| `llama-swap` | default | `llama-swap:cuda` | 8080 | GPU chat model router |
| `nomic-embed` | default | `llama-swap:cpu` | 8089 | Query embeddings (CPU) |
| `qdrant` | `qdrant` | `qdrant/qdrant` | 6333 | Vector DB |
| `reranker` | `cognitive` | `sidecars/rerank` | 8095 | Cross-encoder rerank |
| `sparse-index` | `cognitive` | `sidecars/sparse` | 8096 | BM25 hybrid retrieval |

Internal URLs wired in compose:

| Env | Value |
|-----|-------|
| `LLAMA_SWAP_URL` | `http://llama-swap:8080` |
| `EMBED_URL` | `http://nomic-embed:8089` |
| `RERANKER_URL` | `http://reranker:8095` |
| `SPARSE_INDEX_URL` | `http://sparse-index:8096` |

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

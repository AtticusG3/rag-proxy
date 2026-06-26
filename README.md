# rag_proxy

Transparent RAG middleware in front of [llama-swap](https://github.com/mostlygeek/llama-swap). Clients point at this proxy instead of llama-swap directly; chat requests can be augmented with Qdrant context before they reach any model.

**You do not change client API keys** — point the OpenAI-compatible base URL at the proxy; llama-swap still validates auth downstream.

## Quick start

1. **Start dependencies**: Qdrant (with your collection), `nomic-embed` on `:8089`, llama-swap on `:8080`.
2. **Install**: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
3. **Configure**: `cp .env.example .env` — set `QDRANT_URL` and `QDRANT_COLLECTION`.
4. **Run**: `python rag_proxy.py`
5. **Point clients** at `http://<host>:8088/v1` (same paths and API key as llama-swap).
6. **Verify** — see [Getting started — Verify the stack](docs/getting-started.md#verify-the-stack).

Leave `ENABLE_COGNITIVE_PIPELINE=false` until legacy RAG injects chunks reliably.

### Minimum `.env`

| Variable | Required | Typical value |
| --- | --- | --- |
| `QDRANT_URL` | Yes | `http://<qdrant-host>:6333` |
| `QDRANT_COLLECTION` | If not default | see `.env.example` |
| `LLAMA_SWAP_URL` | If not local | `http://127.0.0.1:8080` |
| `EMBED_URL` | If not local | `http://127.0.0.1:8089` |

Full variable reference: [docs/configuration.md](docs/configuration.md).

## Architecture (brief)

**Legacy (default)** — `ENABLE_COGNITIVE_PIPELINE=false`:

```text
Client -> rag_proxy :8088
            | embed (nomic-embed :8089)
            | search Qdrant
            | inject chunks into system message
            -> llama-swap :8080 -> models
```

**Cognitive (optional)** — tiered stages (tier0, intent, gating, retrieve, rerank, graph, memgraphrag, tools, memory) with per-stage `ENABLE_*` flags and budgets. Fail-open: RAG errors never break the upstream request.

Details: [docs/architecture.md](docs/architecture.md) · rollout: [docs/cognitive-pipeline.md](docs/cognitive-pipeline.md) · [docs/COGNITIVE_RAG_PLAN.md](docs/COGNITIVE_RAG_PLAN.md)

## Documentation

| Guide | Purpose |
| --- | --- |
| [docs/README.md](docs/README.md) | Documentation index |
| [Getting started](docs/getting-started.md) | Install, verify, legacy RAG behavior |
| [Configuration](docs/configuration.md) | All env vars |
| [Headers and clients](docs/headers-and-clients.md) | Open WebUI, Cursor, per-request headers |
| [Deployment](docs/deployment.md) | systemd, Docker |
| [Observability](docs/observability.md) | Traces, metrics, logs |
| [Troubleshooting](docs/troubleshooting.md) | Common issues |
| [Ingest and admin](docs/ingest-and-admin.md) | Content indexing UI and worker |
| [MemGraphRAG](docs/memgraphrag.md) | Graph index build and rollout |

## Deployment

**Linux systemd** (production): edit `rag-proxy.service` and `nomic-embed.service` paths, then `systemctl enable --now nomic-embed rag-proxy`. Walkthrough: [docs/deployment.md](docs/deployment.md).

**Docker**: `docker compose up -d --build` (legacy) or `--profile qdrant --profile cognitive` for full stack. See [docker/README.md](docker/README.md).

Default ports: proxy `8088`, llama-swap `8080`, embed `8089` — all configurable via `.env`.

## Tests

```bash
pip install pytest
pytest tests/ -q
```

Or on Windows: `.\scripts\run-tests.ps1`

Offline unit tests only (no live Qdrant or embed server).

## License

Private.

# rag_proxy documentation

Operator and integrator guides for transparent RAG middleware in front of [llama-swap](https://github.com/mostlygeek/llama-swap).

**New here?** Read the [repository README](../README.md) for a plain-English overview, then [Getting started](getting-started.md) and [Verify the stack](getting-started.md#verify-the-stack).

## Guides

| Guide | What it covers |
| --- | --- |
| [Getting started](getting-started.md) | Install, `.env`, first run, smoke tests, legacy RAG behavior |
| [Configuration](configuration.md) | All environment variables grouped by concern |
| [Architecture](architecture.md) | Legacy vs cognitive, components, fail-open, injection |
| [Cognitive pipeline](cognitive-pipeline.md) | Stage summary, rollout pointers |
| [COGNITIVE_RAG_PLAN.md](COGNITIVE_RAG_PLAN.md) | Detailed cognitive rollout, flag matrix, failure modes |
| [Headers and clients](headers-and-clients.md) | Base URL, per-request headers, client integration |
| [Observability](observability.md) | Traces, JSON logs, metrics, log greps |
| [Deployment](deployment.md) | systemd, Docker, port layout |
| [Troubleshooting](troubleshooting.md) | Symptom → fix table and diagnostic commands |
| [Ingest and admin](ingest-and-admin.md) | RAG admin UI, ingest worker |
| [MemGraphRAG](memgraphrag.md) | Offline index build, runtime enable, verify |
| [Scripts](../scripts/README.md) | Operator and dev script index |

## Quick reference

| Topic | Default / typical |
| --- | --- |
| Proxy port (prod) | `8088` (`PROXY_PORT`) |
| llama-swap | `http://127.0.0.1:8080` |
| nomic-embed | `http://127.0.0.1:8089` (localhost only) |
| Qdrant | set `QDRANT_URL` (e.g. `http://127.0.0.1:6333`) |
| Cognitive pipeline | off (`ENABLE_COGNITIVE_PIPELINE=false`) |
| Client base URL | `http://<host>:8088/v1` |
| Metrics | `GET /metrics` on proxy port when `ENABLE_METRICS=true` |

## Related

- [README.md](../README.md) — repository entry point
- [.env.example](../.env.example) — environment template
- [docker/README.md](../docker/README.md) — Docker compose stack
- [AGENTS.md](../AGENTS.md) — agent/developer map (not operator-facing)

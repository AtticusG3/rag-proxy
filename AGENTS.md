# rag_proxy — Agent Guide

Transparent RAG middleware: optional tiered cognitive pipeline, then forward to llama-swap. Default remains embed → Qdrant dense search → inject (fail-open).

## Engineering principles

All work follows `.cursor/rules/engineering-principles.mdc` (Rules 1–8).

## Repository map

| Path | Purpose |
|------|---------|
| `rag_proxy.py` | Shim entrypoint + backward-compat exports |
| `rag_proxy/app.py` | FastAPI proxy route |
| `rag_proxy/orchestrator.py` | Pipeline runner |
| `rag_proxy/legacy_rag.py` | Embed, Qdrant, extract, inject |
| `rag_proxy/config.py` | Settings / feature flags |
| `rag_proxy/stages/` | Tier 0–3 stages |
| `tests/` | Offline pytest |
| `.env.example` | Env template |
| `docs/COGNITIVE_RAG_PLAN.md` | Operator architecture reference |

## Skills (project)

| Skill | Use when |
|-------|----------|
| `rag-proxy-change` | RAG logic, paths, injection, env config |
| `rag-proxy-test` | Tests |
| `rag-proxy-debug` | Missing/wrong RAG context |
| `rag-proxy-deploy` | systemd, `.env`, homelab |

## Default success criteria

- `pytest tests/ -q` passes; no network in unit tests.
- Fail-open: cognitive errors never break upstream request.
- New env vars in `.env.example` + `rag_proxy/config.py`.

## Cognitive pipeline

- Master switch: `ENABLE_COGNITIVE_PIPELINE` (default **false** = legacy).
- Subsystems: `ENABLE_TIER0_HEURISTICS`, `ENABLE_RETRIEVAL_GATING`, `ENABLE_INTENT_ROUTER`, `ENABLE_HYBRID_RETRIEVAL`, `ENABLE_RERANKER`, `ENABLE_GRAPH_LOOKUP`, `ENABLE_TOOLS`, `ENABLE_ROLLING_MEMORY`, etc.
- Hybrid: dense Qdrant + optional `SPARSE_INDEX_URL` sidecar, RRF merge when `ENABLE_HYBRID_RETRIEVAL=true`.
- Reranker: HTTP sidecar at `RERANKER_URL`, not in-process.

## Learned workspace facts

- Git remote: `https://git.kevynwatkins.com/kevyn/rag-proxy.git`
- Production host `nomad`; ports: proxy `8088`, llama-swap `8080`, nomic-embed `8089`
- Qdrant: `http://192.168.1.36:6333`, collection `nomad_knowledge_base`
- Production tuning: `SIMILARITY_THRESHOLD=0.65`, `TOP_K=5`, `EMBED_MAX_CHARS=2000`

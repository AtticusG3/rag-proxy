# rag_proxy — Agent Guide

Transparent RAG middleware: optional tiered cognitive pipeline, then forward to llama-swap. Default remains embed → Qdrant dense search → inject (fail-open).

## Engineering principles

All work follows `.cursor/rules/engineering-principles.mdc` (Rules 1–8).

## Repository map

| Path | Purpose |
|------|---------|
| `rag_proxy.py` | Shim entrypoint + backward-compat exports |
| `rag_proxy/app.py` | FastAPI proxy route |
| `rag_proxy/orchestrator.py` | Pipeline runner (budget-aware stage loop) |
| `rag_proxy/pipeline_stages.py` | Declarative stage registry (`build_pipeline_stages()`) |
| `rag_proxy/retrieval_policy.py` | Tier0 bypass + gating policy |
| `rag_proxy/context.py` | `RequestContext`, pipeline enums |
| `rag_proxy/observability.py` | Trace IDs, pipeline summaries, `GET /metrics` |
| `rag_proxy/legacy_rag.py` | Embed, Qdrant, extract, inject |
| `rag_proxy/config.py` | Settings / feature flags |
| `rag_proxy/stages/` | Tier 0–3 stage implementations |
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

## User-facing docs

Operator setup, client configuration, smoke tests, and troubleshooting: **README.md** (*First-time setup*, *Verify the stack*, *Troubleshooting*).

Cognitive rollout, headers, and trace log reading: **docs/COGNITIVE_RAG_PLAN.md**.

## Cognitive pipeline

- Master switch: `ENABLE_COGNITIVE_PIPELINE` (default **false** = legacy).
- Stage order (from `pipeline_stages.py`): tier0 → intent → gating → routing → rewrite → retrieve → rerank → graph → tools → memory → context.
- Per-stage skip: orchestrator skips a stage when remaining budget `< min_budget_ms` (from `STAGE_BUDGET_*` and related timeouts).
- Subsystems: `ENABLE_TIER0_HEURISTICS`, `ENABLE_RETRIEVAL_GATING`, `ENABLE_INTENT_ROUTER`, `ENABLE_HYBRID_RETRIEVAL`, `ENABLE_RERANKER`, `ENABLE_GRAPH_LOOKUP`, `ENABLE_TOOLS`, `ENABLE_ROLLING_MEMORY`, etc. Full matrix: `docs/COGNITIVE_RAG_PLAN.md`.
- Hybrid: dense Qdrant + optional `SPARSE_INDEX_URL` sidecar, RRF merge when `ENABLE_HYBRID_RETRIEVAL=true`.
- Reranker: HTTP sidecar at `RERANKER_URL`, not in-process.
- Observability: `ENABLE_REQUEST_TRACE`, `ENABLE_JSON_LOGS`, `ENABLE_METRICS` (`GET /metrics` on proxy port, not a separate listener).

## Learned User Preferences

- Windows PowerShell: chain shell commands with `;`, not `&&` (bash-style chaining fails).
- Prefer commits in vertical slices and PRs in small logical batches for review.

## Learned Workspace Facts

- Git remotes: `origin` (Gitea primary) `https://git.kevynwatkins.com/kevyn/rag-proxy.git`; `github` (secondary) `https://github.com/AtticusG3/rag-proxy.git`
- `gh` CLI works for GitHub (`AtticusG3/rag-proxy`); Gitea `origin` pull requests use the Gitea web UI, not `gh`
- `nomad` is the inference/RAG deploy and test host; `clanker` is studio/frontend only—not part of the rag_proxy stack
- Inference host `nomad`: prod rag-proxy `8088` (systemd), dev checkout `8087`, llama-swap `8080`; `EMBED_URL` is llama.cpp `llama-server` with nomic-embed only on `127.0.0.1:8089` (not proxied on 8088/8087)
- Production install path on nomad: `/home/kevyn/rag_proxy` (`rag-proxy.service` uses `.venv/bin/python rag_proxy.py`; missing venv causes systemd 203/EXEC)
- On nomad shell sessions, `.env` is not auto-loaded; source explicitly (`set -a; . ./.env; set +a`) before smoke scripts
- Qdrant: `http://192.168.1.36:6333`, collection `nomad_knowledge_base`
- Qdrant collection `nomad_knowledge_base` is owned by project-nomad; rag_proxy cannot change upstream collection schema or indexing
- Production tuning: `SIMILARITY_THRESHOLD=0.65`, `TOP_K=5`, `EMBED_MAX_CHARS=2000`
- No admin UI or JSON settings API; config via `.env`/systemd restart; per-request `x-rag-mode` / `x-no-cache` / `x-conversation-id` headers; `GET /metrics` is Prometheus counters only
- Run offline tests from repo root with `.\scripts\run-tests.ps1` (uses `.venv\Scripts\python.exe` when present)

# Cognitive RAG Proxy — Operator Reference

Architecture proposal implemented in the `rag_proxy/` package. See README for quick start.

## Feature flag matrix

| Flag | Default | Purpose |
|------|---------|---------|
| `ENABLE_COGNITIVE_PIPELINE` | false | Master switch; false = legacy always-retrieve |
| `ENABLE_TIER0_HEURISTICS` | false | Regex fast path; skip embed/Qdrant |
| `ENABLE_RETRIEVAL_GATING` | false | Intent/heuristic skip retrieval |
| `GATING_LOG_ONLY` | false | Log gating without skipping (bake-in) |
| `ENABLE_INTENT_ROUTER` | false | Rules + optional `INTENT_MODEL` |
| `ENABLE_QUERY_REWRITE` | false | Deterministic query normalization |
| `ENABLE_HYBRID_RETRIEVAL` | false | Dense + sparse RRF |
| `ENABLE_RERANKER` | false | HTTP rerank sidecar |
| `ENABLE_MODEL_ROUTING` | false | `suggest` or `force` model override |
| `ENABLE_GRAPH_LOOKUP` | false | SQLite infra graph |
| `ENABLE_TOOLS` | false | Whitelisted file reads |
| `ENABLE_ROLLING_MEMORY` | false | Session summaries in SQLite |

## Latency budgets

- `COGNITIVE_LATENCY_BUDGET_MS` (800): skip lower-priority stages when exhausted.
- Priority: context inject > rerank > graph > tools > LLM rewrite.

## External services

| Service | Env | Notes |
|---------|-----|-------|
| Sparse BM25 | `SPARSE_INDEX_URL` | POST `/search` JSON; optional |
| Reranker | `RERANKER_URL` | POST `/rerank` with `pairs`, `top_k` |
| Graph | `GRAPH_DB_PATH` | SQLite `entities` / `edges` tables |
| Memory | `MEMORY_DB_PATH` | SQLite `session_memory` |

## Model recommendations (homelab)

| Role | Model |
|------|-------|
| Intent | Qwen2.5-0.5B / Phi-3.5-mini Q4 |
| Rerank | bge-reranker-base (CPU sidecar) |
| Main chat/reasoning | Existing llama-swap stack |

## Failure modes

| Symptom | Check |
|---------|-------|
| Always slow | Tier 0/gating off; retrieval always runs |
| Never retrieves | Gating too aggressive; lower thresholds |
| Rerank stalls | `RERANK_TIMEOUT_MS`; disable reranker |
| Tools leak paths | Tighten `TOOL_ALLOWED_ROOTS` |

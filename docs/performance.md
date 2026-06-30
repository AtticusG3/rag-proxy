# Performance tuning

Operator guide for latency, memory, and observability trade-offs. Complements [Configuration](configuration.md) and [Observability](observability.md).

## Measure first

1. Set `ENABLE_METRICS=true` and scrape `GET /metrics` (protect with `PROXY_INTERNAL_TOKEN` when set).
2. Enable `ENABLE_REQUEST_TRACE=true` (default) and grep `latency_ms` per stage.
3. Use `GET /debug` (same auth as metrics) for a live snapshot of pools, cache, and feature flags.

Histograms exported when metrics are on:

| Metric | Use |
| --- | --- |
| `rag_stage_latency_seconds{stage}` | Which cognitive stage dominates |
| `rag_augment_duration_seconds` | Total RAG path before upstream |
| `proxy_request_duration_seconds` | End-to-end handler (streaming completes when the stream ends) |
| `rag_embed_cache_hits_total` / `misses_total` | Embed cache effectiveness |

## Legacy vs cognitive

| Mode | Typical cost | When to use |
| --- | --- | --- |
| Legacy (`ENABLE_COGNITIVE_PIPELINE=false`) | Embed + Qdrant + inject | Default homelab; lowest overhead |
| Full cognitive | +intent, gating, rerank, graph, memgraphrag, tools | Higher quality routing; tune `COGNITIVE_LATENCY_BUDGET_MS` |

`COGNITIVE_LATENCY_BUDGET_MS` (default `800`) skips stages that need more remaining budget than their `STAGE_BUDGET_*` minimum. Expensive stages drop first when time runs out.

## Feature flag latency impact

Approximate relative cost (depends on hardware and corpus size):

| Flag | Extra work | Notes |
| --- | --- | --- |
| `ENABLE_HYBRID_RETRIEVAL` | +sparse HTTP call (parallel with dense) | Better recall; needs `SPARSE_INDEX_URL` sidecar |
| `ENABLE_RERANKER` | +rerank HTTP call | CPU sidecar; set `RERANK_TIMEOUT_MS` |
| `ENABLE_QUERY_REWRITE` / `ENABLE_QUERY_REWRITE_LLM` | Regex and/or LLM rewrite | LLM path adds tens–hundreds of ms |
| `ENABLE_INTENT_ROUTER` | Rules + optional tiny LLM | `INTENT_TIMEOUT_MS` caps model call |
| `ENABLE_GRAPH_LOOKUP` | SQLite graph walk (off event loop) | Infra intents only |
| `ENABLE_MEMGRAPHRAG` | Fact scoring + PPR + embed | Heaviest retrieval stage; `STAGE_BUDGET_MEMGRAPHRAG_MS` |
| `ENABLE_TOOLS` | Filesystem reads | `TOOL_BUDGET_MS` |
| `ENABLE_SEMANTIC_DEDUPE` | CPU substring checks | Cheap; reduces duplicate chunks |
| `ENABLE_EMBED_CACHE` | In-process SHA cache (600s TTL) | Good for repeated queries; respect `X-No-Cache` |
| `ENABLE_TOKENIZER_ESTIMATE` | tiktoken encode for budget | Better context fit vs char heuristic |

## Caching

**Embed cache** (`ENABLE_EMBED_CACHE=false` by default):

- Turn on when the same queries repeat (IDE assistants, smoke tests).
- Bypass per request with `X-No-Cache: true`.
- Check `embed_cache` in `GET /debug` and `rag_embed_cache_*` metrics.

HTTP connection pools for embed, Qdrant, and sparse start at app lifespan — no per-request client creation on the retrieval path.

## Context budget

| Setting | Default | Role |
| --- | --- | --- |
| `CONTEXT_BUDGET_RATIO` | `0.25` | Fraction of model context window for RAG chunks |
| `CONTEXT_FALLBACK_CHARS` | `8000` | Budget when model context length unknown |
| `DEFAULT_COMPLETION_RESERVE` | `1024` | Tokens reserved for assistant reply |
| `ENABLE_TOKENIZER_ESTIMATE` | `false` | Use tiktoken (`cl100k_base`) instead of chars/4 |

When tokenizer estimate is on, injection truncates by **tokens**, aligning better with upstream context limits. When off, char-based budget remains (legacy behavior).

Embed input uses **tail** truncation (`prepare_embed_text`) so the latest user text is preserved in long prompts.

## Stage timeouts

`STAGE_EXEC_TIMEOUT_MS` (default `30000`) caps stages without a dedicated `STAGE_BUDGET_*`. Stages with a budget use that value as the execution timeout (e.g. retrieve `50ms`, rerank `200ms`). Timeouts fail-open: the stage is skipped and the request continues.

## Linux production runtime

`uvicorn[standard]` pulls in `uvloop` and `httptools` on Linux. The default `rag_proxy.py` entrypoint uses uvicorn defaults — on Linux production hosts this typically selects uvloop automatically when installed.

Example explicit start (Linux):

```bash
uvicorn rag_proxy.app:app --host 0.0.0.0 --port 8088 --loop uvloop --http httptools
```

Windows dev does not use uvloop.

## Profiling (dev)

Offline investigation tools (not automated in CI):

```bash
py-spy record -o profile.svg -- python rag_proxy.py
python -m cProfile -o proxy.prof rag_proxy.py
```

## Benchmark regression checks

Phase 3 adds optional micro-benchmarks in `tests/test_benchmarks.py` using `pytest-benchmark`.

Install dev dependencies, then run only benchmark tests:

```bash
python -m pip install -r requirements-dev.txt
pytest tests/test_benchmarks.py --benchmark-only
```

If the plugin is missing, these tests auto-skip and do not affect regular unit test runs.

## Related docs

- [Observability](observability.md) — traces, metrics, greps
- [Configuration](configuration.md) — all env vars
- [Cognitive pipeline](cognitive-pipeline.md) — stage order and flags
- [Deployment](deployment.md) — systemd and Docker

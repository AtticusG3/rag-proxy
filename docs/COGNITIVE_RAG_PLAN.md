# Cognitive RAG Proxy — Operator Reference

Detailed rollout phases, flag matrix, and failure modes for the cognitive pipeline. Architecture implemented in the `rag_proxy/` package.

**Related docs:** [Documentation index](README.md) · [Getting started](getting-started.md) · [Cognitive pipeline (summary)](cognitive-pipeline.md) · [MemGraphRAG](memgraphrag.md) · [Configuration](configuration.md) · [Observability](observability.md)

## Enabling cognitive mode (step-by-step)

**Prerequisite:** Legacy RAG works (`ENABLE_COGNITIVE_PIPELINE=false`, logs show `RAG: injected N chunk(s)` for a known-good query). See [Getting started — Verify the stack](getting-started.md#verify-the-stack).

### Phase 0 — Baseline

```bash
# .env
ENABLE_COGNITIVE_PIPELINE=false
```

Confirm one chat request injects chunks. Note the query and scores for comparison later.

### Phase 1 — Pipeline on, tier0 observe-only

```bash
ENABLE_COGNITIVE_PIPELINE=true
ENABLE_TIER0_HEURISTICS=true
ENABLE_RETRIEVAL_GATING=false
GATING_LOG_ONLY=true
ENABLE_REQUEST_TRACE=true
```

Restart proxy. Send:

- A greeting (`hi`) — trace should show tier0 path; with gating still off, retrieval may still run.
- An infra question (`what port is qdrant on?`) — should retrieve.

Watch logs for `trace=… stages=tier0,intent,gating,…` and `gating_would_skip` in JSON logs if `ENABLE_JSON_LOGS=true`.

### Phase 2 — Gating live

```bash
ENABLE_RETRIEVAL_GATING=true
GATING_LOG_ONLY=false
```

- Greeting → expect `RAG: skipped retrieval` or trace `retrieval=skip`, no embed delay.
- Domain question → expect `RAG: injected N chunk(s)` as before.

If good queries stop retrieving, lower `INTENT_CONFIDENCE_THRESHOLD` or disable intent until rules are tuned.

### Phase 3 — Optional subsystems (one per week)

Enable individually; re-run the same test query after each change:

| Order | Flag | Requires | Verify |
|-------|------|----------|--------|
| 1 | `ENABLE_INTENT_ROUTER=true` | optional `INTENT_MODEL` | trace shows `intent=` label |
| 2 | `ENABLE_QUERY_REWRITE=true` | — | `rewrite` in `stage_trace` |
| 3 | `ENABLE_HYBRID_RETRIEVAL=true` | `SPARSE_INDEX_URL` sidecar | hybrid hits in trace |
| 4 | `ENABLE_RERANKER=true` | `RERANKER_URL` sidecar | rerank stage, reordered scores |
| 5 | Tier 3 flags | graph/tools/memory DB paths | see External services |

After each flag: `sudo systemctl restart rag-proxy` (systemd) or restart `python rag_proxy.py`.

## Per-request control (headers)

Send these on chat `POST` requests only. Case-insensitive header names.

| Header | Example | Effect |
|--------|---------|--------|
| `X-RAG-Mode` | `off` | Skip all RAG for this request |
| `X-RAG-Mode` | `force` | Always retrieve; bypass tier0/gating skip |
| `X-RAG-Mode` | `auto` | Default pipeline behavior |
| `X-No-Cache` | `true` | Bypass embed cache when `ENABLE_EMBED_CACHE=true` |
| `X-Conversation-Id` | `session-abc` | Key for rolling memory when enabled |

Example — force retrieval for one call:

```bash
curl -s -X POST "http://127.0.0.1:8088/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_KEY" \
  -H "X-RAG-Mode: force" \
  -d '{"model":"your-model","messages":[{"role":"user","content":"hello"}],"stream":false}'
```

Example — skip RAG for a meta prompt:

```bash
curl -s -X POST "http://127.0.0.1:8088/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "X-RAG-Mode: off" \
  -d '{"model":"your-model","messages":[{"role":"user","content":"### Task: suggest follow-ups"}],"stream":false}'
```

## Reading logs and traces

With `ENABLE_REQUEST_TRACE=true` (default), each cognitive request emits one summary line at INFO:

```text
trace=a1b2c3d4e5f6 tier=tier2_retrieval intent=infra retrieval=full chunks=3 latency_ms={'tier0': 1.2, 'retrieve': 89.4, ...} stages=tier0,intent,gating,retrieve,context
```

| Field | Meaning |
|-------|---------|
| `trace` | Correlation id — grep one request end-to-end |
| `tier` | Highest tier reached |
| `intent` | Classified intent label |
| `retrieval` | `skip`, `light`, or `full` |
| `chunks` | Count injected into system message |
| `stages` | Stages that actually ran (disabled/budget-skipped omitted) |
| `latency_ms` | Per-stage milliseconds |

Set `ENABLE_JSON_LOGS=true` for machine-readable JSON (includes `gating_would_skip`, `scores`, `errors`).

Legacy mode (`ENABLE_COGNITIVE_PIPELINE=false`) uses simpler lines:

```text
RAG: injected 3 chunk(s) (scores: [0.82, 0.71, 0.68]) | query: 'how do I restart rag-proxy'
```

**Useful log greps**

```bash
journalctl -u rag-proxy -f | grep -E 'RAG:|trace='
journalctl -u rag-proxy --since "1 hour ago" | grep 'gating_would_skip'
```

## Pipeline stages

Registered in `pipeline_stages.py`; orchestrator runs in order, skipping disabled stages or those below budget:

`tier0` → `intent` → `gating` → `routing` → `rewrite` → `retrieve` → `rerank` → `graph` → `memgraphrag` → `tools` → `memory` → `context`

Tier0 bypass and gating skip/light/full decisions live in `retrieval_policy.py`.

The `tier0` stage is always registered when the cognitive pipeline is on (`enabled=True` in `pipeline_stages.py`). `ENABLE_TIER0_HEURISTICS` gates heuristic logic inside the stage; when false, tier0 no-ops except `X-RAG-Mode: off|force` header overrides.

## Feature flag matrix

| Flag | Default | Purpose |
|------|---------|---------|
| `ENABLE_COGNITIVE_PIPELINE` | false | Master switch; false = legacy always-retrieve |
| `ENABLE_TIER0_HEURISTICS` | false | Regex fast path; skip embed/Qdrant |
| `ENABLE_RETRIEVAL_GATING` | false | Intent/heuristic skip retrieval |
| `GATING_LOG_ONLY` | false | Log gating without skipping (bake-in) |
| `ENABLE_INTENT_ROUTER` | false | Rules + optional `INTENT_MODEL` |
| `ENABLE_QUERY_REWRITE` | false | Deterministic query normalization |
| `ENABLE_QUERY_REWRITE_LLM` | false | LLM query rewrite (requires rewrite enabled) |
| `ENABLE_HYBRID_RETRIEVAL` | false | Dense + sparse RRF |
| `ENABLE_RERANKER` | false | HTTP rerank sidecar |
| `ENABLE_SEMANTIC_DEDUPE` | false | Embedding-similarity dedupe before inject |
| `ENABLE_EMBED_CACHE` | false | In-request embed cache (respects `X-No-Cache`) |
| `ENABLE_TOKENIZER_ESTIMATE` | false | Token-budget estimation in context assembly |
| `ENABLE_MODEL_ROUTING` | false | `suggest` or `force` model override |
| `ENABLE_GRAPH_LOOKUP` | false | SQLite infra graph |
| `ENABLE_MEMGRAPHRAG` | false | Three-layer memory + PPR passage retrieval |
| `ENABLE_TOOLS` | false | Whitelisted file reads |
| `ENABLE_ROLLING_MEMORY` | false | Session summaries in SQLite |
| `ENABLE_REQUEST_TRACE` | true | Per-request pipeline summary logs |
| `ENABLE_JSON_LOGS` | false | JSON pipeline logs (vs text) |
| `ENABLE_METRICS` | false | `GET /metrics` on proxy port |

Legacy: `METRICS_PORT` > 0 also enables metrics when `ENABLE_METRICS` is unset/false. Not a separate listener.

## Transcript capture

Optional JSONL capture for fine-tuning exports and RAG corpus promotion. Separate from pipeline observability; disabled by default. Not part of the cognitive stage list — runs in the proxy route after the pipeline.

| Variable | Default | Purpose |
| --- | --- | --- |
| `ENABLE_TRANSCRIPT_CAPTURE` | false | Master switch |
| `FINETUNE_LOG_PATH` | `/var/lib/rag_proxy/capture/finetune.jsonl` | Sanitized request + assistant completion |
| `RAG_IMPROVEMENT_LOG_PATH` | `/var/lib/rag_proxy/capture/rag_improvement.jsonl` | Query, retrieval metadata, hits, Q&A |
| `TRANSCRIPT_STRIP_PROXY_ARTEFACTS` | true | Strip RAG/memory prefixes from captured messages |
| `TRANSCRIPT_HEADER_OPT_IN` | false | Require `X-Capture-Log: true` per request |
| `TRANSCRIPT_SAMPLE_RATE` | 1.0 | Sample rate 0.0–1.0 |
| `TRANSCRIPT_HIT_PREVIEW_CHARS` | 300 | Hit preview length in RAG improvement records |
| `ENABLE_RAG_CORPUS_AUTO_INGEST` | false | Promote eligible Q&A pairs to Qdrant after append |
| `RAG_CORPUS_COLLECTION` | `nomad_conversation_derived` | Derived collection (separate from main KB) |
| `RAG_CORPUS_MIN_ANSWER_CHARS` | 100 | Minimum answer length for promotion |
| `RAG_CORPUS_REQUIRE_CHUNKS` | false | Require injected chunks before promotion |

Offline helpers:

```bash
python scripts/export_finetune_dataset.py --input /var/lib/rag_proxy/capture/finetune.jsonl --output finetune_messages.jsonl
python scripts/promote_rag_corpus.py --input /var/lib/rag_proxy/capture/rag_improvement.jsonl --dry-run
```

Full reference: [Configuration — Transcript capture](configuration.md#transcript-capture).

## Latency budgets

- `COGNITIVE_LATENCY_BUDGET_MS` (800): global budget; orchestrator skips stages when remaining ms `< min_budget_ms`.
- Per-stage minimums (skip if budget too low):
  - `STAGE_BUDGET_ROUTING_MS` (0) — model routing
  - `STAGE_BUDGET_REWRITE_MS` (20) — query rewrite
  - `STAGE_BUDGET_RETRIEVE_MS` (50) — embed + Qdrant/hybrid
  - `STAGE_BUDGET_GRAPH_MS` (100) — graph lookup
  - `STAGE_BUDGET_MEMGRAPHRAG_MS` (200) — MemGraphRAG stage
  - Rerank/tools use `RERANK_TIMEOUT_MS` / `TOOL_BUDGET_MS` as their min budgets.
- Priority when budget exhausted: context inject > rerank > graph > tools > LLM rewrite.
- `TIER0_MAX_CHARS` (80): max query length for tier0 heuristic bypass.

## External services

| Service | Env | Notes |
|---------|-----|-------|
| Sparse BM25 | `SPARSE_INDEX_URL` | POST `/search` JSON; optional |
| Reranker | `RERANKER_URL` | POST `/rerank` with `pairs`, `top_k` |
| Graph | `GRAPH_DB_PATH` | SQLite `entities` / `edges` tables |
| MemGraphRAG | `MEMGRAPHRAG_DB_PATH` | SQLite three-layer index — [MemGraphRAG operator guide](memgraphrag.md) |
| Memory | `MEMORY_DB_PATH` | SQLite `session_memory` |

## Model recommendations (examples)

| Role | Model |
|------|-------|
| Intent | Qwen2.5-0.5B / Phi-3.5-mini Q4 |
| Rerank | bge-reranker-base (CPU sidecar) |
| Main chat/reasoning | Existing llama-swap stack |

## Failure modes

| Symptom | What to do |
|---------|------------|
| Always slow | Tier 0/gating off — every request embeds + searches Qdrant; enable `ENABLE_TIER0_HEURISTICS` and `ENABLE_RETRIEVAL_GATING` |
| Never retrieves | Gating too aggressive — set `GATING_LOG_ONLY=true` and inspect traces; lower `INTENT_CONFIDENCE_THRESHOLD`; try `X-RAG-Mode: force` |
| Rerank stalls | Increase `RERANK_TIMEOUT_MS` or set `ENABLE_RERANKER=false` |
| Tools leak paths | Narrow `TOOL_ALLOWED_ROOTS` to comma-separated absolute paths |
| Stage skipped unexpectedly | Compare `latency_ms` total to `COGNITIVE_LATENCY_BUDGET_MS`; raise `STAGE_BUDGET_*` or global budget |
| No trace logs | Set `ENABLE_REQUEST_TRACE=true`; ensure `LOG_LEVEL=INFO` |
| Metrics 404 | Set `ENABLE_METRICS=true` (or legacy `METRICS_PORT>0`); hit `http://<proxy>:8088/metrics` |
| Chat works, RAG silent | Fail-open — check WARNING lines for embed/Qdrant errors; run [smoke tests](getting-started.md#verify-the-stack) |

Cognitive and RAG errors are fail-open: the original request body is forwarded unchanged.

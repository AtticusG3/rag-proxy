# Troubleshooting

Common symptoms when operating rag_proxy. Start with [Getting started — Verify the stack](getting-started.md#verify-the-stack) if you have not already.

## Chat works but no knowledge-base context

| Check | Action |
| --- | --- |
| Threshold too high | `LOG_LEVEL=DEBUG`; look for `no chunks above threshold` — lower `SIMILARITY_THRESHOLD` (e.g. `0.55`) |
| Empty collection | `curl "$QDRANT_URL/collections/$QDRANT_COLLECTION"` — confirm points exist for the topic |
| Wrong collection | Verify `QDRANT_COLLECTION` matches indexed data |
| Gating skipped retrieval | Cognitive mode: `ENABLE_REQUEST_TRACE=true`; look for `retrieval=skip` — try `X-RAG-Mode: force` |
| Query mismatch | Rephrase; embed uses last **user** message only |

## Never injects / RAG silent

| Check | Action |
| --- | --- |
| Placeholder URL | Startup warning `QDRANT_URL still has placeholder` — fix `.env` |
| Embed down | `curl -X POST "$EMBED_URL/v1/embeddings" ...` — see smoke test in getting started |
| Qdrant unreachable | `curl "$QDRANT_URL/collections/..."` from proxy host |
| Fail-open masking errors | Chat still returns 200 — read WARNING lines in logs |

## Only some messages get RAG

Open WebUI "follow-up" / `### Task:` prompts may be skipped by design. Send a normal user question or use `X-RAG-Mode: force` for testing.

## Streaming broken or truncated

Usually upstream llama-swap or model issue. Test the same `stream: true` request against `LLAMA_SWAP_URL` (`:8080`) directly. Proxy relays SSE as-is.

Abandoned upstream streams are closed after `UPSTREAM_STREAM_ABANDON_SEC` with no relayed bytes — increase if clients pause consumption for long periods.

## Cognitive mode feels random

| Check | Action |
| --- | --- |
| Traces off | Set `ENABLE_REQUEST_TRACE=true`, `LOG_LEVEL=INFO` |
| Gating too aggressive | `GATING_LOG_ONLY=true`; inspect `gating_would_skip` in JSON logs |
| Intent threshold | Lower `INTENT_CONFIDENCE_THRESHOLD` or disable `ENABLE_INTENT_ROUTER` temporarily |
| Budget skips stages | Compare total `latency_ms` to `COGNITIVE_LATENCY_BUDGET_MS`; raise `STAGE_BUDGET_*` |

See [COGNITIVE_RAG_PLAN.md — Failure modes](COGNITIVE_RAG_PLAN.md#failure-modes).

## Request succeeds when RAG fails

**Expected** — fail-open by design. RAG errors must not break upstream chat. Fix the underlying embed/Qdrant issue; do not expect HTTP 5xx from RAG alone.

## systemd / startup

| Symptom | Fix |
| --- | --- |
| `status=203/EXEC` | Missing `.venv` at `ExecStart` path |
| `Address already in use` | Change `PROXY_PORT` or stop conflicting service |
| Env not applied | Confirm `EnvironmentFile=` path; restart after edits |

## Metrics 404

Set `ENABLE_METRICS=true` (or legacy `METRICS_PORT>0`). Hit `http://<proxy_host>:<PROXY_PORT>/metrics` — not a separate port.

## No trace logs

`ENABLE_REQUEST_TRACE=true` and `LOG_LEVEL=INFO`. Cognitive mode required for `trace=` lines; legacy uses `RAG: injected ...` only.

## Docker / sidecars

| Symptom | Action |
| --- | --- |
| Hybrid no sparse hits | `SPARSE_INDEX_URL` set; sparse sidecar healthy (`8096/health`); reindex after Qdrant changes |
| Rerank stalls | Increase `RERANK_TIMEOUT_MS` or `ENABLE_RERANKER=false` |
| First rerank start slow | Downloads `BAAI/bge-reranker-base` (~400MB) |

[docker/README.md](../docker/README.md)

## Tools path leaks

Narrow `TOOL_ALLOWED_ROOTS` to comma-separated absolute paths. Never set to `/` in production.

## Quick diagnostic commands

```bash
# Passthrough
curl -s "http://127.0.0.1:8088/v1/models"

# Force RAG on a short query
curl -s -X POST "http://127.0.0.1:8088/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "X-RAG-Mode: force" \
  -d '{"model":"your-model","messages":[{"role":"user","content":"test"}],"stream":false}'

# Follow logs
journalctl -u rag-proxy -f | grep -E 'RAG:|trace=|WARNING'
```

## Per-request overrides

| Goal | Header |
| --- | --- |
| Always retrieve | `X-RAG-Mode: force` |
| Skip RAG once | `X-RAG-Mode: off` |
| Bypass embed cache | `X-No-Cache: true` |

[Headers and clients](headers-and-clients.md)

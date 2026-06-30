# Observability

Logs, request traces, and optional Prometheus-style counters on the main proxy port.

## Request traces (cognitive mode)

`ENABLE_REQUEST_TRACE=true` (default) emits one summary line per cognitive request at INFO.

Text format example:

```text
trace=a1b2c3d4e5f6 tier=tier2_retrieval intent=infra retrieval=full chunks=3 latency_ms={'tier0': 1.2, 'retrieve': 89.4, ...} stages=tier0,intent,gating,retrieve,context
```

| Field | Meaning |
| --- | --- |
| `trace` | Correlation id — grep one request end-to-end |
| `tier` | Highest pipeline tier reached |
| `intent` | Classified intent label |
| `retrieval` | `skip`, `light`, or `full` |
| `chunks` | Count injected into system message |
| `stages` | Stages that actually ran (disabled/budget-skipped omitted) |
| `latency_ms` | Per-stage milliseconds |

### JSON logs

Set `ENABLE_JSON_LOGS=true` for machine-readable JSON including `gating_would_skip`, `scores`, `errors`, `stage_trace`, and `cache_hits`.

Useful during gating bake-in (`GATING_LOG_ONLY=true`).

## Legacy mode logs

With `ENABLE_COGNITIVE_PIPELINE=false`:

```text
RAG: injected 3 chunk(s) (scores: [0.82, 0.71, 0.68]) | query: 'how do I restart rag-proxy'
```

Other legacy lines:

| Line | Meaning |
| --- | --- |
| `RAG: no chunks above threshold=...` | Search ran; nothing above `SIMILARITY_THRESHOLD` |
| `RAG: skipped retrieval` | Gating or tier0 skipped embed/Qdrant (cognitive) |

Implementation: `observability.log_rag_request()` and `log_pipeline_summary()`.

## Useful greps (systemd)

```bash
journalctl -u rag-proxy -f | grep -E 'RAG:|trace='
journalctl -u rag-proxy --since "1 hour ago" | grep 'gating_would_skip'
journalctl -u rag-proxy --since today | grep -E 'RAG:|trace='
```

Ensure `LOG_LEVEL=INFO` (or `DEBUG` for verbose RAG detail).

## Metrics

`ENABLE_METRICS=true` exposes `GET /metrics` on the **proxy port** (not a separate listener).

```bash
curl -s "http://127.0.0.1:8088/metrics"
```

Prometheus text format via `prometheus_client`. Example series:

| Metric | Type | Meaning |
| --- | --- | --- |
| `rag_requests_total{outcome}` | Counter | Pipeline completions: `hit`, `miss`, `skip` |
| `rag_chunks_injected_total` | Counter | Sum of chunks injected across requests |
| `rag_augment_errors_total` | Counter | Augmentation failures (request forwarded unmodified) |
| `rag_embed_cache_hits_total` / `rag_embed_cache_misses_total` | Counter | Embed cache when `ENABLE_EMBED_CACHE=true` |
| `rag_stage_latency_seconds{stage}` | Histogram | Per-stage pipeline latency |
| `rag_augment_duration_seconds` | Histogram | RAG augmentation wall time only |
| `proxy_request_duration_seconds` | Histogram | Full proxy handler (buffered responses include upstream read; streaming measures until stream ends) |
| `upstream_active_streams` | Gauge | Active upstream SSE streams |

Per-request `latency_ms` and `cache_hits` remain in trace logs (`ENABLE_REQUEST_TRACE`).

Set `ENABLE_METRICS=true` to expose metrics on the proxy port.

`GET /debug` returns a JSON snapshot of pools, embed cache, and feature flags (same `PROXY_INTERNAL_TOKEN` gate as metrics when configured). See [Performance](performance.md).

When metrics are disabled, `GET /metrics` returns `404` with body `metrics disabled`.

## Log level

| Variable | Default | Notes |
| --- | --- | --- |
| `LOG_LEVEL` | `INFO` | Set `DEBUG` for embed/Qdrant detail |

## Fail-open and warnings

RAG failures log at WARNING and do not change HTTP status. If chat works but traces show `chunks=0`, check WARNING lines for embed/Qdrant errors and run [Getting started — Verify the stack](getting-started.md#verify-the-stack).

## Related docs

- Cognitive trace field reference: [COGNITIVE_RAG_PLAN.md — Reading logs](COGNITIVE_RAG_PLAN.md#reading-logs-and-traces)
- Config flags: [Configuration — Observability](configuration.md#observability)
- Symptom table: [Troubleshooting](troubleshooting.md)

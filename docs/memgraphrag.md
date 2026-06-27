# MemGraphRAG operator guide

MemGraphRAG adds **graph-guided passage retrieval** on top of dense Qdrant search. It uses a three-layer SQLite memory (schema → fact → passage), built offline from your corpus or an existing Qdrant collection, then queried at request time via fact scoring, optional reranking, and Personalized PageRank (PPR).

**Default: off.** Requires `ENABLE_COGNITIVE_PIPELINE=true` and a built index at `MEMGRAPHRAG_DB_PATH`.

Related: [Cognitive pipeline](cognitive-pipeline.md) · [Configuration](configuration.md) · [Architecture](architecture.md)

## When to use it

| Use MemGraphRAG when… | Skip it when… |
| --- | --- |
| You want multi-hop context via entity–relation graphs | Legacy dense RAG is enough |
| You have (or can sample) a representative chunk corpus | You cannot run an LLM for offline extraction |
| You can afford ~100–500 ms extra per request (budget + embed/rerank) | You need the lowest possible latency |

MemGraphRAG **supplements** Qdrant hits — the stage appends `source="memgraphrag"` chunks to `ctx.hits`. It does not replace dense retrieval. On failure or empty results, prior hits are preserved (fail-open).

## Architecture

```text
Offline (build_memgraphrag_index.py)
  chunks  -->  LLM entity/relation extraction  -->  ontology filter
           -->  ThreeLayerMemory (SQLite)  -->  fact embeddings (nomic-embed)

Online (tier3_memgraphrag stage, after graph, before tools)
  query  -->  embed query
        -->  score facts (cosine vs stored fact embeddings)
        -->  rerank top facts (optional cross-encoder)
        -->  PPR on fact graph
        -->  aggregate passage scores  -->  append ChunkHits
```

### Three layers

| Layer | Contents | Role |
| --- | --- | --- |
| Schema | `(head_type, relation, tail_type)` patterns | Thematic denoising; groups facts |
| Fact | `(head, relation, tail)` triples | Graph nodes for PPR |
| Passage | Original chunk text + `chunk_id` | What gets injected into context |

Facts link to schemas and passages. PPR walks fact–fact edges (same schema, shared passages).

## Prerequisites

| Service | Build time | Runtime |
| --- | --- | --- |
| OpenAI-compatible LLM | Entity + relation extraction | — |
| nomic-embed (`EMBED_URL`) | Fact embedding (`--embed-url`) | Query + fact scoring |
| Reranker sidecar (`RERANKER_URL`) | — | Optional; improves fact ordering |
| Cognitive pipeline | — | `ENABLE_COGNITIVE_PIPELINE=true` |

Typical service URLs (adjust to your layout):

- LLM: `MEMGRAPH_BUILD_LLM_URL` (default `http://127.0.0.1:8080/v1`) — use a remote endpoint when local GPU is busy with ingest, e.g. `http://192.168.1.202:8081/v1`
- Model: `MEMGRAPH_BUILD_LLM_MODEL` (default `qwen3.5-9b-turbo`)
- Embed: `MEMGRAPH_BUILD_EMBED_URL` or `EMBED_URL` (e.g. `http://127.0.0.1:18089`)
- Reranker: `http://127.0.0.1:8095` (cognitive Docker profile or `sidecars/rerank`)

Set build LLM vars in `rag-proxy.env` or **rag-admin → Settings → MemGraphRAG index build**. The admin UI can start the build job and monitor logs.

Ensure the output directory exists and is writable (e.g. `/var/lib/rag_proxy/`).

## Build the index

Run from repo root with venv active. The script is **network-heavy** (LLM + embed calls). Env vars `MEMGRAPH_BUILD_*` are read by the CLI and rag-admin job runner (see [Configuration](configuration.md)).

### From Qdrant (recommended)

Samples chunks stratified by a payload field (default `source`) so the graph reflects your collection mix. If your Qdrant build lacks the facet API (HTTP 404), the script **falls back to scroll sampling** automatically.

```bash
python -m scripts.build_memgraphrag_index \
  --source qdrant \
  --qdrant-url http://127.0.0.1:6333 \
  --collection your_collection \
  --output /var/lib/rag_proxy/memgraphrag.sqlite \
  --llm-url http://192.168.1.202:8081/v1 \
  --llm-model qwen3.5-9b-turbo \
  --max-chunks 1000 \
  --stratify-field source \
  --embed-url http://127.0.0.1:18089
```

| Flag | Default | Notes |
| --- | --- | --- |
| `--max-chunks` | `1000` when source=qdrant | Stratified sample size; `0` = all (file mode only) |
| `--stratify-field` | `source` | Qdrant payload field for proportional sampling |
| `--payload-text-field` | `text` | Payload key holding chunk text |
| `--min-schema-freq` | `2` | Drop rare relation schemas (thematic denoising) |
| `--concurrency` | `3` | Parallel LLM requests |
| `--max-chars` | `2000` | Truncate chunk text sent to LLM |
| `--skip-relations` | off | Entities only — faster but **no facts for PPR** |
| `--skip-embed` | off | Skip fact embeddings — **runtime scoring skips those facts** |

### From a text file or directory

```bash
python -m scripts.build_memgraphrag_index \
  --input /path/to/corpus.txt \
  --output /var/lib/rag_proxy/memgraphrag.sqlite \
  --llm-url http://127.0.0.1:8080/v1 \
  --llm-model your-chat-model \
  --chunk-size 512 \
  --overlap 64 \
  --embed-url http://127.0.0.1:8089
```

For a directory, all `**/*.txt` files are loaded and chunked.

### Build output

Successful runs log stats like `Built memory: {...} -> /var/lib/rag_proxy/memgraphrag.sqlite`. Expect:

- Wall time dominated by LLM extraction (minutes to hours for hundreds of chunks)
- `JSON parse failures` warnings if the model returns non-JSON — retry with a more instruction-following model or lower `--concurrency`

Rebuild the index when your Qdrant corpus changes materially. There is no incremental update path today.

## Enable at runtime

Add to `.env` (requires cognitive pipeline):

```bash
ENABLE_COGNITIVE_PIPELINE=true
ENABLE_MEMGRAPHRAG=true
MEMGRAPHRAG_DB_PATH=/var/lib/rag_proxy/memgraphrag.sqlite
```

Optional tuning (defaults in [Configuration](configuration.md)):

| Variable | Default | Purpose |
| --- | --- | --- |
| `MEMGRAPHRAG_FACT_TOP_K` | `20` | Facts passed to reranker / PPR seeds |
| `MEMGRAPHRAG_PPR_DAMPING` | `0.85` | PageRank damping factor |
| `MEMGRAPHRAG_PPR_ITERATIONS` | `20` | PPR iterations |
| `MEMGRAPHRAG_PASSAGE_NODE_WEIGHT` | `0.5` | Weight when aggregating fact scores to passages |
| `STAGE_BUDGET_MEMGRAPHRAG_MS` | `200` | Skip stage if remaining budget below this |

For best results, also enable the reranker sidecar:

```bash
ENABLE_RERANKER=true
RERANKER_URL=http://127.0.0.1:8095
```

Without `RERANKER_URL`, fact reranking uses uniform scores (still works, weaker ordering).

Restart the proxy after changes: `sudo systemctl restart rag-proxy`.

## Verify

### 1. Offline retrieval smoke test

Bypasses the full proxy; exercises embed + optional rerank + PPR against the built SQLite file:

```bash
python scripts/test_memgraphrag_retrieve.py \
  --index /var/lib/rag_proxy/memgraphrag.sqlite \
  --embed-url http://127.0.0.1:8089/v1/embeddings \
  --rerank-url http://127.0.0.1:8095/rerank \
  --queries "your test question about indexed topics"
```

Expect passage previews with scores. `No facts in index` means rebuild without `--skip-relations`.

### 2. Proxy integration

With `ENABLE_REQUEST_TRACE=true`, send a chat request that should match your graph:

```bash
curl -s -X POST "http://127.0.0.1:8088/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_KEY" \
  -H "X-RAG-Mode: force" \
  -d '{"model":"your-model","messages":[{"role":"user","content":"question about your graph topics"}],"stream":false}'
```

Look for:

- `memgraphrag:N` in `stage_trace` / trace log line
- Injected chunks with graph-derived passages (may mix with Qdrant hits)

See [Observability](observability.md) for trace field reference.

## Pipeline behavior

- **Stage order**: after `graph`, before `tools` and `memory` (`pipeline_stages.py`).
- **Appends hits** — does not clear Qdrant results from earlier `retrieve` / `rerank` stages.
- **Empty memory** — logs `MemGraphRAG memory is empty, skipping`; request continues.
- **Errors** — logged as warnings; `ctx.errors` gets `memgraphrag:...`; upstream request still forwarded.

MemGraphRAG does **not** dense-fallback inside the stage. If no facts score, it returns no additional hits.

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Stage never runs | `ENABLE_MEMGRAPHRAG=false` or cognitive off | Set flags; check budget (`STAGE_BUDGET_MEMGRAPHRAG_MS`) |
| `memory is empty, skipping` | Index missing or no triples extracted | Rebuild; check LLM JSON output; avoid `--skip-relations` |
| `No facts scored` at runtime | Facts built with `--skip-embed` | Rebuild with embed step |
| Build very slow | LLM bottleneck | Lower `--max-chunks` for trial; tune `--concurrency` |
| Sparse / wrong passages | Small or biased sample | Increase `--max-chunks`; check `--stratify-field` |
| No `memgraphrag` in trace | Budget skip or stage disabled | Raise `COGNITIVE_LATENCY_BUDGET_MS`; grep `MemGraphRAG` in logs |
| Reranker warnings | Sidecar down or wrong URL | Start reranker or leave `RERANKER_URL` unset (degraded mode) |

## Operational notes

- **Index size**: SQLite holds schemas, facts, passages, and fact embedding vectors. Disk grows with chunk sample size and triple count.
- **Collection schema**: If you do not control the Qdrant collection, coordinate with the owner — sampling reads points only; see [Ingest and admin](ingest-and-admin.md).
- **Reindex cadence**: Re-run the build script after major ingest updates; there is no live sync from Qdrant to MemGraphRAG.
- **Paper reference**: Implements memory-guided retrieval inspired by MemGraphRAG (arxiv 2606.00610); code in `rag_proxy/memgraphrag/`.

## Related docs

- [Cognitive pipeline](cognitive-pipeline.md) — rollout order; enable MemGraphRAG in phase 3
- [COGNITIVE_RAG_PLAN.md](COGNITIVE_RAG_PLAN.md) — full flag matrix
- [Ingest and admin](ingest-and-admin.md) — how content reaches Qdrant before graph build
- [docker/README.md](../docker/README.md) — reranker and cognitive sidecars

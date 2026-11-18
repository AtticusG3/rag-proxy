# rag_proxy

Transparent RAG middleware: optional tiered cognitive pipeline in front of any
OpenAI-compatible upstream. This glossary defines the shared language for the
proxy, its sidecars, ingest, and admin.

## Language

**Proxy**:
The FastAPI service that accepts OpenAI-compatible chat requests, optionally
enriches them with retrieved context, and forwards them upstream.
_Avoid_: gateway, middleware server

**Upstream**:
The OpenAI-compatible chat backend the proxy forwards to (`LLAMA_SWAP_URL`).
_Avoid_: backend, LLM server

**Fail-open**:
The invariant that a cognitive or retrieval error never breaks the upstream
request; the request proceeds without injected context.
_Avoid_: graceful degradation, fallback

**Inject**:
Placing retrieved context text into the outgoing request's messages, wrapped in
marker lines so it can be stripped later.
_Avoid_: augment, stuff

**Cognitive pipeline**:
The optional tiered stage sequence (tier0 → … → context) run before forwarding,
enabled by a master switch and per-subsystem flags.
_Avoid_: smart mode, RAG mode

**Tier0**:
The first cognitive stage: a cheap heuristic fast path that skips retrieval for
short greetings and acknowledgements. It never skips a query containing `?` or
an infra signal, but a later gating stage may still skip such a query.
_Avoid_: pre-filter, cheap check

**Gating**:
The cognitive stage that decides per request between `skip`, `light` and `full`
retrieval (`RetrievalDecision`), honouring `X-RAG-Mode: off` / `force` and
observe-only mode while the flag is being rolled out.
_Avoid_: routing (the model-choice stage), throttling

**Embed pool**:
The production embeddings deployment behind `EMBED_URL` (model id
`embed-pool`): an HTTP `/v1/embeddings` endpoint on the upstream host, pooling
replicas of one embedding model behind a single address.
_Avoid_: embed server (ambiguous), embedding sidecar

**Sidecar**:
A small standalone HTTP service next to the proxy that offloads one retrieval
concern (rerank, sparse, dense ANN, MCP tools).
_Avoid_: worker, helper service

**Dense retrieval**:
Vector similarity search over Qdrant embeddings.
_Avoid_: semantic search

**Sparse retrieval**:
Lexical BM25 search served by the sparse sidecar.
_Avoid_: keyword search

**Hybrid retrieval**:
Dense and sparse results merged by reciprocal-rank fusion.
_Avoid_: combined search

**Chunk**:
One indexed text segment of a document, with its payload metadata.
_Avoid_: passage (outside MemGraphRAG), segment

**Ingest**:
The worker pipeline that turns source files into chunks and writes them to
Qdrant (+ sparse reindex).
_Avoid_: indexing (ambiguous with sparse rebuild)

**Dual-write**:
Ingest's write path across Qdrant and the TurboVec sidecar: each chunk upsert
goes to Qdrant and is also sent best-effort to TurboVec `/add`, and a source
delete calls Qdrant delete and best-effort TurboVec `/remove`. Calls that fail
on the sidecar are logged and dropped; the scheduled sidecar reindex from Qdrant
catches up anything missed.
_Avoid_: replication, mirroring

**MemGraphRAG layers**:
The three SQLite memory layers the MemGraphRAG index is built from: *schema*
(`head_type, relation, tail_type` patterns for denoising and grouping), *fact*
(`head, relation, tail` triples, the nodes PPR walks) and *passage* (original
chunk text plus its `chunk_id`, what actually gets injected).
_Avoid_: graph tiers, memory levels

**Scroll**:
Paged iteration over a Qdrant collection used to build a snapshot.
_Avoid_: dump, export

**Snapshot**:
The on-disk artifacts (CSC matrix, metadata, params) a sidecar memory-maps at
startup to serve search without a rebuild.
_Avoid_: cache file, index dump

**Rebuild**:
A full re-derivation of a sidecar's index from the Qdrant collection via
scroll; the expensive exception to snapshot serving.
_Avoid_: refresh (incremental update), reindex-verb-outside-API

**Atomic swap**:
Replacing the live snapshot directory only after a complete replacement exists,
so search never serves a half-built index.
_Avoid_: rename-over, hot reload

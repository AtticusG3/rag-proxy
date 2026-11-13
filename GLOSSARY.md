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

"""Setting field metadata for the admin settings UI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TargetEnv = Literal["admin", "proxy", "sqlite"]


@dataclass(frozen=True)
class SettingField:
    key: str
    label: str
    group: str
    field_type: Literal["bool", "int", "float", "str", "select", "text"]
    target: TargetEnv
    default: str
    hot: bool = False
    options: tuple[str, ...] = ()
    help_text: str = ""


SETTING_FIELDS: tuple[SettingField, ...] = (
    # Ingest (admin env, hot-reload worker)
    SettingField("INGEST_BATCH_SIZE", "Embed batch size", "ingest", "int", "admin", "64", hot=True),
    SettingField("INGEST_EMBED_CONCURRENCY", "Embed concurrency", "ingest", "int", "admin", "4", hot=True),
    SettingField(
        "INGEST_SPARSE_REINDEX",
        "BM25 reindex mode",
        "ingest",
        "select",
        "admin",
        "idle",
        hot=True,
        options=("off", "each", "idle"),
        help_text="When to rebuild the sparse/BM25 sidecar index after dense ingest.",
    ),
    SettingField("INGEST_STALL_MINUTES", "Stall timeout (minutes)", "ingest", "int", "admin", "15", hot=True),
    SettingField("INGEST_MAX_ARTICLES", "ZIM article cap (0=unlimited)", "ingest", "int", "admin", "0", hot=True),
    SettingField("EMBED_MAX_CHARS", "Max chars per embed", "ingest", "int", "admin", "2000", hot=True),
    SettingField(
        "INGEST_EMBED_URLS",
        "Ingest embed pool URLs",
        "ingest",
        "text",
        "admin",
        "",
        hot=True,
        help_text="Comma-separated pool embed endpoints (ports 18089+); blank uses Primary embed URL only.",
    ),
    SettingField(
        "EMBED_URL",
        "Primary embed URL",
        "ingest",
        "str",
        "admin",
        "http://127.0.0.1:8089",
        help_text="Query embed (mirrored to rag-proxy.env). Bulk pool URLs go in Ingest embed pool URLs.",
    ),
    SettingField("QDRANT_URL", "Qdrant URL", "ingest", "str", "admin", "http://127.0.0.1:6333"),
    SettingField("QDRANT_COLLECTION", "Qdrant collection", "ingest", "str", "admin", "nomad_knowledge_base"),
    SettingField("SPARSE_INDEX_URL", "Sparse/BM25 sidecar URL", "ingest", "str", "admin", ""),
    SettingField("RAG_PROXY_URL", "RAG proxy URL", "ingest", "str", "admin", "http://127.0.0.1:8081"),
    SettingField(
        "INGEST_CHUNK_SIZE_TOKENS",
        "Chunk size (tokens)",
        "ingest",
        "int",
        "admin",
        "512",
        help_text="Requires python scripts/requeue_all_ingest.py for existing files.",
    ),
    SettingField(
        "INGEST_CHUNK_OVERLAP_TOKENS",
        "Chunk overlap (tokens)",
        "ingest",
        "int",
        "admin",
        "64",
        help_text="Requires python scripts/requeue_all_ingest.py for existing files.",
    ),
    SettingField(
        "INGEST_CHUNK_TOKENIZER",
        "Chunk tokenizer",
        "ingest",
        "str",
        "admin",
        "nomic-ai/nomic-embed-text-v1.5",
    ),
    SettingField(
        "INGEST_CHUNK_SEMANTIC",
        "Semantic chunking",
        "ingest",
        "bool",
        "admin",
        "true",
        help_text="Requires chonkie[semantic]. Re-chunk existing files after change.",
    ),
    SettingField(
        "INGEST_CHUNK_SEMANTIC_MODEL",
        "Semantic chunk model",
        "ingest",
        "str",
        "admin",
        "minishlab/potion-base-32M",
    ),
    SettingField(
        "INGEST_CHUNK_MIN_TOKENS",
        "Min chunk tokens (merge)",
        "ingest",
        "int",
        "admin",
        "100",
        help_text="Undersized chunks below this are merged when possible.",
    ),
    # Proxy upstream + legacy RAG (rag-proxy.env)
    SettingField(
        "LLAMA_SWAP_URL",
        "llama-swap URL",
        "proxy_rag",
        "str",
        "proxy",
        "http://127.0.0.1:8080",
        help_text="Upstream for chat completions and INTENT_MODEL. Model id only in Intent model field.",
    ),
    SettingField("EMBED_RETRIES", "Embed retries", "proxy_rag", "int", "proxy", "2"),
    SettingField("TOP_K", "Top K chunks", "proxy_rag", "int", "proxy", "5"),
    SettingField("SIMILARITY_THRESHOLD", "Similarity threshold", "proxy_rag", "float", "proxy", "0.65"),
    SettingField("ENABLE_HYBRID_RETRIEVAL", "Hybrid dense+BM25", "proxy_rag", "bool", "proxy", "false"),
    SettingField("HYBRID_DENSE_WEIGHT", "Dense weight (hybrid)", "proxy_rag", "float", "proxy", "0.7"),
    SettingField("ENABLE_RERANKER", "Cross-encoder rerank", "proxy_rag", "bool", "proxy", "false"),
    SettingField("RERANKER_URL", "Reranker sidecar URL", "proxy_rag", "str", "proxy", "http://127.0.0.1:8095"),
    SettingField("RERANK_TOP_K", "Rerank top K", "proxy_rag", "int", "proxy", "5"),
    SettingField("RETRIEVAL_CANDIDATE_K", "Retrieval candidate K", "proxy_rag", "int", "proxy", "20"),
    SettingField("RECENCY_WEIGHT", "Recency weight", "proxy_rag", "float", "proxy", "0.1"),
    SettingField("RERANK_TIMEOUT_MS", "Rerank timeout (ms)", "proxy_rag", "int", "proxy", "200"),
    SettingField("ENABLE_SEMANTIC_DEDUPE", "Semantic dedupe", "proxy_rag", "bool", "proxy", "false"),
    SettingField("ENABLE_EMBED_CACHE", "Embed cache", "proxy_rag", "bool", "proxy", "false"),
    # Cognitive pipeline
    SettingField("ENABLE_COGNITIVE_PIPELINE", "Cognitive pipeline", "cognitive", "bool", "proxy", "false"),
    SettingField("ENABLE_TIER0_HEURISTICS", "Tier-0 heuristics", "cognitive", "bool", "proxy", "false"),
    SettingField("ENABLE_INTENT_ROUTER", "Intent router", "cognitive", "bool", "proxy", "false"),
    SettingField("ENABLE_RETRIEVAL_GATING", "Retrieval gating", "cognitive", "bool", "proxy", "false"),
    SettingField("GATING_LOG_ONLY", "Gating log-only (bake-in)", "cognitive", "bool", "proxy", "false"),
    SettingField("ENABLE_QUERY_REWRITE", "Query rewrite", "cognitive", "bool", "proxy", "false"),
    SettingField("ENABLE_QUERY_REWRITE_LLM", "Query rewrite (LLM)", "cognitive", "bool", "proxy", "false"),
    SettingField("ENABLE_GRAPH_LOOKUP", "Graph lookup", "cognitive", "bool", "proxy", "false"),
    SettingField("ENABLE_TOOLS", "Tool stage", "cognitive", "bool", "proxy", "false"),
    SettingField("ENABLE_ROLLING_MEMORY", "Rolling memory", "cognitive", "bool", "proxy", "false"),
    SettingField(
        "INTENT_MODEL",
        "Intent model",
        "cognitive",
        "str",
        "proxy",
        "",
        help_text="llama-swap model id (e.g. openrouter/owl-alpha). Uses LLAMA_SWAP_URL, not Build LLM API URL.",
    ),
    SettingField("INTENT_CONFIDENCE_THRESHOLD", "Intent confidence threshold", "cognitive", "float", "proxy", "0.55"),
    SettingField("INTENT_TIMEOUT_MS", "Intent timeout (ms)", "cognitive", "int", "proxy", "150"),
    SettingField("COGNITIVE_LATENCY_BUDGET_MS", "Latency budget (ms)", "cognitive", "int", "proxy", "800"),
    SettingField("STAGE_BUDGET_RETRIEVE_MS", "Retrieve stage budget (ms)", "cognitive", "int", "proxy", "50"),
    SettingField("STAGE_BUDGET_GRAPH_MS", "Graph stage budget (ms)", "cognitive", "int", "proxy", "100"),
    SettingField(
        "GRAPH_DB_PATH",
        "Graph SQLite path",
        "cognitive",
        "str",
        "proxy",
        "/var/lib/rag_proxy/graph.sqlite",
    ),
    SettingField("GRAPH_MAX_DEPTH", "Graph max depth", "cognitive", "int", "proxy", "2"),
    SettingField(
        "MEMORY_DB_PATH",
        "Rolling memory DB path",
        "cognitive",
        "str",
        "proxy",
        "/var/lib/rag_proxy/memory.sqlite",
    ),
    SettingField("MEMORY_TTL_HOURS", "Memory TTL (hours)", "cognitive", "int", "proxy", "72"),
    SettingField("MEMORY_REFRESH_TURNS", "Memory refresh turns", "cognitive", "int", "proxy", "8"),
    # MemGraphRAG runtime (proxy env)
    SettingField("ENABLE_MEMGRAPHRAG", "MemGraphRAG stage", "memgraphrag", "bool", "proxy", "false"),
    SettingField(
        "MEMGRAPHRAG_DB_PATH",
        "MemGraphRAG SQLite path",
        "memgraphrag",
        "str",
        "proxy",
        "/var/lib/rag_proxy/memgraphrag.sqlite",
    ),
    SettingField("MEMGRAPHRAG_FACT_TOP_K", "Fact top K", "memgraphrag", "int", "proxy", "20"),
    SettingField("MEMGRAPHRAG_PPR_DAMPING", "PPR damping", "memgraphrag", "float", "proxy", "0.85"),
    SettingField("MEMGRAPHRAG_PPR_ITERATIONS", "PPR iterations", "memgraphrag", "int", "proxy", "20"),
    SettingField(
        "MEMGRAPHRAG_PASSAGE_NODE_WEIGHT",
        "Passage node weight",
        "memgraphrag",
        "float",
        "proxy",
        "0.5",
    ),
    SettingField("STAGE_BUDGET_MEMGRAPHRAG_MS", "Stage budget (ms)", "memgraphrag", "int", "proxy", "200"),
    # MemGraphRAG offline build (sqlite only)
    SettingField(
        "MEMGRAPH_BUILD_LLM_URL",
        "Build LLM API URL",
        "memgraph_build",
        "str",
        "sqlite",
        "http://192.168.1.202:8081/v1",
        help_text="OpenAI-compatible endpoint for entity/relation extraction (remote qwen when local GPU is busy).",
    ),
    SettingField(
        "MEMGRAPH_BUILD_LLM_MODEL",
        "Build LLM model",
        "memgraph_build",
        "str",
        "sqlite",
        "qwen3.5-9b-turbo",
    ),
    SettingField("MEMGRAPH_BUILD_MAX_CHUNKS", "Sample chunk count", "memgraph_build", "int", "sqlite", "1000"),
    SettingField("MEMGRAPH_BUILD_CONCURRENCY", "Build LLM concurrency", "memgraph_build", "int", "sqlite", "3"),
    SettingField("MEMGRAPH_BUILD_EMBED_URL", "Build embed URL", "memgraph_build", "str", "sqlite", "",
                 help_text="Blank uses EMBED_URL from ingest/proxy settings."),
    SettingField(
        "MEMGRAPH_BUILD_SKIP_RELATIONS",
        "Skip relation extraction",
        "memgraph_build",
        "bool",
        "sqlite",
        "false",
    ),
    # Observability
    SettingField("LOG_LEVEL", "Log level", "observability", "select", "admin", "INFO", options=("DEBUG", "INFO", "WARNING", "ERROR")),
    SettingField("ENABLE_REQUEST_TRACE", "Request traces", "observability", "bool", "proxy", "true"),
    SettingField("ENABLE_JSON_LOGS", "JSON pipeline logs", "observability", "bool", "proxy", "false"),
    SettingField("ENABLE_METRICS", "Prometheus metrics", "observability", "bool", "proxy", "false"),
    SettingField("ENABLE_TRANSCRIPT_CAPTURE", "Transcript capture", "observability", "bool", "proxy", "false"),
)

SETTING_GROUPS: tuple[str, ...] = (
    "ingest",
    "proxy_rag",
    "cognitive",
    "memgraphrag",
    "memgraph_build",
    "observability",
)

GROUP_LABELS: dict[str, str] = {
    "ingest": "Dense ingest & BM25",
    "proxy_rag": "Proxy RAG retrieval",
    "cognitive": "Cognitive pipeline",
    "memgraphrag": "MemGraphRAG runtime",
    "memgraph_build": "MemGraphRAG index build",
    "observability": "Logging & metrics",
}

INGEST_PAUSED_KEY = "INGEST_PAUSED"

# Shared keys saved on the ingest tab are mirrored into rag-proxy.env.
INGEST_MIRROR_TO_PROXY: tuple[str, ...] = (
    "EMBED_URL",
    "QDRANT_URL",
    "QDRANT_COLLECTION",
    "SPARSE_INDEX_URL",
)

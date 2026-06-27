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
        help_text="Comma-separated embed endpoints; blank uses EMBED_URL only.",
    ),
    SettingField("EMBED_URL", "Primary embed URL", "ingest", "str", "admin", "http://127.0.0.1:18089"),
    SettingField("QDRANT_URL", "Qdrant URL", "ingest", "str", "admin", "http://127.0.0.1:6333"),
    SettingField("QDRANT_COLLECTION", "Qdrant collection", "ingest", "str", "admin", "nomad_knowledge_base"),
    SettingField("SPARSE_INDEX_URL", "Sparse/BM25 sidecar URL", "ingest", "str", "admin", "http://127.0.0.1:18096"),
    SettingField("RAG_PROXY_URL", "RAG proxy URL", "ingest", "str", "admin", "http://127.0.0.1:8081"),
    # Proxy legacy RAG
    SettingField("TOP_K", "Top K chunks", "proxy_rag", "int", "proxy", "5"),
    SettingField("SIMILARITY_THRESHOLD", "Similarity threshold", "proxy_rag", "float", "proxy", "0.65"),
    SettingField("ENABLE_HYBRID_RETRIEVAL", "Hybrid dense+BM25", "proxy_rag", "bool", "proxy", "false"),
    SettingField("HYBRID_DENSE_WEIGHT", "Dense weight (hybrid)", "proxy_rag", "float", "proxy", "0.7"),
    SettingField("ENABLE_RERANKER", "Cross-encoder rerank", "proxy_rag", "bool", "proxy", "false"),
    SettingField("RERANKER_URL", "Reranker sidecar URL", "proxy_rag", "str", "proxy", "http://127.0.0.1:18095"),
    SettingField("RERANK_TOP_K", "Rerank top K", "proxy_rag", "int", "proxy", "5"),
    # Cognitive pipeline
    SettingField("ENABLE_COGNITIVE_PIPELINE", "Cognitive pipeline", "cognitive", "bool", "proxy", "false"),
    SettingField("ENABLE_TIER0_HEURISTICS", "Tier-0 heuristics", "cognitive", "bool", "proxy", "false"),
    SettingField("ENABLE_INTENT_ROUTER", "Intent router", "cognitive", "bool", "proxy", "false"),
    SettingField("ENABLE_RETRIEVAL_GATING", "Retrieval gating", "cognitive", "bool", "proxy", "false"),
    SettingField("GATING_LOG_ONLY", "Gating log-only (bake-in)", "cognitive", "bool", "proxy", "false"),
    SettingField("ENABLE_QUERY_REWRITE", "Query rewrite", "cognitive", "bool", "proxy", "false"),
    SettingField("ENABLE_GRAPH_LOOKUP", "Graph lookup", "cognitive", "bool", "proxy", "false"),
    SettingField("ENABLE_TOOLS", "Tool stage", "cognitive", "bool", "proxy", "false"),
    SettingField("ENABLE_ROLLING_MEMORY", "Rolling memory", "cognitive", "bool", "proxy", "false"),
    SettingField("INTENT_MODEL", "Intent model", "cognitive", "str", "proxy", ""),
    SettingField("COGNITIVE_LATENCY_BUDGET_MS", "Latency budget (ms)", "cognitive", "int", "proxy", "800"),
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
    SettingField("MEMGRAPH_BUILD_EMBED_URL", "Build embed URL", "memgraph_build", "str", "sqlite", ""),
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

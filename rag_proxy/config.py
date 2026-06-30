"""Environment configuration and feature flags."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field

from rag_proxy.env_parse import parse_bool

log = logging.getLogger("rag-proxy")


def _env_bool(name: str, default: bool = False) -> bool:
    """Parse a boolean environment variable."""
    return parse_bool(os.getenv(name), default)


def _env_int(name: str, default: int) -> int:
    """Parse an integer env var; fall back on invalid values."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        log.warning("Invalid integer for %s=%r; using default %s", name, raw, default)
        return default


def _env_float(name: str, default: float) -> float:
    """Parse a float env var; fall back on invalid values."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        log.warning("Invalid float for %s=%r; using default %s", name, raw, default)
        return default


@dataclass
class Settings:
    """Runtime configuration loaded from environment variables."""
    # Upstream / data plane
    llama_swap_url: str = field(default_factory=lambda: os.getenv("LLAMA_SWAP_URL", "http://127.0.0.1:8080"))
    embed_url: str = field(default_factory=lambda: os.getenv("EMBED_URL", "http://127.0.0.1:8089"))
    qdrant_url: str = field(default_factory=lambda: os.getenv("QDRANT_URL", "http://192.168.1.36:6333"))
    qdrant_collection: str = field(
        default_factory=lambda: os.getenv("QDRANT_COLLECTION", "nomad_knowledge_base")
    )
    top_k: int = field(default_factory=lambda: _env_int("TOP_K", 5))
    similarity_threshold: float = field(
        default_factory=lambda: _env_float("SIMILARITY_THRESHOLD", 0.65)
    )
    proxy_host: str = field(default_factory=lambda: os.getenv("PROXY_HOST", "0.0.0.0"))
    proxy_port: int = field(default_factory=lambda: _env_int("PROXY_PORT", 8088))
    proxy_internal_token: str = field(
        default_factory=lambda: os.getenv("PROXY_INTERNAL_TOKEN", "")
    )
    upstream_timeout_sec: float = field(
        default_factory=lambda: _env_float("UPSTREAM_TIMEOUT_SEC", 600.0)
    )
    upstream_max_connections: int = field(
        default_factory=lambda: _env_int("UPSTREAM_MAX_CONNECTIONS", 50)
    )
    upstream_max_keepalive: int = field(
        default_factory=lambda: _env_int("UPSTREAM_MAX_KEEPALIVE", 0)
    )
    upstream_keepalive_expiry_sec: float = field(
        default_factory=lambda: _env_float("UPSTREAM_KEEPALIVE_EXPIRY_SEC", 15.0)
    )
    upstream_idle_sweep_sec: float = field(
        default_factory=lambda: _env_float("UPSTREAM_IDLE_SWEEP_SEC", 30.0)
    )
    upstream_stream_abandon_sec: float = field(
        default_factory=lambda: _env_float("UPSTREAM_STREAM_ABANDON_SEC", 120.0)
    )
    embed_max_chars: int = field(default_factory=lambda: _env_int("EMBED_MAX_CHARS", 2000))
    embed_retries: int = field(default_factory=lambda: _env_int("EMBED_RETRIES", 2))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    # Cognitive pipeline master
    enable_cognitive_pipeline: bool = field(
        default_factory=lambda: _env_bool("ENABLE_COGNITIVE_PIPELINE", False)
    )
    enable_tier0_heuristics: bool = field(
        default_factory=lambda: _env_bool("ENABLE_TIER0_HEURISTICS", False)
    )
    enable_intent_router: bool = field(
        default_factory=lambda: _env_bool("ENABLE_INTENT_ROUTER", False)
    )
    enable_retrieval_gating: bool = field(
        default_factory=lambda: _env_bool("ENABLE_RETRIEVAL_GATING", False)
    )
    enable_query_rewrite: bool = field(
        default_factory=lambda: _env_bool("ENABLE_QUERY_REWRITE", False)
    )
    enable_query_rewrite_llm: bool = field(
        default_factory=lambda: _env_bool("ENABLE_QUERY_REWRITE_LLM", False)
    )
    enable_hybrid_retrieval: bool = field(
        default_factory=lambda: _env_bool("ENABLE_HYBRID_RETRIEVAL", False)
    )
    enable_reranker: bool = field(default_factory=lambda: _env_bool("ENABLE_RERANKER", False))
    enable_semantic_dedupe: bool = field(
        default_factory=lambda: _env_bool("ENABLE_SEMANTIC_DEDUPE", False)
    )
    enable_graph_lookup: bool = field(
        default_factory=lambda: _env_bool("ENABLE_GRAPH_LOOKUP", False)
    )
    enable_memgraphrag: bool = field(
        default_factory=lambda: _env_bool("ENABLE_MEMGRAPHRAG", False)
    )
    enable_model_routing: bool = field(
        default_factory=lambda: _env_bool("ENABLE_MODEL_ROUTING", False)
    )
    enable_tools: bool = field(default_factory=lambda: _env_bool("ENABLE_TOOLS", False))
    enable_rolling_memory: bool = field(
        default_factory=lambda: _env_bool("ENABLE_ROLLING_MEMORY", False)
    )
    enable_embed_cache: bool = field(default_factory=lambda: _env_bool("ENABLE_EMBED_CACHE", False))
    enable_json_logs: bool = field(default_factory=lambda: _env_bool("ENABLE_JSON_LOGS", False))
    enable_request_trace: bool = field(
        default_factory=lambda: _env_bool("ENABLE_REQUEST_TRACE", True)
    )
    enable_tokenizer_estimate: bool = field(
        default_factory=lambda: _env_bool("ENABLE_TOKENIZER_ESTIMATE", False)
    )
    gating_log_only: bool = field(default_factory=lambda: _env_bool("GATING_LOG_ONLY", False))

    # Budgets
    cognitive_latency_budget_ms: int = field(
        default_factory=lambda: _env_int("COGNITIVE_LATENCY_BUDGET_MS", 800)
    )
    stage_budget_routing_ms: int = field(
        default_factory=lambda: _env_int("STAGE_BUDGET_ROUTING_MS", 0)
    )
    stage_budget_rewrite_ms: int = field(
        default_factory=lambda: _env_int("STAGE_BUDGET_REWRITE_MS", 20)
    )
    stage_budget_retrieve_ms: int = field(
        default_factory=lambda: _env_int("STAGE_BUDGET_RETRIEVE_MS", 50)
    )
    stage_budget_graph_ms: int = field(
        default_factory=lambda: _env_int("STAGE_BUDGET_GRAPH_MS", 100)
    )
    stage_budget_memgraphrag_ms: int = field(
        default_factory=lambda: _env_int("STAGE_BUDGET_MEMGRAPHRAG_MS", 200)
    )
    retrieval_candidate_k: int = field(default_factory=lambda: _env_int("RETRIEVAL_CANDIDATE_K", 20))
    context_budget_ratio: float = field(
        default_factory=lambda: _env_float("CONTEXT_BUDGET_RATIO", 0.25)
    )
    context_fallback_chars: int = field(
        default_factory=lambda: _env_int("CONTEXT_FALLBACK_CHARS", 8000)
    )
    default_completion_reserve: int = field(
        default_factory=lambda: _env_int("DEFAULT_COMPLETION_RESERVE", 1024)
    )

    # Intent
    intent_model: str = field(default_factory=lambda: os.getenv("INTENT_MODEL", ""))
    intent_confidence_threshold: float = field(
        default_factory=lambda: _env_float("INTENT_CONFIDENCE_THRESHOLD", 0.55)
    )
    intent_timeout_ms: int = field(default_factory=lambda: _env_int("INTENT_TIMEOUT_MS", 150))

    # Hybrid / rerank
    hybrid_dense_weight: float = field(
        default_factory=lambda: _env_float("HYBRID_DENSE_WEIGHT", 0.7)
    )
    sparse_index_url: str = field(default_factory=lambda: os.getenv("SPARSE_INDEX_URL", ""))
    recency_weight: float = field(default_factory=lambda: _env_float("RECENCY_WEIGHT", 0.1))
    reranker_url: str = field(
        default_factory=lambda: os.getenv("RERANKER_URL", "http://127.0.0.1:8095")
    )
    rerank_top_k: int = field(default_factory=lambda: _env_int("RERANK_TOP_K", 5))
    rerank_timeout_ms: int = field(default_factory=lambda: _env_int("RERANK_TIMEOUT_MS", 200))

    # Graph / tools / memory
    graph_db_path: str = field(
        default_factory=lambda: os.getenv("GRAPH_DB_PATH", "/var/lib/rag_proxy/graph.sqlite")
    )
    graph_max_depth: int = field(default_factory=lambda: _env_int("GRAPH_MAX_DEPTH", 2))
    tool_allowed_roots: str = field(
        default_factory=lambda: os.getenv("TOOL_ALLOWED_ROOTS", "")
    )
    tool_timeout_sec: float = field(default_factory=lambda: _env_float("TOOL_TIMEOUT_SEC", 5.0))
    tool_budget_ms: int = field(default_factory=lambda: _env_int("TOOL_BUDGET_MS", 300))
    tool_max_output_chars: int = field(default_factory=lambda: _env_int("TOOL_MAX_OUTPUT_CHARS", 4000))
    memory_db_path: str = field(
        default_factory=lambda: os.getenv("MEMORY_DB_PATH", "/var/lib/rag_proxy/memory.sqlite")
    )
    memory_ttl_hours: int = field(default_factory=lambda: _env_int("MEMORY_TTL_HOURS", 72))
    memory_refresh_turns: int = field(default_factory=lambda: _env_int("MEMORY_REFRESH_TURNS", 8))

    # MemGraphRAG
    memgraphrag_db_path: str = field(
        default_factory=lambda: os.getenv("MEMGRAPHRAG_DB_PATH", "/var/lib/rag_proxy/memgraphrag.sqlite")
    )
    memgraphrag_fact_top_k: int = field(
        default_factory=lambda: _env_int("MEMGRAPHRAG_FACT_TOP_K", 20)
    )
    memgraphrag_ppr_damping: float = field(
        default_factory=lambda: _env_float("MEMGRAPHRAG_PPR_DAMPING", 0.85)
    )
    memgraphrag_ppr_iterations: int = field(
        default_factory=lambda: _env_int("MEMGRAPHRAG_PPR_ITERATIONS", 20)
    )
    memgraphrag_passage_node_weight: float = field(
        default_factory=lambda: _env_float("MEMGRAPHRAG_PASSAGE_NODE_WEIGHT", 0.5)
    )

    # MemGraphRAG offline index build (scripts/build_memgraphrag_index.py, rag-admin)
    memgraph_build_llm_url: str = field(
        default_factory=lambda: os.getenv("MEMGRAPH_BUILD_LLM_URL", "http://127.0.0.1:8080/v1")
    )
    memgraph_build_llm_model: str = field(
        default_factory=lambda: os.getenv("MEMGRAPH_BUILD_LLM_MODEL", "qwen3.5-9b-turbo")
    )
    memgraph_build_max_chunks: int = field(
        default_factory=lambda: _env_int("MEMGRAPH_BUILD_MAX_CHUNKS", 1000)
    )
    memgraph_build_concurrency: int = field(
        default_factory=lambda: _env_int("MEMGRAPH_BUILD_CONCURRENCY", 3)
    )
    memgraph_build_embed_url: str = field(
        default_factory=lambda: os.getenv("MEMGRAPH_BUILD_EMBED_URL", "")
    )
    memgraph_build_skip_relations: bool = field(
        default_factory=lambda: _env_bool("MEMGRAPH_BUILD_SKIP_RELATIONS", False)
    )

    # Model registry / routing
    model_registry_ttl_sec: int = field(
        default_factory=lambda: _env_int("MODEL_REGISTRY_TTL_SEC", 300)
    )
    model_capabilities_json: str = field(
        default_factory=lambda: os.getenv("MODEL_CAPABILITIES_JSON", "")
    )
    model_routes_json: str = field(default_factory=lambda: os.getenv("MODEL_ROUTES_JSON", ""))
    model_routing_mode: str = field(
        default_factory=lambda: os.getenv("MODEL_ROUTING_MODE", "suggest")
    )

    # Transcript capture
    enable_transcript_capture: bool = field(
        default_factory=lambda: _env_bool("ENABLE_TRANSCRIPT_CAPTURE", False)
    )
    finetune_log_path: str = field(
        default_factory=lambda: os.getenv(
            "FINETUNE_LOG_PATH",
            "/var/lib/rag_proxy/capture/finetune.jsonl",
        )
    )
    rag_improvement_log_path: str = field(
        default_factory=lambda: os.getenv(
            "RAG_IMPROVEMENT_LOG_PATH",
            "/var/lib/rag_proxy/capture/rag_improvement.jsonl",
        )
    )
    transcript_strip_proxy_artefacts: bool = field(
        default_factory=lambda: _env_bool("TRANSCRIPT_STRIP_PROXY_ARTEFACTS", True)
    )
    transcript_header_opt_in: bool = field(
        default_factory=lambda: _env_bool("TRANSCRIPT_HEADER_OPT_IN", False)
    )
    transcript_sample_rate: float = field(
        default_factory=lambda: _env_float("TRANSCRIPT_SAMPLE_RATE", 1.0)
    )
    transcript_hit_preview_chars: int = field(
        default_factory=lambda: _env_int("TRANSCRIPT_HIT_PREVIEW_CHARS", 300)
    )
    enable_rag_corpus_auto_ingest: bool = field(
        default_factory=lambda: _env_bool("ENABLE_RAG_CORPUS_AUTO_INGEST", False)
    )
    rag_corpus_collection: str = field(
        default_factory=lambda: os.getenv(
            "RAG_CORPUS_COLLECTION",
            "nomad_conversation_derived",
        )
    )
    rag_corpus_min_answer_chars: int = field(
        default_factory=lambda: _env_int("RAG_CORPUS_MIN_ANSWER_CHARS", 100)
    )
    rag_corpus_require_chunks: bool = field(
        default_factory=lambda: _env_bool("RAG_CORPUS_REQUIRE_CHUNKS", False)
    )

    # Observability
    enable_metrics: bool = field(default_factory=lambda: _env_bool("ENABLE_METRICS", False))

    # Tier 0 tuning
    tier0_max_chars: int = field(default_factory=lambda: _env_int("TIER0_MAX_CHARS", 80))

    def model_routes(self) -> dict[str, str]:
        """Parse MODEL_ROUTES_JSON into a route map."""
        raw = self.model_routes_json.strip()
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}

    def model_capabilities_overrides(self) -> dict:
        """Parse MODEL_CAPABILITIES_JSON overrides."""
        raw = self.model_capabilities_json.strip()
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}

    def tool_roots(self) -> list[str]:
        """Split TOOL_ALLOWED_ROOTS into a path list."""
        raw = self.tool_allowed_roots.strip()
        if not raw:
            return []
        return [p.strip() for p in raw.split(",") if p.strip()]


settings = Settings()

CHAT_PATHS = {"v1/chat/completions", "api/chat"}

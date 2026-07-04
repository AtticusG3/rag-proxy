"""In-app tuning maps and field placeholders for the settings UI."""

from __future__ import annotations

from dataclasses import dataclass

from rag_admin.settings_schema import SettingField


@dataclass(frozen=True)
class TuningItem:
    key: str
    note: str


@dataclass(frozen=True)
class TuningSection:
    title: str
    intro: str
    items: tuple[TuningItem, ...] = ()


GROUP_TUNING: dict[str, tuple[TuningSection, ...]] = {
    "ingest": (
        TuningSection(
            "GPU embed pool (planner inputs)",
            "Edit max instances and VRAM below, save, then click Scale ingest capacity. That pauses ingest, runs chunk+embed benchmarks, applies the plan to the embed pool, and resumes ingest with updated settings.",
            (
                TuningItem(
                    "NOMIC_POOL_MAX_INSTANCES",
                    "Cap on nomic-embed@PORT systemd units (llama-server count).",
                ),
                TuningItem(
                    "NOMIC_POOL_PARALLEL",
                    "--parallel per unit (systemd); should match planner parallel per instance.",
                ),
                TuningItem(
                    "NOMIC_POOL_VRAM_RESERVE_MIB",
                    "Leave headroom for chat models on the same GPU.",
                ),
            ),
        ),
        TuningSection(
            "Throughput pipeline",
            "Files run in parallel; embed batches share one GPU pool cap.",
            (
                TuningItem(
                    "INGEST_FILE_CONCURRENCY",
                    "Parallel files (threads). Empty = auto max(1, min(4, pool URLs)).",
                ),
                TuningItem(
                    "INGEST_EMBED_CONCURRENCY",
                    "Max in-flight embed batches across all files. Match pool: instances x --parallel.",
                ),
                TuningItem(
                    "INGEST_BATCH_SIZE",
                    "Texts per embed HTTP request. Smaller batches + higher concurrency often wins.",
                ),
                TuningItem(
                    "INGEST_EMBED_URLS",
                    "GPU pool endpoints (ports 18089+). Usually written by scale_ingest_capacity.py.",
                ),
                TuningItem(
                    "EMBED_URL",
                    "Query-time embed (:8089). Bulk ingest uses the pool URLs above.",
                ),
            ),
        ),
        TuningSection(
            "Chunking (Chonkie)",
            "Strategy is picked per file (code, markdown, semantic, sentence). These set size and overlap.",
            (
                TuningItem(
                    "INGEST_CHUNK_SIZE_TOKENS",
                    "Target size in tokens (512 is the nomic sweet spot).",
                ),
                TuningItem(
                    "INGEST_CHUNK_OVERLAP_TOKENS",
                    "Overlap in tokens (~12.5% of size; not kilobytes).",
                ),
                TuningItem(
                    "INGEST_CHUNK_SEMANTIC",
                    "Enable semantic boundaries for dense PDFs; slower CPU. false = faster bulk.",
                ),
                TuningItem(
                    "INGEST_CHUNK_MIN_TOKENS",
                    "Merge adjacent tiny chunks before embed.",
                ),
            ),
        ),
        TuningSection(
            "After chunk changes",
            "Changing chunk size, overlap, or semantic mode does not re-embed existing files automatically.",
            (
                TuningItem(
                    "requeue",
                    "Run: python scripts/requeue_all_ingest.py",
                ),
            ),
        ),
    ),
    "proxy_rag": (
        TuningSection(
            "Legacy retrieval",
            "Used when cognitive pipeline is off, or as the retrieve stage baseline.",
            (
                TuningItem("TOP_K", "Chunks injected into the prompt after threshold filter."),
                TuningItem("SIMILARITY_THRESHOLD", "Minimum cosine score (0.65 is a common starting point)."),
                TuningItem("ENABLE_HYBRID_RETRIEVAL", "Dense Qdrant + BM25 sidecar with RRF merge."),
                TuningItem("ENABLE_RERANKER", "Cross-encoder rerank via RERANKER_URL sidecar."),
            ),
        ),
    ),
    "cognitive": (
        TuningSection(
            "Rollout order",
            "Master switch ENABLE_COGNITIVE_PIPELINE, then enable stages one at a time. Use GATING_LOG_ONLY to observe gating without skipping retrieval.",
            (
                TuningItem("COGNITIVE_LATENCY_BUDGET_MS", "Total pipeline time cap; stages skip when budget is low."),
                TuningItem("INTENT_MODEL", "llama-swap model id at LLAMA_SWAP_URL (not the MemGraph build LLM)."),
            ),
        ),
    ),
    "memgraphrag": (
        TuningSection(
            "Runtime stage",
            "Runs after graph lookup in the cognitive pipeline. Requires a built SQLite index and ENABLE_MEMGRAPHRAG=true.",
            (
                TuningItem("MEMGRAPHRAG_DB_PATH", "SQLite graph built offline (see MemGraphRAG index build tab)."),
                TuningItem("STAGE_BUDGET_MEMGRAPHRAG_MS", "Time cap for fact scoring + PPR + passage fetch."),
            ),
        ),
    ),
    "memgraph_build": (
        TuningSection(
            "Offline index",
            "Separate from ingest. Samples Qdrant chunks, calls an OpenAI-compatible LLM for entities/relations.",
            (
                TuningItem("MEMGRAPH_BUILD_LLM_URL", "Remote GPU-friendly endpoint when local llama-swap is busy."),
                TuningItem("MEMGRAPH_BUILD_MAX_CHUNKS", "Sample size for build (not full corpus unless raised)."),
            ),
        ),
    ),
    "observability": (
        TuningSection(
            "Operators",
            "Request traces log pipeline stage summaries. Metrics exposes GET /metrics on the proxy port.",
            (
                TuningItem("ENABLE_REQUEST_TRACE", "Per-request pipeline summary in proxy logs."),
                TuningItem("ENABLE_METRICS", "Prometheus counters on proxy port (not a separate listener)."),
            ),
        ),
    ),
}


def field_placeholder(field: SettingField) -> str:
    """Placeholder text for empty inputs (schema default or auto sentinel)."""
    if field.key == "INGEST_FILE_CONCURRENCY":
        return "auto (1-4 by pool size)"
    if field.default == "":
        return "(empty)"
    return field.default

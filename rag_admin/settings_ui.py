"""UI metadata for the settings control plane (sections, advanced mode, workflows)."""

from __future__ import annotations

from dataclasses import dataclass

# Keys shown only when Settings mode=advanced. Basic mode omits these from the form
# (save_group skips keys not in the form, so overrides are preserved).
ADVANCED_KEYS: frozenset[str] = frozenset(
    {
        "INGEST_CHUNK_CONCURRENCY",
        "INGEST_HIDE_INDEXED_MINUTES",
        "INGEST_MAX_ARTICLES",
        "EMBED_MAX_CHARS",
        "INGEST_EMBED_URLS",
        "NOMIC_POOL_MIN_INSTANCES",
        "NOMIC_POOL_PARALLEL_PER_INSTANCE",
        "NOMIC_POOL_VRAM_PER_INSTANCE_MIB",
        "NOMIC_POOL_PORT_BASE",
        "NOMIC_POOL_GPU_INDEX",
        "RAG_PROXY_URL",
        "INGEST_CHUNK_TOKENIZER",
        "INGEST_CHUNK_SEMANTIC_MODEL",
        "INGEST_CHUNK_MIN_TOKENS",
        "LLAMA_SWAP_URL",
        "EMBED_RETRIES",
        "QDRANT_VECTORS_ON_DISK",
        "HYBRID_DENSE_WEIGHT",
        "RERANKER_URL",
        "RERANK_TOP_K",
        "RETRIEVAL_CANDIDATE_K",
        "RECENCY_WEIGHT",
        "RERANK_TIMEOUT_MS",
        "ENABLE_SEMANTIC_DEDUPE",
        "ENABLE_EMBED_CACHE",
        "INTENT_MODEL_URL",
        "INTENT_MODEL_AUTO_TTL_SEC",
        "INTENT_CONFIDENCE_THRESHOLD",
        "INTENT_TIMEOUT_MS",
        "STAGE_BUDGET_RETRIEVE_MS",
        "STAGE_BUDGET_GRAPH_MS",
        "GRAPH_DB_PATH",
        "GRAPH_MAX_DEPTH",
        "MEMORY_DB_PATH",
        "MEMORY_TTL_HOURS",
        "MEMORY_REFRESH_TURNS",
        "MEMGRAPHRAG_PPR_DAMPING",
        "MEMGRAPHRAG_PPR_ITERATIONS",
        "MEMGRAPHRAG_PASSAGE_NODE_WEIGHT",
        "STAGE_BUDGET_MEMGRAPHRAG_MS",
        "MEMGRAPH_BUILD_CONCURRENCY",
        "MEMGRAPH_BUILD_EMBED_URL",
        "MEMGRAPH_BUILD_SKIP_RELATIONS",
        "ENABLE_JSON_LOGS",
        "ENABLE_TRANSCRIPT_CAPTURE",
    }
)

# Per-group ordered sections for the left rail and form headings.
GROUP_SECTIONS: dict[str, tuple[tuple[str, str], ...]] = {
    "ingest": (
        ("throughput", "Throughput"),
        ("pool", "Pool planner"),
        ("connections", "Connections"),
        ("chunking", "Chunking"),
    ),
    "proxy_rag": (
        ("retrieval", "Retrieval"),
        ("hybrid", "Hybrid & rerank"),
        ("endpoints", "Endpoints"),
    ),
    "cognitive": (
        ("stages", "Stage switches"),
        ("intent", "Intent & budgets"),
        ("graph_memory", "Graph & memory"),
    ),
    "memgraphrag": (
        ("runtime", "Runtime"),
    ),
    "memgraph_build": (
        ("build", "Build"),
    ),
    "observability": (
        ("logging", "Logging"),
    ),
}

FIELD_SECTION: dict[str, str] = {
    "INGEST_BATCH_SIZE": "throughput",
    "INGEST_EMBED_CONCURRENCY": "throughput",
    "INGEST_FILE_CONCURRENCY": "throughput",
    "INGEST_CHUNK_CONCURRENCY": "throughput",
    "INGEST_SPARSE_REINDEX": "throughput",
    "INGEST_STALL_MINUTES": "throughput",
    "INGEST_HIDE_INDEXED_MINUTES": "throughput",
    "INGEST_MAX_ARTICLES": "throughput",
    "EMBED_MAX_CHARS": "throughput",
    "INGEST_EMBED_URLS": "throughput",
    "INGEST_TURBOVEC_REINDEX": "throughput",
    "NOMIC_POOL_MAX_INSTANCES": "pool",
    "NOMIC_POOL_MIN_INSTANCES": "pool",
    "NOMIC_POOL_PARALLEL_PER_INSTANCE": "pool",
    "NOMIC_POOL_VRAM_PER_INSTANCE_MIB": "pool",
    "NOMIC_POOL_VRAM_RESERVE_MIB": "pool",
    "NOMIC_POOL_PORT_BASE": "pool",
    "NOMIC_POOL_GPU_INDEX": "pool",
    "EMBED_URL": "connections",
    "QDRANT_URL": "connections",
    "QDRANT_COLLECTION": "connections",
    "SPARSE_INDEX_URL": "connections",
    "TURBOVEC_URL": "connections",
    "RAG_PROXY_URL": "connections",
    "INGEST_CHUNK_SIZE_TOKENS": "chunking",
    "INGEST_CHUNK_OVERLAP_TOKENS": "chunking",
    "INGEST_CHUNK_TOKENIZER": "chunking",
    "INGEST_CHUNK_SEMANTIC": "chunking",
    "INGEST_CHUNK_SEMANTIC_MODEL": "chunking",
    "INGEST_CHUNK_MIN_TOKENS": "chunking",
    "TOP_K": "retrieval",
    "SIMILARITY_THRESHOLD": "retrieval",
    "DENSE_BACKEND": "retrieval",
    "QDRANT_VECTORS_ON_DISK": "retrieval",
    "ENABLE_HYBRID_RETRIEVAL": "hybrid",
    "HYBRID_DENSE_WEIGHT": "hybrid",
    "ENABLE_RERANKER": "hybrid",
    "RERANKER_URL": "hybrid",
    "RERANK_TOP_K": "hybrid",
    "RETRIEVAL_CANDIDATE_K": "hybrid",
    "RECENCY_WEIGHT": "hybrid",
    "RERANK_TIMEOUT_MS": "hybrid",
    "ENABLE_SEMANTIC_DEDUPE": "hybrid",
    "ENABLE_EMBED_CACHE": "hybrid",
    "LLAMA_SWAP_URL": "endpoints",
    "EMBED_RETRIES": "endpoints",
    "ENABLE_COGNITIVE_PIPELINE": "stages",
    "ENABLE_TIER0_HEURISTICS": "stages",
    "ENABLE_INTENT_ROUTER": "stages",
    "ENABLE_RETRIEVAL_GATING": "stages",
    "GATING_LOG_ONLY": "stages",
    "ENABLE_QUERY_REWRITE": "stages",
    "ENABLE_QUERY_REWRITE_LLM": "stages",
    "ENABLE_GRAPH_LOOKUP": "stages",
    "ENABLE_TOOLS": "stages",
    "ENABLE_ROLLING_MEMORY": "stages",
    "INTENT_MODEL": "intent",
    "INTENT_MODEL_URL": "intent",
    "INTENT_MODEL_AUTO_TTL_SEC": "intent",
    "INTENT_CONFIDENCE_THRESHOLD": "intent",
    "INTENT_TIMEOUT_MS": "intent",
    "COGNITIVE_LATENCY_BUDGET_MS": "intent",
    "STAGE_BUDGET_RETRIEVE_MS": "intent",
    "STAGE_BUDGET_GRAPH_MS": "intent",
    "GRAPH_DB_PATH": "graph_memory",
    "GRAPH_MAX_DEPTH": "graph_memory",
    "MEMORY_DB_PATH": "graph_memory",
    "MEMORY_TTL_HOURS": "graph_memory",
    "MEMORY_REFRESH_TURNS": "graph_memory",
    "ENABLE_MEMGRAPHRAG": "runtime",
    "MEMGRAPHRAG_DB_PATH": "runtime",
    "MEMGRAPHRAG_FACT_TOP_K": "runtime",
    "MEMGRAPHRAG_PPR_DAMPING": "runtime",
    "MEMGRAPHRAG_PPR_ITERATIONS": "runtime",
    "MEMGRAPHRAG_PASSAGE_NODE_WEIGHT": "runtime",
    "STAGE_BUDGET_MEMGRAPHRAG_MS": "runtime",
    "MEMGRAPH_BUILD_LLM_URL": "build",
    "MEMGRAPH_BUILD_LLM_MODEL": "build",
    "MEMGRAPH_BUILD_MAX_CHUNKS": "build",
    "MEMGRAPH_BUILD_CONCURRENCY": "build",
    "MEMGRAPH_BUILD_EMBED_URL": "build",
    "MEMGRAPH_BUILD_SKIP_RELATIONS": "build",
    "LOG_LEVEL": "logging",
    "ENABLE_REQUEST_TRACE": "logging",
    "ENABLE_JSON_LOGS": "logging",
    "ENABLE_METRICS": "logging",
    "ENABLE_TRANSCRIPT_CAPTURE": "logging",
}

COGNITIVE_STAGE_MAP: tuple[tuple[str, str], ...] = (
    ("tier0", "Tier-0"),
    ("intent", "Intent"),
    ("gate", "Gate"),
    ("rewrite", "Rewrite"),
    ("retrieve", "Retrieve"),
    ("graph", "Graph"),
    ("memgraph", "MemGraph"),
    ("tools", "Tools"),
    ("memory", "Memory"),
    ("context", "Context"),
)

GROUP_NAV_LABELS: dict[str, str] = {
    "ingest": "Ingest",
    "proxy_rag": "Retrieval",
    "cognitive": "Cognitive",
    "memgraphrag": "MemGraph runtime",
    "memgraph_build": "MemGraph build",
    "observability": "Observability",
}

GROUP_NAV_HINTS: dict[str, str] = {
    "ingest": "Capacity, pool, chunking",
    "proxy_rag": "TOP_K, hybrid, rerank",
    "cognitive": "Pipeline stages & budgets",
    "memgraphrag": "Runtime stage knobs",
    "memgraph_build": "Offline index build",
    "observability": "Traces & metrics",
}


@dataclass(frozen=True)
class WorkflowStep:
    label: str
    done: bool = False
    active: bool = False
    note: str = ""


def workflow_steps_for(
    group: str,
    *,
    pool_env_exists: bool,
    pool_scale_job: bool,
    pool_scale_starting: bool,
    build_job: bool,
    ingest_paused: bool,
) -> tuple[WorkflowStep, ...]:
    if group == "ingest":
        scaling = pool_scale_job or pool_scale_starting
        return (
            WorkflowStep("Edit plan", done=True, note="Pool planner fields"),
            WorkflowStep("Save", done=True, note="Persist overrides"),
            WorkflowStep(
                "Scale",
                active=scaling or not pool_env_exists,
                done=pool_env_exists and not scaling,
                note="Runs capacity planner",
            ),
            WorkflowStep(
                "Verify",
                done=pool_env_exists and not scaling and not ingest_paused,
                active=pool_env_exists and not scaling and ingest_paused,
                note="Pool URLs + ingest running",
            ),
        )
    if group == "memgraph_build":
        return (
            WorkflowStep("Configure LLM", done=True),
            WorkflowStep("Start build", active=bool(build_job), done=False),
            WorkflowStep("Enable runtime", note="MemGraph runtime tab"),
            WorkflowStep("Restart proxy", note="Apply ENABLE_MEMGRAPHRAG"),
        )
    if group == "cognitive":
        return (
            WorkflowStep("Master on", done=True, note="ENABLE_COGNITIVE_PIPELINE"),
            WorkflowStep("Log-only gate", note="GATING_LOG_ONLY bake-in"),
            WorkflowStep("Enable stages", note="One at a time"),
            WorkflowStep("Restart proxy", note="Apply proxy env"),
        )
    if group == "proxy_rag":
        return (
            WorkflowStep("Tune retrieval", done=True),
            WorkflowStep("Save", done=True),
            WorkflowStep("Restart proxy", note="Cold apply"),
            WorkflowStep("Smoke query", note="Check injected context"),
        )
    return ()


def settings_query(tab: str, mode: str) -> str:
    safe_mode = mode if mode in ("basic", "advanced") else "basic"
    return f"/settings?tab={tab}&mode={safe_mode}"


def _truthy(raw: str | None) -> bool:
    return (raw or "").strip().lower() in ("1", "true", "yes", "on")


def _nonempty(raw: str | None) -> bool:
    return bool((raw or "").strip())


# Human labels for select option values (value -> label). Missing keys fall back to value.
SELECT_OPTION_LABELS: dict[str, dict[str, str]] = {
    "INGEST_SPARSE_REINDEX": {
        "off": "Off",
        "each": "After each file",
        "idle": "When idle",
    },
    "INGEST_TURBOVEC_REINDEX": {
        "off": "Off",
        "each": "After each file",
        "idle": "When idle",
    },
    "DENSE_BACKEND": {
        "qdrant": "Qdrant",
        "turbovec": "TurboVec",
    },
    "LOG_LEVEL": {
        "DEBUG": "Debug",
        "INFO": "Info",
        "WARNING": "Warning",
        "ERROR": "Error",
    },
}

# When key is enabled (truthy), listed bool keys should also be on.
SETTING_REQUIRES: dict[str, tuple[str, ...]] = {
    "ENABLE_TIER0_HEURISTICS": ("ENABLE_COGNITIVE_PIPELINE",),
    "ENABLE_INTENT_ROUTER": ("ENABLE_COGNITIVE_PIPELINE",),
    "ENABLE_RETRIEVAL_GATING": ("ENABLE_COGNITIVE_PIPELINE",),
    "GATING_LOG_ONLY": ("ENABLE_COGNITIVE_PIPELINE", "ENABLE_RETRIEVAL_GATING"),
    "ENABLE_QUERY_REWRITE": ("ENABLE_COGNITIVE_PIPELINE",),
    "ENABLE_QUERY_REWRITE_LLM": (
        "ENABLE_COGNITIVE_PIPELINE",
        "ENABLE_QUERY_REWRITE",
    ),
    "ENABLE_GRAPH_LOOKUP": ("ENABLE_COGNITIVE_PIPELINE",),
    "ENABLE_TOOLS": ("ENABLE_COGNITIVE_PIPELINE",),
    "ENABLE_ROLLING_MEMORY": ("ENABLE_COGNITIVE_PIPELINE",),
    "ENABLE_MEMGRAPHRAG": ("ENABLE_COGNITIVE_PIPELINE",),
}

# When key is enabled, listed keys should be non-empty (URL / model / path).
SETTING_REQUIRES_NONEMPTY: dict[str, tuple[str, ...]] = {
    "ENABLE_HYBRID_RETRIEVAL": ("SPARSE_INDEX_URL",),
    "ENABLE_RERANKER": ("RERANKER_URL",),
    "ENABLE_INTENT_ROUTER": ("INTENT_MODEL",),
    "ENABLE_QUERY_REWRITE_LLM": ("INTENT_MODEL",),
    "ENABLE_GRAPH_LOOKUP": ("GRAPH_DB_PATH",),
    "ENABLE_ROLLING_MEMORY": ("MEMORY_DB_PATH",),
    "ENABLE_MEMGRAPHRAG": ("MEMGRAPHRAG_DB_PATH",),
}

# Soft warnings for specific select values (not just on/off).
SETTING_VALUE_REQUIRES_NONEMPTY: dict[str, dict[str, tuple[str, ...]]] = {
    "DENSE_BACKEND": {
        "turbovec": ("TURBOVEC_URL",),
    },
}


def option_labels_for(key: str, options: tuple[str, ...]) -> list[dict[str, str]]:
    labels = SELECT_OPTION_LABELS.get(key, {})
    return [{"value": opt, "label": labels.get(opt, opt)} for opt in options]


def collect_dep_keys() -> frozenset[str]:
    keys: set[str] = set(SETTING_REQUIRES)
    keys.update(SETTING_REQUIRES_NONEMPTY)
    keys.update(SETTING_VALUE_REQUIRES_NONEMPTY)
    for reqs in SETTING_REQUIRES.values():
        keys.update(reqs)
    for reqs in SETTING_REQUIRES_NONEMPTY.values():
        keys.update(reqs)
    for by_value in SETTING_VALUE_REQUIRES_NONEMPTY.values():
        for reqs in by_value.values():
            keys.update(reqs)
    return frozenset(keys)


def evaluate_setting_warnings(values: dict[str, str]) -> list[dict[str, str]]:
    """Return soft config warnings for the current effective values."""
    from rag_admin.settings_schema import SETTING_FIELDS

    labels = {field.key: field.label for field in SETTING_FIELDS}
    warnings: list[dict[str, str]] = []

    def label_of(key: str) -> str:
        return labels.get(key, key)

    for key, parents in SETTING_REQUIRES.items():
        if not _truthy(values.get(key)):
            continue
        missing = [parent for parent in parents if not _truthy(values.get(parent))]
        if not missing:
            continue
        warnings.append(
            {
                "key": key,
                "severity": "requires",
                "message": (
                    f"{label_of(key)} is on, but requires "
                    + ", ".join(label_of(m) for m in missing)
                    + "."
                ),
            }
        )

    for key, needed in SETTING_REQUIRES_NONEMPTY.items():
        if not _truthy(values.get(key)):
            continue
        missing = [dep for dep in needed if not _nonempty(values.get(dep))]
        if not missing:
            continue
        warnings.append(
            {
                "key": key,
                "severity": "requires_nonempty",
                "message": (
                    f"{label_of(key)} is on, but "
                    + ", ".join(label_of(m) for m in missing)
                    + " is empty."
                ),
            }
        )

    for key, by_value in SETTING_VALUE_REQUIRES_NONEMPTY.items():
        current = (values.get(key) or "").strip()
        needed = by_value.get(current)
        if not needed:
            continue
        missing = [dep for dep in needed if not _nonempty(values.get(dep))]
        if not missing:
            continue
        warnings.append(
            {
                "key": key,
                "severity": "value_requires_nonempty",
                "message": (
                    f"{label_of(key)} is set to {current}, but "
                    + ", ".join(label_of(m) for m in missing)
                    + " is empty."
                ),
            }
        )

    return warnings


def settings_control_plane(values: dict[str, str]) -> dict:
    """JSON payload for live switch/select dependency checks in settings.js."""
    from rag_admin.settings_schema import SETTING_FIELDS

    labels = {field.key: field.label for field in SETTING_FIELDS}
    dep_keys = collect_dep_keys()
    scoped_values = {key: values.get(key, "") for key in dep_keys}
    # Always include current-page fields that participate in value warnings.
    for key in SETTING_VALUE_REQUIRES_NONEMPTY:
        scoped_values[key] = values.get(key, "")
    return {
        "values": scoped_values,
        "labels": {key: labels.get(key, key) for key in set(scoped_values) | set(labels)},
        "requires": {key: list(parents) for key, parents in SETTING_REQUIRES.items()},
        "requires_nonempty": {
            key: list(deps) for key, deps in SETTING_REQUIRES_NONEMPTY.items()
        },
        "value_requires_nonempty": {
            key: {val: list(deps) for val, deps in by_value.items()}
            for key, by_value in SETTING_VALUE_REQUIRES_NONEMPTY.items()
        },
        "warnings": evaluate_setting_warnings(values),
    }

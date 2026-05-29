"""Request context and pipeline enums."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PipelineTier(str, Enum):
    TIER0_BYPASS = "tier0_bypass"
    TIER1_LIGHT = "tier1_light"
    TIER2_RETRIEVAL = "tier2_retrieval"
    TIER3_HEAVY = "tier3_heavy"


class IntentLabel(str, Enum):
    SIMPLE_CHAT = "simple_chat"
    INFRA_DEBUG = "infra_debug"
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    RESEARCH = "research"
    SUMMARIZATION = "summarization"
    TROUBLESHOOTING = "troubleshooting"
    LOG_ANALYSIS = "log_analysis"
    PLANNING = "planning"
    CREATIVE = "creative"
    RETRIEVAL_HEAVY = "retrieval_heavy"
    REASONING_HEAVY = "reasoning_heavy"
    UNKNOWN = "unknown"


class RetrievalDecision(str, Enum):
    SKIP = "skip"
    LIGHT = "light"
    FULL = "full"


@dataclass
class ChunkHit:
    id: str
    text: str
    score: float
    source: str = "dense"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RequestContext:
    path: str = ""
    raw_body: bytes = b""
    data: dict[str, Any] = field(default_factory=dict)
    messages: list[dict] = field(default_factory=list)
    query_text: str | None = None
    requested_model: str | None = None
    stream: bool = False
    rag_mode_header: str | None = None
    no_cache: bool = False
    conversation_id: str | None = None

    tier: PipelineTier = PipelineTier.TIER2_RETRIEVAL
    intent: IntentLabel = IntentLabel.UNKNOWN
    intent_confidence: float = 0.0
    retrieval: RetrievalDecision = RetrievalDecision.FULL
    retrieval_query: str | None = None
    selected_model: str | None = None
    hits: list[ChunkHit] = field(default_factory=list)
    chunk_texts: list[str] = field(default_factory=list)
    injected_tokens_est: int = 0
    stage_trace: list[str] = field(default_factory=list)
    latency_ms: dict[str, float] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    cache_hits: list[str] = field(default_factory=list)
    trace_id: str = ""
    cognitive_start_ms: float = 0.0
    gating_would_skip: bool = False

    def effective_query(self) -> str | None:
        return self.retrieval_query or self.query_text

    def top_k_for_retrieval(self, settings_top_k: int, light_top_k: int = 3) -> int:
        if self.retrieval == RetrievalDecision.LIGHT:
            return min(light_top_k, settings_top_k)
        return settings_top_k

"""Tier 1: intent classification (rules + optional tiny model)."""

from __future__ import annotations

import re

from rag_proxy.registry.models import ModelRegistry
from rag_proxy.clients.llama_swap import classify_intent_via_model, resolve_intent_model
from rag_proxy.config import settings
from rag_proxy.context import IntentLabel, RequestContext

_CODE_FILE = re.compile(r"\.(py|rs|go|ts|js|yaml|yml|toml|md)\b", re.I)
_LOG_LINE = re.compile(r"^\d{4}-\d{2}-\d{2}|ERROR|WARN|FATAL|traceback", re.I | re.M)


def _rules_intent(query: str) -> tuple[IntentLabel, float]:
    """Classify intent via keyword and pattern rules."""
    q = query.lower()
    if _LOG_LINE.search(query) or "log" in q and ("analyze" in q or "parse" in q):
        return IntentLabel.LOG_ANALYSIS, 0.85
    if any(w in q for w in ("kubectl", "docker", "systemctl", "compose", "qdrant", "systemd")):
        return IntentLabel.INFRA_DEBUG, 0.8
    if "review" in q and ("pr" in q or "code" in q or _CODE_FILE.search(query)):
        return IntentLabel.CODE_REVIEW, 0.8
    if any(w in q for w in ("implement", "write a function", "refactor", "fix this bug")):
        return IntentLabel.CODE_GENERATION, 0.75
    if _CODE_FILE.search(query):
        return IntentLabel.CODE_GENERATION, 0.7
    if any(w in q for w in ("summarize", "summary", "tl;dr")):
        return IntentLabel.SUMMARIZATION, 0.75
    if any(w in q for w in ("plan", "roadmap", "architecture")):
        return IntentLabel.PLANNING, 0.7
    if any(w in q for w in ("research", "compare", "explain in depth")):
        return IntentLabel.RESEARCH, 0.7
    if any(w in q for w in ("troubleshoot", "not working", "broken", "why does")):
        return IntentLabel.TROUBLESHOOTING, 0.75
    if any(w in q for w in ("write a story", "poem", "creative")):
        return IntentLabel.CREATIVE, 0.7
    if len(query) > 200 and any(w in q for w in ("document", "docs", "knowledge")):
        return IntentLabel.RETRIEVAL_HEAVY, 0.65
    if any(w in q for w in ("step by step", "reason", "prove")):
        return IntentLabel.REASONING_HEAVY, 0.65
    if any(w in q for w in ("what is", "what's", "who is", "tell me about", "how does", "explain", "describe")):
        return IntentLabel.RESEARCH, 0.7
    if len(query) < 60:
        return IntentLabel.SIMPLE_CHAT, 0.6
    return IntentLabel.UNKNOWN, 0.0


def _intent_from_dict(data: dict) -> tuple[IntentLabel, float]:
    """Parse intent label and confidence from model JSON."""
    label = data.get("intent", "unknown")
    try:
        conf = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        return IntentLabel.UNKNOWN, 0.0
    try:
        return IntentLabel(label), conf
    except ValueError:
        return IntentLabel.UNKNOWN, 0.0


async def run_intent(ctx: RequestContext, _registry: ModelRegistry) -> None:
    """Classify query intent into ctx.intent."""
    if not ctx.query_text:
        return

    label, conf = _rules_intent(ctx.query_text)
    if label == IntentLabel.UNKNOWN and settings.intent_model:
        model = await resolve_intent_model()
        if model:
            data = await classify_intent_via_model(
                model,
                ctx.query_text,
                settings.intent_timeout_ms,
                base_url=settings.intent_base_url(),
            )
            if data:
                label, conf = _intent_from_dict(data)

    if conf < settings.intent_confidence_threshold:
        label = IntentLabel.UNKNOWN

    ctx.intent = label
    ctx.intent_confidence = conf
    ctx.stage_trace.append(f"intent:{label.value}:{conf:.2f}")

"""Model capability discovery from llama-swap."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

from rag_proxy.clients.llama_swap import fetch_models
from rag_proxy.config import settings

log = logging.getLogger("rag-proxy")


@dataclass
class ModelCapabilities:
    model_id: str
    context_length: int | None = None
    supports_tools: bool | None = None
    supports_vision: bool | None = None
    supports_embedding: bool | None = None
    tags: set[str] = field(default_factory=set)


_CODING_HINTS = re.compile(r"code|coder|bonsai|deepseek-coder|starcoder", re.I)
_REASONING_HINTS = re.compile(r"reason|think|opus|qwen3", re.I)
_EMBED_HINTS = re.compile(r"embed|nomic", re.I)


class ModelRegistry:
    def __init__(self) -> None:
        self._cache: dict[str, ModelCapabilities] = {}
        self._ids: list[str] = []
        self._fetched_at: float = 0.0

    async def refresh(self, force: bool = False) -> None:
        now = time.time()
        if not force and self._cache and now - self._fetched_at < settings.model_registry_ttl_sec:
            return
        models = await fetch_models()
        self._ids = []
        self._cache = {}
        for m in models:
            mid = m.get("id", "")
            if not mid:
                continue
            self._ids.append(mid)
            caps = self._infer_capabilities(mid, m)
            self._cache[mid] = caps
        self._apply_env_overrides()
        self._fetched_at = now

    def _apply_env_overrides(self) -> None:
        overrides = settings.model_capabilities_overrides()
        for mid, raw in overrides.items():
            if not isinstance(raw, dict):
                continue
            base = self._cache.get(mid) or ModelCapabilities(model_id=mid)
            if "context_length" in raw:
                base.context_length = int(raw["context_length"])
            if "tags" in raw and isinstance(raw["tags"], list):
                base.tags.update(raw["tags"])
            self._cache[mid] = base

    def _infer_capabilities(self, model_id: str, raw: dict) -> ModelCapabilities:
        tags: set[str] = {"chat"}
        if _CODING_HINTS.search(model_id):
            tags.add("coding")
        if _REASONING_HINTS.search(model_id):
            tags.add("reasoning")
        if _EMBED_HINTS.search(model_id):
            tags.add("embedding")
        ctx = None
        for key in ("context_length", "max_context_length", "n_ctx"):
            if key in raw:
                try:
                    ctx = int(raw[key])
                except (TypeError, ValueError):
                    pass
        return ModelCapabilities(
            model_id=model_id,
            context_length=ctx,
            tags=tags,
        )

    def get(self, model_id: str) -> ModelCapabilities:
        return self._cache.get(model_id) or ModelCapabilities(model_id=model_id)

    def list_ids(self) -> list[str]:
        return list(self._ids)

    def resolve_context_limit(self, model_id: str | None) -> int:
        if model_id:
            caps = self.get(model_id)
            if caps.context_length:
                return caps.context_length
        return settings.context_fallback_chars * 4 // 4  # char budget fallback as pseudo tokens

    def model_exists(self, model_id: str) -> bool:
        if not self._ids:
            return True
        return model_id in self._ids

"""Shared client bundle for pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field

from rag_proxy.registry.models import ModelRegistry


@dataclass
class ClientBundle:
    model_registry: ModelRegistry = field(default_factory=ModelRegistry)

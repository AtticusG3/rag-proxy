"""MemGraphRAG: Memory-based Graph Retrieval-Augmented Generation.

Based on arxiv 2606.00610 — Three-layer memory (schema/fact/passage) with
conflict resolution and Personalized PageRank retrieval.

Note: imports are lazy so that the offline build script can use ThreeLayerMemory
without pulling in retrieval.py (which needs the full rag_proxy config).
"""

from __future__ import annotations

__all__ = ["ThreeLayerMemory", "load_memory", "MemGraphRetriever"]


def __getattr__(name: str):
    if name in ("ThreeLayerMemory", "load_memory"):
        from rag_proxy.memgraphrag.memory import ThreeLayerMemory, load_memory as _load
        # Cache in module namespace to avoid re-import
        globals()["ThreeLayerMemory"] = ThreeLayerMemory
        globals()["load_memory"] = _load
        return ThreeLayerMemory if name == "ThreeLayerMemory" else _load
    if name == "MemGraphRetriever":
        from rag_proxy.memgraphrag.retrieval import MemGraphRetriever
        globals()["MemGraphRetriever"] = MemGraphRetriever
        return MemGraphRetriever
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

"""Qdrant writer vector-size env and dim-mismatch fail-loud behavior."""

from __future__ import annotations

import pytest

from ingest.qdrant_writer import (
    DEFAULT_VECTOR_SIZE,
    build_point,
    configured_vector_size,
)


def test_configured_vector_size_defaults_to_768(monkeypatch) -> None:
    monkeypatch.delenv("QDRANT_VECTOR_SIZE", raising=False)
    assert configured_vector_size() == DEFAULT_VECTOR_SIZE


def test_configured_vector_size_reads_env(monkeypatch) -> None:
    monkeypatch.setenv("QDRANT_VECTOR_SIZE", "2560")
    assert configured_vector_size() == 2560


def test_build_point_raises_on_embedding_dim_mismatch(monkeypatch) -> None:
    """Wrong-width upserts must fail loudly so nomic/Qwen spaces are never mixed silently."""
    monkeypatch.setenv("QDRANT_VECTOR_SIZE", "768")
    with pytest.raises(ValueError, match="embedding dim 3 does not match QDRANT_VECTOR_SIZE=768"):
        build_point(
            text="chunk",
            source="src",
            title="t",
            chunk_idx=0,
            embedding=[0.1, 0.2, 0.3],
        )


def test_build_point_accepts_matching_dim(monkeypatch) -> None:
    monkeypatch.setenv("QDRANT_VECTOR_SIZE", "4")
    point = build_point(
        text="chunk",
        source="src",
        title="t",
        chunk_idx=0,
        embedding=[0.1, 0.2, 0.3, 0.4],
    )
    assert point["vector"] == [0.1, 0.2, 0.3, 0.4]

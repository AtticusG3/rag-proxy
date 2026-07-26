"""Capacity planner must cap ingest when Qdrant collection is large."""

from __future__ import annotations

from ingest.capacity_planner import (
    CapacityPlannerConfig,
    plan_ingest_capacity,
    qdrant_ingest_limits,
)
from ingest.embed_pool import EmbedPoolConfig
from ingest.host_profile import GpuProfile, HostProfile
from ingest.qdrant_profile import QdrantCollectionProfile


def _host(*, cores: int = 6, ram_available: int = 62000) -> HostProfile:
    return HostProfile(
        cpu_model="Intel i5-9600T",
        cpu_logical_cores=cores,
        ram_total_mib=65536,
        ram_available_mib=ram_available,
        disks=(),
        gpu=GpuProfile(
            name="Tesla V100-SXM2-32GB",
            total_mib=32768,
            used_mib=1024,
            free_mib=31744,
            utilization_pct=0,
        ),
        probed_at="2026-07-08T00:00:00+00:00",
    )


def _profile(points: int) -> QdrantCollectionProfile:
    return QdrantCollectionProfile(
        points_count=points,
        indexed_vectors_count=points,
        segment_count=10,
        status="green",
        optimizer_status="ok",
    )


def test_qdrant_huge_collection_caps_embed_and_batch_on_slow_cpu() -> None:
    """2M+ points on a 6-core host must not run embed-saturated upserts."""
    _, limits = qdrant_ingest_limits(
        _profile(2_500_000),
        host=_host(cores=6),
        cfg=CapacityPlannerConfig(),
    )
    assert limits.file_concurrency_cap == 1
    assert limits.batch_size_cap == 32
    assert limits.embed_concurrency_cap == 16


def test_qdrant_large_collection_on_slow_cpu_allows_parallel_files() -> None:
    """~1M points on 6 cores soft-caps; must not force serial file ingest."""
    _, limits = qdrant_ingest_limits(
        _profile(1_140_000),
        host=_host(cores=6),
        cfg=CapacityPlannerConfig(),
    )
    assert limits.file_concurrency_cap == 2
    assert limits.batch_size_cap == 64
    assert limits.embed_concurrency_cap == 24


def test_plan_applies_qdrant_caps_when_profile_provided(monkeypatch) -> None:
    """Planner should lower knobs when Qdrant probe reports a huge collection."""
    monkeypatch.setenv("QDRANT_URL", "http://127.0.0.1:6333")
    monkeypatch.setenv("QDRANT_COLLECTION", "nomad_knowledge_base")

    def fake_probe(_url: str, _collection: str, **_kwargs):
        return _profile(2_532_722)

    monkeypatch.setattr("ingest.capacity_planner.probe_qdrant_collection", fake_probe)

    plan = plan_ingest_capacity(
        host=_host(cores=6),
        pool_config=EmbedPoolConfig(max_instances=4, parallel_per_instance=16),
        planner_config=CapacityPlannerConfig(max_file_concurrency=8),
        semantic_requested=False,
    )

    assert plan.ingest_file_concurrency == 1
    assert plan.ingest_batch_size <= 32
    assert plan.ingest_embed_concurrency <= 16
    assert plan.qdrant is not None
    assert "qdrant" in plan.rationale


def test_plan_large_qdrant_keeps_bench_chunk_concurrency(monkeypatch) -> None:
    """Chunk concurrency is CPU-bound and must survive Qdrant file=2 soft caps."""
    from ingest.bench_fit import BenchFit

    monkeypatch.setenv("QDRANT_URL", "http://127.0.0.1:6333")
    monkeypatch.setenv("QDRANT_COLLECTION", "nomad_knowledge_base")
    monkeypatch.setattr(
        "ingest.capacity_planner.probe_qdrant_collection",
        lambda *_a, **_k: _profile(1_140_000),
    )
    bench = BenchFit(chunk_concurrency=3, chunk_chunks_per_min=9000.0)
    plan = plan_ingest_capacity(
        host=_host(cores=6),
        pool_config=EmbedPoolConfig(max_instances=2, parallel_per_instance=8),
        planner_config=CapacityPlannerConfig(max_file_concurrency=8),
        semantic_requested=False,
        bench=bench,
    )
    assert plan.ingest_file_concurrency == 2
    # cpu_cap = 6 // 2 = 3
    assert plan.ingest_chunk_concurrency == 3
    assert plan.ingest_embed_concurrency <= 24

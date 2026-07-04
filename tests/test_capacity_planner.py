"""Tests for the multi-resource ingest capacity planner."""

from __future__ import annotations

from ingest.bench_fit import BenchFit
from ingest.capacity_planner import (
    CapacityPlannerConfig,
    plan_ingest_capacity,
    render_capacity_env,
)
from ingest.embed_pool import EmbedPoolConfig
from ingest.host_profile import DiskProfile, GpuProfile, HostProfile


def _host(
    *,
    cores: int = 16,
    ram_available: int | None = 32768,
    gpu: GpuProfile | None = None,
    disks: tuple[DiskProfile, ...] = (),
) -> HostProfile:
    return HostProfile(
        cpu_model="Test CPU",
        cpu_logical_cores=cores,
        ram_total_mib=65536 if ram_available is not None else None,
        ram_available_mib=ram_available,
        disks=disks,
        gpu=gpu,
        probed_at="2026-07-04T00:00:00+00:00",
    )


def _gpu(name: str = "NVIDIA GeForce RTX 3090", free: int = 20480) -> GpuProfile:
    return GpuProfile(
        name=name,
        total_mib=24576,
        used_mib=24576 - free,
        free_mib=free,
        utilization_pct=5,
    )


POOL_CFG = EmbedPoolConfig(max_instances=8, parallel_per_instance=16)
PLANNER_CFG = CapacityPlannerConfig()


def test_big_host_scales_to_pool_and_cpu_limits() -> None:
    plan = plan_ingest_capacity(
        host=_host(cores=16, ram_available=32768, gpu=_gpu()),
        pool_config=POOL_CFG,
        planner_config=PLANNER_CFG,
    )
    # (20480 - 2048) // 1024 = 18, capped at max_instances=8
    assert plan.embed_pool.instance_count == 8
    assert plan.nomic_pool_parallel == 16  # 3090 is high tier
    assert plan.ingest_embed_concurrency == 8 * 16
    assert plan.ingest_batch_size == 32  # high concurrency -> small batches
    # min(instances=8, cpu 16//2=8, ram (32768-4096)//2048=14, max=8) = 8
    assert plan.ingest_file_concurrency == 8
    assert plan.ingest_chunk_semantic is True


def test_low_bandwidth_gpu_reduces_parallel() -> None:
    plan = plan_ingest_capacity(
        host=_host(gpu=_gpu(name="NVIDIA GeForce RTX 3060", free=8192)),
        pool_config=POOL_CFG,
        planner_config=PLANNER_CFG,
    )
    assert plan.nomic_pool_parallel == 8
    assert plan.ingest_embed_concurrency == plan.embed_pool.instance_count * 8
    assert "3060" in plan.rationale["nomic_pool_parallel"]


def test_few_cores_cap_file_concurrency() -> None:
    plan = plan_ingest_capacity(
        host=_host(cores=4, gpu=_gpu()),
        pool_config=POOL_CFG,
        planner_config=PLANNER_CFG,
    )
    assert plan.ingest_file_concurrency == 2  # 4 cores // chunk_cpu_share 2
    assert "cpu" in plan.rationale["ingest_file_concurrency"]


def test_low_ram_disables_semantic_and_caps_files() -> None:
    plan = plan_ingest_capacity(
        host=_host(cores=16, ram_available=6144, gpu=_gpu()),
        pool_config=POOL_CFG,
        planner_config=PLANNER_CFG,
    )
    # ram cap: (6144 - 4096) // 2048 = 1
    assert plan.ingest_file_concurrency == 1
    assert plan.ingest_chunk_semantic is False
    assert "RAM" in plan.rationale["ingest_chunk_semantic"]


def test_semantic_never_enabled_against_operator_setting() -> None:
    plan = plan_ingest_capacity(
        host=_host(gpu=_gpu()),
        pool_config=POOL_CFG,
        planner_config=PLANNER_CFG,
        semantic_requested=False,
    )
    assert plan.ingest_chunk_semantic is False
    assert "operator" in plan.rationale["ingest_chunk_semantic"]


def test_slow_disk_caps_file_concurrency() -> None:
    slow = DiskProfile(path="/mnt/nas", free_mib=100000, seq_read_mbps=40.0)
    plan = plan_ingest_capacity(
        host=_host(gpu=_gpu(), disks=(slow,)),
        pool_config=POOL_CFG,
        planner_config=PLANNER_CFG,
    )
    assert plan.ingest_file_concurrency == PLANNER_CFG.slow_disk_file_cap
    assert "slow disk" in plan.rationale["ingest_file_concurrency"]


def test_no_gpu_falls_back_conservatively() -> None:
    plan = plan_ingest_capacity(
        host=_host(gpu=None),
        pool_config=POOL_CFG,
        planner_config=PLANNER_CFG,
    )
    assert plan.embed_pool.use_gpu_pool is False
    assert plan.embed_pool.instance_count == 1
    assert plan.ingest_file_concurrency == 1
    # Unknown GPU tier (mid) caps parallel at 12; one instance -> moderate batches.
    assert plan.nomic_pool_parallel == 12
    assert plan.ingest_embed_concurrency == 12
    assert plan.ingest_batch_size == 64


def test_missing_ram_probe_skips_ram_caps() -> None:
    plan = plan_ingest_capacity(
        host=_host(ram_available=None, gpu=_gpu()),
        pool_config=POOL_CFG,
        planner_config=PLANNER_CFG,
    )
    assert plan.ingest_file_concurrency == 8
    assert plan.ingest_chunk_semantic is True


def test_plan_uses_bench_fit_for_chunk_embed_and_batch() -> None:
    bench = BenchFit(
        chunk_concurrency=2,
        chunk_chunks_per_min=15082.0,
        embed_concurrency=48,
        batch_size=32,
        embed_chunks_per_min=2250.0,
        rationale={
            "ingest_chunk_concurrency": "2 from bench knee (15082 chunks/min)",
            "ingest_embed_concurrency": "48 from bench (2250 chunks/min)",
            "ingest_batch_size": "32 from bench",
        },
    )
    plan = plan_ingest_capacity(
        host=_host(cores=16, ram_available=32768, gpu=_gpu()),
        pool_config=POOL_CFG,
        planner_config=PLANNER_CFG,
        bench=bench,
    )
    assert plan.ingest_chunk_concurrency == 2
    assert plan.ingest_embed_concurrency == 48
    assert plan.ingest_batch_size == 32
    assert "bench" in plan.rationale["ingest_chunk_concurrency"]
    assert "bench" in plan.rationale["ingest_embed_concurrency"]


def test_render_capacity_env_includes_all_knobs() -> None:
    plan = plan_ingest_capacity(
        host=_host(gpu=_gpu()),
        pool_config=POOL_CFG,
        planner_config=PLANNER_CFG,
    )
    rendered = render_capacity_env(plan)
    for key in (
        "INGEST_EMBED_URLS=",
        "INGEST_EMBED_CONCURRENCY=",
        "INGEST_FILE_CONCURRENCY=",
        "INGEST_BATCH_SIZE=",
        "INGEST_CHUNK_CONCURRENCY=",
        "INGEST_CHUNK_SEMANTIC=",
        "INGEST_SPARSE_REINDEX=",
        "NOMIC_POOL_PARALLEL=",
        "NOMIC_POOL_INSTANCE_COUNT=",
        "CAPACITY_CPU_CORES=",
        "CAPACITY_GPU_NAME=",
    ):
        assert key in rendered, key
    assert "# rationale" in rendered

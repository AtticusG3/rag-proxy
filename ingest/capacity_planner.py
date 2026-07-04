"""Multi-resource ingest capacity planning.

Combines the VRAM embed pool plan with CPU, RAM, and disk caps into one set of
ingest knobs. Every cap fails open: a missing probe simply skips that dimension.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace

from ingest.bench_fit import BenchFit
from ingest.embed_pool import EmbedPoolConfig, EmbedPoolPlan, load_embed_pool_config, plan_embed_pool
from ingest.gpu_catalog import lookup_gpu_tier
from ingest.host_profile import HostProfile, probe_host
from ingest.port_avoidance import describe_port_skips, loopback_reserved_ports


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    return float(raw)


@dataclass(frozen=True)
class CapacityPlannerConfig:
    ram_reserve_mib: int = 4096
    ram_per_file_mib: int = 2048
    semantic_ram_floor_mib: int = 8192
    semantic_cpu_floor: int = 4
    chunk_cpu_share: int = 2
    max_file_concurrency: int = 8
    min_disk_seq_read_mbps: float = 100.0
    slow_disk_file_cap: int = 2
    sparse_reindex_during_bulk: str = "off"


def load_capacity_planner_config() -> CapacityPlannerConfig:
    return CapacityPlannerConfig(
        ram_reserve_mib=_env_int("INGEST_CAPACITY_RAM_RESERVE_MIB", 4096),
        ram_per_file_mib=_env_int("INGEST_CAPACITY_RAM_PER_FILE_MIB", 2048),
        semantic_ram_floor_mib=_env_int("INGEST_CAPACITY_SEMANTIC_RAM_FLOOR_MIB", 8192),
        semantic_cpu_floor=_env_int("INGEST_CAPACITY_SEMANTIC_CPU_FLOOR", 4),
        chunk_cpu_share=_env_int("INGEST_CAPACITY_CHUNK_CPU_SHARE", 2),
        max_file_concurrency=_env_int("INGEST_CAPACITY_MAX_FILE_CONCURRENCY", 8),
        min_disk_seq_read_mbps=_env_float("INGEST_CAPACITY_MIN_DISK_MBPS", 100.0),
        slow_disk_file_cap=_env_int("INGEST_CAPACITY_SLOW_DISK_FILE_CAP", 2),
        sparse_reindex_during_bulk=os.getenv(
            "INGEST_CAPACITY_SPARSE_REINDEX", "off"
        ).strip().lower()
        or "off",
    )


@dataclass(frozen=True)
class IngestCapacityPlan:
    embed_pool: EmbedPoolPlan
    ingest_file_concurrency: int
    ingest_embed_concurrency: int
    ingest_batch_size: int
    ingest_chunk_concurrency: int
    ingest_chunk_semantic: bool
    ingest_sparse_reindex: str
    nomic_pool_parallel: int
    host: HostProfile
    rationale: dict[str, str]


def _pick_batch_size(embed_concurrency: int) -> int:
    """Higher parallelism favors smaller batches; low parallelism favors big ones."""
    if embed_concurrency >= 16:
        return 32
    if embed_concurrency >= 8:
        return 64
    return 128


def _semantic_enabled(
    host: HostProfile,
    cfg: CapacityPlannerConfig,
    semantic_requested: bool,
) -> bool:
    if not semantic_requested:
        return False
    if host.ram_available_mib is not None and host.ram_available_mib < cfg.semantic_ram_floor_mib:
        return False
    if host.cpu_logical_cores < cfg.semantic_cpu_floor:
        return False
    return True


def _semantic_rationale(
    host: HostProfile,
    cfg: CapacityPlannerConfig,
    semantic_requested: bool,
) -> tuple[bool, str]:
    if not semantic_requested:
        return False, "disabled by operator setting"
    if host.ram_available_mib is not None and host.ram_available_mib < cfg.semantic_ram_floor_mib:
        return (
            False,
            f"disabled: {host.ram_available_mib} MiB RAM below "
            f"{cfg.semantic_ram_floor_mib} MiB floor",
        )
    if host.cpu_logical_cores < cfg.semantic_cpu_floor:
        return (
            False,
            f"disabled: {host.cpu_logical_cores} cores below "
            f"{cfg.semantic_cpu_floor}-core floor",
        )
    return True, "enabled (host above RAM/CPU floors)"


def resolve_semantic_chunking(
    host: HostProfile,
    cfg: CapacityPlannerConfig,
    semantic_requested: bool,
) -> bool:
    """Whether semantic chunking will be enabled after RAM/CPU floors."""
    return _semantic_enabled(host, cfg, semantic_requested)


def plan_ingest_capacity(
    *,
    host: HostProfile | None = None,
    pool_config: EmbedPoolConfig | None = None,
    planner_config: CapacityPlannerConfig | None = None,
    semantic_requested: bool = True,
    data_paths: tuple[str, ...] = (),
    bench: BenchFit | None = None,
) -> IngestCapacityPlan:
    """Compute the full ingest capacity plan from a host snapshot."""
    pool_cfg = pool_config or load_embed_pool_config()
    cfg = planner_config or load_capacity_planner_config()
    if host is None:
        host = probe_host(disk_paths=data_paths, gpu_index=pool_cfg.gpu_index)

    rationale: dict[str, str] = {}

    # GPU tier caps per-instance parallel on low-bandwidth cards.
    tier = lookup_gpu_tier(host.gpu.name if host.gpu else None)
    parallel = min(pool_cfg.parallel_per_instance, tier.parallel_per_instance)
    if parallel < pool_cfg.parallel_per_instance:
        rationale["nomic_pool_parallel"] = (
            f"reduced to {parallel} for {tier.name}-bandwidth GPU "
            f"({host.gpu.name if host.gpu else 'unknown'})"
        )
    else:
        rationale["nomic_pool_parallel"] = f"configured value {parallel}"

    memory = (
        (host.gpu.total_mib, host.gpu.used_mib, host.gpu.free_mib) if host.gpu else None
    )
    embed_pool = plan_embed_pool(
        replace(pool_cfg, parallel_per_instance=parallel),
        memory=memory,
    )
    instances = embed_pool.instance_count
    pool_reason = (
        f"{instances} instance(s) from {embed_pool.gpu_free_mib} MiB free VRAM"
        if embed_pool.use_gpu_pool
        else f"{instances} instance(s), no GPU probe (fallback)"
    )
    skip_note = describe_port_skips(
        requested_base=pool_cfg.port_base,
        ports=embed_pool.ports,
        reserved=loopback_reserved_ports(),
    )
    if skip_note:
        pool_reason = f"{pool_reason}; {skip_note}"
    rationale["embed_pool"] = pool_reason

    semantic, semantic_reason = _semantic_rationale(host, cfg, semantic_requested)
    rationale["ingest_chunk_semantic"] = semantic_reason

    embed_concurrency = embed_pool.ingest_embed_concurrency
    pool_ceiling = instances * parallel

    # File concurrency: min() across every available dimension.
    caps: dict[str, int] = {"embed pool instances": max(1, instances)}
    cpu_cap = max(1, host.cpu_logical_cores // cfg.chunk_cpu_share)
    caps[f"cpu ({host.cpu_logical_cores} cores / {cfg.chunk_cpu_share})"] = cpu_cap
    if host.ram_available_mib is not None:
        ram_cap = max(
            1, (host.ram_available_mib - cfg.ram_reserve_mib) // cfg.ram_per_file_mib
        )
        caps[f"ram ({host.ram_available_mib} MiB available)"] = ram_cap
    slow_disks = [
        disk
        for disk in host.disks
        if disk.seq_read_mbps is not None
        and disk.seq_read_mbps < cfg.min_disk_seq_read_mbps
    ]
    if slow_disks:
        caps[f"slow disk ({slow_disks[0].path})"] = cfg.slow_disk_file_cap
    caps["configured max"] = cfg.max_file_concurrency

    file_concurrency = max(1, min(caps.values()))
    limiting = min(caps, key=lambda key: caps[key])
    rationale["ingest_file_concurrency"] = f"{file_concurrency}, limited by {limiting}"

    if bench and bench.chunk_concurrency is not None:
        chunk_concurrency = max(1, min(bench.chunk_concurrency, file_concurrency))
        if bench.rationale.get("ingest_chunk_concurrency"):
            rationale["ingest_chunk_concurrency"] = (
                f"{chunk_concurrency} ({bench.rationale['ingest_chunk_concurrency']}, "
                f"capped by file concurrency {file_concurrency})"
            )
        else:
            rationale["ingest_chunk_concurrency"] = (
                f"{chunk_concurrency} from bench, capped by file concurrency {file_concurrency}"
            )
    else:
        chunk_concurrency = max(1, min(file_concurrency, cpu_cap))
        rationale["ingest_chunk_concurrency"] = (
            f"{chunk_concurrency} (min of file concurrency and cpu cap)"
        )

    if bench and bench.batch_size is not None:
        batch_size = bench.batch_size
        rationale["ingest_batch_size"] = bench.rationale.get(
            "ingest_batch_size",
            f"{batch_size} from bench",
        )
    else:
        batch_size = _pick_batch_size(embed_concurrency)
        rationale["ingest_batch_size"] = (
            f"{batch_size} for embed concurrency {embed_concurrency}"
        )

    if bench and bench.embed_concurrency is not None:
        embed_concurrency = max(instances, min(bench.embed_concurrency, pool_ceiling))
        if bench.rationale.get("ingest_embed_concurrency"):
            rationale["ingest_embed_concurrency"] = (
                f"{embed_concurrency} ({bench.rationale['ingest_embed_concurrency']}, "
                f"pool ceiling {pool_ceiling})"
            )
        else:
            rationale["ingest_embed_concurrency"] = (
                f"{embed_concurrency} from bench, pool ceiling {pool_ceiling}"
            )
    else:
        rationale["ingest_embed_concurrency"] = (
            f"{embed_concurrency} ({instances} instances x {parallel} parallel)"
        )

    rationale["ingest_sparse_reindex"] = (
        f"{cfg.sparse_reindex_during_bulk} during bulk ingest (rebuild once at end)"
    )

    return IngestCapacityPlan(
        embed_pool=embed_pool,
        ingest_file_concurrency=file_concurrency,
        ingest_embed_concurrency=embed_concurrency,
        ingest_batch_size=batch_size,
        ingest_chunk_concurrency=chunk_concurrency,
        ingest_chunk_semantic=semantic,
        ingest_sparse_reindex=cfg.sparse_reindex_during_bulk,
        nomic_pool_parallel=parallel,
        host=host,
        rationale=rationale,
    )


def render_capacity_env(plan: IngestCapacityPlan) -> str:
    """Render the full capacity plan as an env file (pool keys + ingest knobs)."""
    lines = [
        "# Generated by scale_ingest_capacity.py; do not edit by hand.",
        f"INGEST_EMBED_URLS={plan.embed_pool.ingest_embed_urls}",
        f"INGEST_EMBED_CONCURRENCY={plan.ingest_embed_concurrency}",
        f"INGEST_FILE_CONCURRENCY={plan.ingest_file_concurrency}",
        f"INGEST_BATCH_SIZE={plan.ingest_batch_size}",
        f"INGEST_CHUNK_CONCURRENCY={plan.ingest_chunk_concurrency}",
        f"INGEST_CHUNK_SEMANTIC={'true' if plan.ingest_chunk_semantic else 'false'}",
        f"INGEST_SPARSE_REINDEX={plan.ingest_sparse_reindex}",
        f"NOMIC_POOL_PARALLEL={plan.nomic_pool_parallel}",
        f"NOMIC_POOL_INSTANCE_COUNT={plan.embed_pool.instance_count}",
        f"NOMIC_POOL_PORTS={','.join(str(port) for port in plan.embed_pool.ports)}",
        f"CAPACITY_CPU_CORES={plan.host.cpu_logical_cores}",
    ]
    if plan.host.cpu_model:
        lines.append(f"CAPACITY_CPU_MODEL={plan.host.cpu_model}")
    if plan.host.ram_available_mib is not None:
        lines.append(f"CAPACITY_RAM_AVAILABLE_MIB={plan.host.ram_available_mib}")
    if plan.embed_pool.gpu_free_mib is not None:
        lines.append(f"NOMIC_POOL_GPU_FREE_MIB={plan.embed_pool.gpu_free_mib}")
    if plan.host.gpu and plan.host.gpu.name:
        lines.append(f"CAPACITY_GPU_NAME={plan.host.gpu.name}")
    lines.append(f"CAPACITY_PROBED_AT={plan.host.probed_at}")
    for key, reason in sorted(plan.rationale.items()):
        lines.append(f"# rationale {key}: {reason}")
    return "\n".join(lines) + "\n"

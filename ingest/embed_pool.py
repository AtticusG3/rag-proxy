"""VRAM-aware planning for multi-instance nomic-embed pools."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass

from ingest.port_avoidance import alloc_embed_pool_ports, loopback_reserved_ports


@dataclass(frozen=True)
class EmbedPoolConfig:
    vram_per_instance_mib: int = 1024
    vram_reserve_mib: int = 2048
    port_base: int = 18089
    max_instances: int = 12
    min_instances: int = 1
    parallel_per_instance: int = 16
    gpu_index: int = 0


@dataclass(frozen=True)
class EmbedPoolPlan:
    instance_count: int
    ports: tuple[int, ...]
    ingest_embed_urls: str
    ingest_embed_concurrency: int
    gpu_total_mib: int | None
    gpu_used_mib: int | None
    gpu_free_mib: int | None
    use_gpu_pool: bool


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    return int(raw)


def load_embed_pool_config() -> EmbedPoolConfig:
    return EmbedPoolConfig(
        vram_per_instance_mib=_env_int("NOMIC_POOL_VRAM_PER_INSTANCE_MIB", 1024),
        vram_reserve_mib=_env_int("NOMIC_POOL_VRAM_RESERVE_MIB", 2048),
        port_base=_env_int("NOMIC_POOL_PORT_BASE", 18089),
        max_instances=_env_int("NOMIC_POOL_MAX_INSTANCES", 12),
        min_instances=_env_int("NOMIC_POOL_MIN_INSTANCES", 1),
        parallel_per_instance=_env_int("NOMIC_POOL_PARALLEL_PER_INSTANCE", 16),
        gpu_index=_env_int("NOMIC_POOL_GPU_INDEX", 0),
    )


def query_gpu_memory_mib(gpu_index: int = 0) -> tuple[int, int, int] | None:
    """Return (total, used, free) MiB for a GPU, or None when nvidia-smi is unavailable."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={gpu_index}",
                "--query-gpu=memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None

    line = result.stdout.strip().splitlines()[0]
    parts = [part.strip() for part in line.split(",")]
    if len(parts) != 3:
        return None
    total, used, free = (int(float(part)) for part in parts)
    return total, used, free


def compute_instance_count(
    *,
    gpu_free_mib: int,
    config: EmbedPoolConfig,
) -> int:
    """Derive pool size from free VRAM after reserve."""
    usable = gpu_free_mib - config.vram_reserve_mib
    if usable < config.vram_per_instance_mib:
        return config.min_instances
    count = usable // config.vram_per_instance_mib
    return max(config.min_instances, min(config.max_instances, count))


def pool_plan_from_env(
    values: dict[str, str],
    *,
    parallel: int | None = None,
) -> EmbedPoolPlan | None:
    """Rebuild pool topology from a previously written pool env (no VRAM probe)."""
    urls_raw = values.get("INGEST_EMBED_URLS", "").strip()
    if not urls_raw:
        return None
    urls = tuple(url.strip() for url in urls_raw.split(",") if url.strip())
    if not urls:
        return None

    ports_raw = values.get("NOMIC_POOL_PORTS", "").strip()
    if ports_raw:
        try:
            ports = tuple(int(part.strip()) for part in ports_raw.split(",") if part.strip())
        except ValueError:
            ports = ()
    else:
        ports = ()
    if not ports:
        parsed: list[int] = []
        for url in urls:
            try:
                parsed.append(int(url.rsplit(":", 1)[-1]))
            except ValueError:
                return None
        ports = tuple(parsed)
    if len(ports) != len(urls):
        return None

    try:
        count = int(values.get("NOMIC_POOL_INSTANCE_COUNT", "").strip() or len(ports))
    except ValueError:
        count = len(ports)
    count = len(ports) if count != len(ports) else count

    if parallel is None:
        raw_parallel = values.get("NOMIC_POOL_PARALLEL", "").strip()
        try:
            parallel = int(raw_parallel) if raw_parallel else 8
        except ValueError:
            parallel = 8

    free_raw = values.get("NOMIC_POOL_GPU_FREE_MIB", "").strip()
    try:
        free_mib = int(free_raw) if free_raw else None
    except ValueError:
        free_mib = None

    return EmbedPoolPlan(
        instance_count=count,
        ports=ports,
        ingest_embed_urls=",".join(urls),
        ingest_embed_concurrency=count * max(1, parallel),
        gpu_total_mib=None,
        gpu_used_mib=None,
        gpu_free_mib=free_mib,
        use_gpu_pool=True,
    )


def plan_embed_pool(
    config: EmbedPoolConfig | None = None,
    *,
    memory: tuple[int, int, int] | None | str = "probe",
) -> EmbedPoolPlan:
    """Build an embed pool plan from config and current GPU memory.

    Pass memory=(total, used, free) to reuse an existing probe, or None to force
    the no-GPU fallback.
    """
    cfg = config or load_embed_pool_config()
    if memory == "probe":
        memory = query_gpu_memory_mib(cfg.gpu_index)

    if memory is None:
        reserved = loopback_reserved_ports()
        ports = alloc_embed_pool_ports(
            port_base=cfg.port_base,
            count=cfg.min_instances,
            reserved=reserved,
        )
        urls = ",".join(f"http://127.0.0.1:{port}" for port in ports)
        return EmbedPoolPlan(
            instance_count=len(ports),
            ports=ports,
            ingest_embed_urls=urls,
            ingest_embed_concurrency=len(ports) * cfg.parallel_per_instance,
            gpu_total_mib=None,
            gpu_used_mib=None,
            gpu_free_mib=None,
            use_gpu_pool=False,
        )

    total, used, free = memory
    count = compute_instance_count(gpu_free_mib=free, config=cfg)
    reserved = loopback_reserved_ports()
    ports = alloc_embed_pool_ports(
        port_base=cfg.port_base,
        count=count,
        reserved=reserved,
    )
    urls = ",".join(f"http://127.0.0.1:{port}" for port in ports)
    return EmbedPoolPlan(
        instance_count=len(ports),
        ports=ports,
        ingest_embed_urls=urls,
        ingest_embed_concurrency=len(ports) * cfg.parallel_per_instance,
        gpu_total_mib=total,
        gpu_used_mib=used,
        gpu_free_mib=free,
        use_gpu_pool=True,
    )

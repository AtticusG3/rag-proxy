"""Host resource probing for ingest capacity planning.

Every probe fails open: missing tools or non-Linux hosts yield None fields and the
planner skips the corresponding caps.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("ingest.host_profile")

DISK_BENCH_FILENAME = ".ingest_disk_bench.json"
DISK_BENCH_MAX_AGE_SEC = 7 * 24 * 3600
DISK_BENCH_SIZE_MIB = 32


@dataclass(frozen=True)
class DiskProfile:
    path: str
    free_mib: int | None
    seq_read_mbps: float | None


@dataclass(frozen=True)
class GpuProfile:
    name: str | None
    total_mib: int
    used_mib: int
    free_mib: int
    utilization_pct: int | None


@dataclass(frozen=True)
class HostProfile:
    cpu_model: str | None
    cpu_logical_cores: int
    ram_total_mib: int | None
    ram_available_mib: int | None
    disks: tuple[DiskProfile, ...]
    gpu: GpuProfile | None
    probed_at: str


def read_cpu_model() -> str | None:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    processor = platform.processor()
    return processor or None


def read_meminfo() -> tuple[int, int] | None:
    """Return (total_mib, available_mib) from /proc/meminfo, or None."""
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            values: dict[str, int] = {}
            for line in handle:
                key, _, rest = line.partition(":")
                parts = rest.split()
                if parts and key in ("MemTotal", "MemAvailable"):
                    values[key] = int(parts[0]) // 1024
            if "MemTotal" in values and "MemAvailable" in values:
                return values["MemTotal"], values["MemAvailable"]
    except (OSError, ValueError):
        pass
    return None


def query_gpu_profile(gpu_index: int = 0) -> GpuProfile | None:
    """nvidia-smi probe extending the VRAM query with name and utilization."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={gpu_index}",
                "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None

    line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    parts = [part.strip() for part in line.split(",")]
    if len(parts) != 5:
        return None
    try:
        total, used, free = (int(float(part)) for part in parts[1:4])
        util = int(float(parts[4])) if parts[4] not in ("", "[N/A]") else None
    except ValueError:
        return None
    return GpuProfile(
        name=parts[0] or None,
        total_mib=total,
        used_mib=used,
        free_mib=free,
        utilization_pct=util,
    )


def _bench_cache_path(path: str) -> Path:
    return Path(path) / DISK_BENCH_FILENAME


def _read_cached_bench(path: str) -> float | None:
    cache = _bench_cache_path(path)
    try:
        payload = json.loads(cache.read_text(encoding="utf-8"))
        if time.time() - float(payload["measured_at"]) <= DISK_BENCH_MAX_AGE_SEC:
            return float(payload["seq_read_mbps"])
    except (OSError, KeyError, ValueError):
        pass
    return None


def bench_disk_seq_read(path: str, *, size_mib: int = DISK_BENCH_SIZE_MIB) -> float | None:
    """Measure sequential read MB/s via a temp file; caches result beside the dir.

    OS page cache makes this an optimistic bound, which is fine: the planner only
    uses it to detect very slow storage (network mounts, SD cards).
    """
    if not os.path.isdir(path):
        return None
    cached = _read_cached_bench(path)
    if cached is not None:
        return cached

    block = b"\0" * (1024 * 1024)
    tmp_name = None
    try:
        with tempfile.NamedTemporaryFile(dir=path, delete=False) as handle:
            tmp_name = handle.name
            for _ in range(size_mib):
                handle.write(block)
            handle.flush()
            os.fsync(handle.fileno())
        start = time.perf_counter()
        with open(tmp_name, "rb") as handle:
            while handle.read(1024 * 1024):
                pass
        elapsed = time.perf_counter() - start
    except OSError as exc:
        log.warning("disk bench failed for %s: %s", path, exc)
        return None
    finally:
        if tmp_name:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass

    if elapsed <= 0:
        return None
    mbps = round(size_mib / elapsed, 1)
    try:
        _bench_cache_path(path).write_text(
            json.dumps({"seq_read_mbps": mbps, "measured_at": time.time()}),
            encoding="utf-8",
        )
    except OSError:
        pass
    return mbps


def probe_disk(path: str, *, bench: bool = False) -> DiskProfile:
    free_mib: int | None = None
    try:
        free_mib = shutil.disk_usage(path).free // (1024 * 1024)
    except OSError:
        pass
    seq_read = bench_disk_seq_read(path) if bench else _read_cached_bench(path)
    return DiskProfile(path=path, free_mib=free_mib, seq_read_mbps=seq_read)


def probe_host(
    *,
    disk_paths: tuple[str, ...] = (),
    gpu_index: int = 0,
    bench_disks: bool = False,
) -> HostProfile:
    """Snapshot host resources; every field degrades to None on probe failure."""
    mem = read_meminfo()
    return HostProfile(
        cpu_model=read_cpu_model(),
        cpu_logical_cores=os.cpu_count() or 1,
        ram_total_mib=mem[0] if mem else None,
        ram_available_mib=mem[1] if mem else None,
        disks=tuple(probe_disk(path, bench=bench_disks) for path in disk_paths),
        gpu=query_gpu_profile(gpu_index),
        probed_at=datetime.now(timezone.utc).isoformat(),
    )

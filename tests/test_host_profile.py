"""Tests for host resource probing used by the ingest capacity planner."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

from ingest.gpu_catalog import DEFAULT_TIER, HIGH_TIER, LOW_TIER, lookup_gpu_tier
from ingest.host_profile import (
    DISK_BENCH_FILENAME,
    GpuProfile,
    probe_disk,
    probe_host,
    query_gpu_profile,
    read_meminfo,
)


def test_read_meminfo_parses_proc(tmp_path: Path, monkeypatch) -> None:
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(
        "MemTotal:       32768000 kB\n"
        "MemFree:         1000000 kB\n"
        "MemAvailable:   16384000 kB\n",
        encoding="utf-8",
    )
    real_open = open

    def fake_open(path, *args, **kwargs):
        if path == "/proc/meminfo":
            return real_open(meminfo, *args, **kwargs)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", fake_open)
    result = read_meminfo()
    assert result == (32000, 16000)


def test_read_meminfo_fails_open_without_proc(monkeypatch) -> None:
    def raise_oserror(*args, **kwargs):
        raise OSError("no proc")

    monkeypatch.setattr("builtins.open", raise_oserror)
    assert read_meminfo() is None


@patch("ingest.host_profile.subprocess.run")
def test_query_gpu_profile_parses_nvidia_smi(mock_run: MagicMock) -> None:
    mock_run.return_value = MagicMock(
        stdout="NVIDIA GeForce RTX 3090, 24576, 4096, 20480, 37\n"
    )
    profile = query_gpu_profile(0)
    assert profile == GpuProfile(
        name="NVIDIA GeForce RTX 3090",
        total_mib=24576,
        used_mib=4096,
        free_mib=20480,
        utilization_pct=37,
    )


@patch("ingest.host_profile.subprocess.run", side_effect=OSError("missing"))
def test_query_gpu_profile_fails_open(mock_run: MagicMock) -> None:
    assert query_gpu_profile(0) is None


def test_probe_disk_uses_cached_bench(tmp_path: Path) -> None:
    cache = tmp_path / DISK_BENCH_FILENAME
    cache.write_text(
        json.dumps({"seq_read_mbps": 512.5, "measured_at": time.time()}),
        encoding="utf-8",
    )
    profile = probe_disk(str(tmp_path))
    assert profile.seq_read_mbps == 512.5
    assert profile.free_mib is not None and profile.free_mib > 0


def test_probe_disk_ignores_stale_cache(tmp_path: Path) -> None:
    cache = tmp_path / DISK_BENCH_FILENAME
    cache.write_text(
        json.dumps({"seq_read_mbps": 512.5, "measured_at": time.time() - 30 * 24 * 3600}),
        encoding="utf-8",
    )
    profile = probe_disk(str(tmp_path))
    assert profile.seq_read_mbps is None


def test_probe_disk_bench_writes_cache(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("ingest.host_profile.DISK_BENCH_SIZE_MIB", 1)
    profile = probe_disk(str(tmp_path), bench=True)
    assert profile.seq_read_mbps is not None and profile.seq_read_mbps > 0
    assert (tmp_path / DISK_BENCH_FILENAME).is_file()
    # No leftover temp bench file besides the cache.
    leftovers = [p for p in tmp_path.iterdir() if p.name != DISK_BENCH_FILENAME]
    assert leftovers == []


@patch("ingest.host_profile.query_gpu_profile", return_value=None)
@patch("ingest.host_profile.read_meminfo", return_value=None)
def test_probe_host_fails_open(mock_mem: MagicMock, mock_gpu: MagicMock) -> None:
    profile = probe_host(disk_paths=("/nonexistent/dir",))
    assert profile.cpu_logical_cores >= 1
    assert profile.ram_total_mib is None
    assert profile.gpu is None
    assert profile.disks[0].free_mib is None
    assert profile.disks[0].seq_read_mbps is None


def test_lookup_gpu_tier_buckets() -> None:
    assert lookup_gpu_tier("NVIDIA GeForce RTX 3090") is HIGH_TIER
    assert lookup_gpu_tier("NVIDIA GeForce RTX 3060") is LOW_TIER
    assert lookup_gpu_tier("Some Unknown GPU") is DEFAULT_TIER
    assert lookup_gpu_tier(None) is DEFAULT_TIER

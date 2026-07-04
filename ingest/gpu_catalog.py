"""Static GPU name to bandwidth tier lookup for capacity planning."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GpuTier:
    name: str
    parallel_per_instance: int


HIGH_TIER = GpuTier(name="high", parallel_per_instance=16)
MID_TIER = GpuTier(name="mid", parallel_per_instance=12)
LOW_TIER = GpuTier(name="low", parallel_per_instance=8)

# Case-insensitive substring patterns checked in order; first match wins.
# Coarse buckets by memory bandwidth class, extend as new hardware appears.
_TIER_PATTERNS: tuple[tuple[str, GpuTier], ...] = (
    # Datacenter / workstation, HBM or wide GDDR6X buses
    ("h100", HIGH_TIER),
    ("h200", HIGH_TIER),
    ("a100", HIGH_TIER),
    ("v100", HIGH_TIER),
    ("a6000", HIGH_TIER),
    ("l40", HIGH_TIER),
    # Consumer high end (>=~700 GB/s)
    ("5090", HIGH_TIER),
    ("4090", HIGH_TIER),
    ("3090", HIGH_TIER),
    ("5080", HIGH_TIER),
    ("4080", HIGH_TIER),
    ("3080", HIGH_TIER),
    # Consumer mid (~400-600 GB/s)
    ("5070", MID_TIER),
    ("4070", MID_TIER),
    ("3070", MID_TIER),
    ("2080", MID_TIER),
    ("a4000", MID_TIER),
    # Laptop / narrow-bus workstation cards
    ("a2000", LOW_TIER),
    # Narrow-bus / entry cards
    ("4060", LOW_TIER),
    ("3060", LOW_TIER),
    ("3050", LOW_TIER),
    ("2060", LOW_TIER),
    ("1660", LOW_TIER),
    ("1650", LOW_TIER),
    ("t4", LOW_TIER),
)

DEFAULT_TIER = MID_TIER


def lookup_gpu_tier(gpu_name: str | None) -> GpuTier:
    """Return the bandwidth tier for a GPU name; unknown or missing names get MID."""
    if not gpu_name:
        return DEFAULT_TIER
    lowered = gpu_name.lower()
    for pattern, tier in _TIER_PATTERNS:
        if pattern in lowered:
            return tier
    return DEFAULT_TIER

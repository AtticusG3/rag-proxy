"""Tests for VRAM-aware embed pool planning."""

from __future__ import annotations

from ingest.embed_pool import EmbedPoolConfig, compute_instance_count, plan_embed_pool


def test_compute_instance_count_respects_reserve_and_cap():
  cfg = EmbedPoolConfig(
    vram_per_instance_mib=1024,
    vram_reserve_mib=2048,
    max_instances=8,
    min_instances=1,
  )
  assert compute_instance_count(gpu_free_mib=32768, config=cfg) == 8
  assert compute_instance_count(gpu_free_mib=4096, config=cfg) == 2
  assert compute_instance_count(gpu_free_mib=2500, config=cfg) == 1


def test_plan_embed_pool_without_gpu_falls_back_to_single_port(monkeypatch):
  monkeypatch.delenv("NOMIC_POOL_PORT_BASE", raising=False)

  def _no_gpu(_index: int = 0):
    return None

  monkeypatch.setattr("ingest.embed_pool.query_gpu_memory_mib", _no_gpu)
  plan = plan_embed_pool(EmbedPoolConfig(port_base=18089, min_instances=1))
  assert plan.instance_count == 1
  assert plan.ports == (18089,)
  assert plan.use_gpu_pool is False
  assert "http://127.0.0.1:18089" in plan.ingest_embed_urls


def test_plan_embed_pool_scales_with_free_vram(monkeypatch):
  monkeypatch.setenv("NOMIC_POOL_VRAM_RESERVE_MIB", "2048")
  monkeypatch.setenv("NOMIC_POOL_VRAM_PER_INSTANCE_MIB", "1024")
  monkeypatch.setenv("NOMIC_POOL_MAX_INSTANCES", "10")

  monkeypatch.setattr(
    "ingest.embed_pool.query_gpu_memory_mib",
    lambda _index=0: (32768, 12000, 20768),
  )
  plan = plan_embed_pool()
  assert plan.instance_count == 10
  assert plan.ports[0] == 18089
  assert plan.ports[-1] == 18098
  assert plan.ingest_embed_concurrency == 10 * 32

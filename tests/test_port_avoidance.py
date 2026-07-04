"""Tests for loopback port conflict avoidance when planning embed pools."""

from __future__ import annotations

from pathlib import Path

from ingest.embed_pool import EmbedPoolConfig, plan_embed_pool
from ingest.port_avoidance import (
    alloc_embed_pool_ports,
    describe_port_skips,
    embed_pool_stop_ports,
    is_loopback_host,
    loopback_reserved_ports,
    merge_config_dir_env,
    port_from_url,
    ports_from_embed_urls,
)


def test_is_loopback_host():
    assert is_loopback_host("127.0.0.1")
    assert is_loopback_host("localhost")
    assert is_loopback_host("127.0.0.2")
    assert not is_loopback_host("192.168.1.36")
    assert not is_loopback_host(None)


def test_port_from_url_ignores_remote_hosts():
    assert port_from_url("http://192.168.1.36:6333") is None
    assert port_from_url("http://127.0.0.1:6333") == 6333
    assert port_from_url("http://localhost:8096/reindex") == 8096


def test_loopback_reserved_ports_uses_homelab_defaults(monkeypatch):
    for key in (
        "EMBED_URL",
        "QDRANT_URL",
        "RERANKER_URL",
        "LLAMA_SWAP_URL",
        "RAG_PROXY_URL",
        "SPARSE_INDEX_URL",
        "PROXY_PORT",
        "ADMIN_PORT",
        "INGEST_EMBED_URLS",
    ):
        monkeypatch.delenv(key, raising=False)

    reserved = loopback_reserved_ports(include_defaults=True)
    assert reserved == frozenset({8089, 6333, 8095, 8080, 8081, 8088, 8087})


def test_loopback_reserved_ports_includes_sparse_when_set(monkeypatch):
    monkeypatch.setenv("SPARSE_INDEX_URL", "http://127.0.0.1:8096")
    reserved = loopback_reserved_ports(include_defaults=False)
    assert 8096 in reserved


def test_loopback_reserved_ports_ignores_remote_sparse(monkeypatch):
    monkeypatch.setenv("SPARSE_INDEX_URL", "http://sparse-index:8096")
    reserved = loopback_reserved_ports(include_defaults=False)
    assert 8096 not in reserved


def test_ports_from_embed_urls():
    raw = "http://127.0.0.1:18089,http://127.0.0.1:18090,http://10.0.0.5:9000"
    assert ports_from_embed_urls(raw) == {18089, 18090}


def test_alloc_embed_pool_ports_skips_reserved():
    reserved = frozenset({8095, 8096})
    ports = alloc_embed_pool_ports(port_base=8094, count=3, reserved=reserved)
    assert ports == (8094, 8097, 8098)


def test_embed_pool_stop_ports_covers_skips():
    reserved = frozenset({8095})
    stop_ports = embed_pool_stop_ports(18089, 8, extra=4, reserved=reserved)
    assert 18089 in stop_ports
    assert 18089 + 8 + 4 + 1 in stop_ports


def test_describe_port_skips_when_base_blocked():
    reserved = frozenset({18089})
    ports = (18090, 18091)
    note = describe_port_skips(requested_base=18089, ports=ports, reserved=reserved)
    assert note is not None
    assert "18090" in note
    assert "18089" in note


def test_plan_embed_pool_skips_loopback_conflict(monkeypatch):
    monkeypatch.setenv("NOMIC_POOL_PORT_BASE", "8094")
    monkeypatch.setenv("SPARSE_INDEX_URL", "http://127.0.0.1:8096")
    monkeypatch.setenv("RERANKER_URL", "http://127.0.0.1:8095")

    monkeypatch.setattr(
        "ingest.embed_pool.query_gpu_memory_mib",
        lambda _index=0: (32768, 8000, 24768),
    )
    plan = plan_embed_pool(
        EmbedPoolConfig(port_base=8094, max_instances=4, min_instances=1),
        memory=(32768, 8000, 24768),
    )
    assert 8095 not in plan.ports
    assert 8096 not in plan.ports
    assert len(plan.ports) == 4


def test_merge_config_dir_env_overlays_files(tmp_path: Path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "rag-proxy.env").write_text("QDRANT_URL=http://127.0.0.1:6333\n", encoding="utf-8")
    (config_dir / "rag-admin.env").write_text(
        "SPARSE_INDEX_URL=http://127.0.0.1:8096\n", encoding="utf-8"
    )
    scale_env = config_dir / "nomic-embed-scale.env"
    scale_env.write_text("NOMIC_POOL_PORT_BASE=18089\n", encoding="utf-8")

    merged = merge_config_dir_env(config_dir, scale_env)
    assert merged["NOMIC_POOL_PORT_BASE"] == "18089"
    assert merged["QDRANT_URL"] == "http://127.0.0.1:6333"
    assert merged["SPARSE_INDEX_URL"] == "http://127.0.0.1:8096"
    reserved = loopback_reserved_ports(merged, include_defaults=False)
    assert {6333, 8096}.issubset(reserved)

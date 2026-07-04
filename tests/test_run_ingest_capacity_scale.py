"""Tests for run_ingest_capacity_scale.py orchestration."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_ingest_capacity_scale.py"


def _load():
    spec = importlib.util.spec_from_file_location("run_ingest_capacity_scale", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_read_pool_urls(tmp_path: Path) -> None:
    mod = _load()
    pool = tmp_path / "pool.env"
    pool.write_text(
        "INGEST_EMBED_URLS=http://127.0.0.1:18090,http://127.0.0.1:18091\n",
        encoding="utf-8",
    )
    assert "18090" in mod._read_pool_urls(pool)


@patch.object(sys, "argv", ["run_ingest_capacity_scale.py", "--skip-bench"])
def test_main_skip_bench_runs_apply_twice(tmp_path: Path, monkeypatch) -> None:
    mod = _load()
    pool_env = tmp_path / "pool.env"
    scale_env = tmp_path / "scale.env"
    scale_env.write_text("NOMIC_POOL_PORT_BASE=18089\n", encoding="utf-8")
    out_dir = tmp_path / "out"

    calls: list[str] = []

    monkeypatch.setattr(mod, "stop_embed_stack", lambda _env: calls.append("stop"))
    monkeypatch.setattr(mod, "wait_gpu_clear", lambda **kwargs: calls.append("wait"))
    monkeypatch.setattr(
        mod,
        "run_scale_apply",
        lambda **kwargs: calls.append("write" if kwargs.get("write_env_only") else "apply") or 0,
    )
    monkeypatch.setattr(mod, "restart_query_embed", lambda: calls.append("query"))

    monkeypatch.setenv("CONFIG_DIR", str(tmp_path))
    with patch.object(
        mod.argparse.ArgumentParser,
        "parse_args",
        return_value=MagicMock(
            pool_env=str(pool_env),
            scale_env=str(scale_env),
            out_dir=str(out_dir),
            skip_bench=True,
            semantic_requested=None,
        ),
    ):
        assert mod.main() == 0

    assert calls == ["stop", "wait", "apply", "write", "query"]

"""Tests for run_ingest_capacity_scale exit semantics."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_ingest_capacity_scale.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_ingest_capacity_scale", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_main_exits_zero_when_benches_warn_but_plan_applies(tmp_path: Path) -> None:
    mod = _load_module()
    pool_env = tmp_path / "pool.env"
    scale_env = tmp_path / "scale.env"
    pool_env.write_text("INGEST_EMBED_URLS=http://127.0.0.1:18089\n", encoding="utf-8")
    scale_env.write_text("", encoding="utf-8")
    out_dir = tmp_path / "out"

    with patch.object(mod, "apply_config_env"):
        with patch.object(mod, "stop_embed_stack"):
            with patch.object(mod, "wait_gpu_clear"):
                with patch.object(mod, "run_chunk_bench", return_value=1):
                    with patch.object(mod, "run_scale_apply", return_value=0):
                        with patch.object(mod, "run_embed_bench", return_value=1):
                            with patch.object(mod, "restart_query_embed"):
                                with patch.object(
                                    sys,
                                    "argv",
                                    [
                                        "run_ingest_capacity_scale.py",
                                        "--pool-env",
                                        str(pool_env),
                                        "--scale-env",
                                        str(scale_env),
                                        "--out-dir",
                                        str(out_dir),
                                    ],
                                ):
                                    assert mod.main() == 0


def test_main_exits_nonzero_when_apply_fails(tmp_path: Path) -> None:
    mod = _load_module()
    pool_env = tmp_path / "pool.env"
    scale_env = tmp_path / "scale.env"
    out_dir = tmp_path / "out"

    with patch.object(mod, "apply_config_env"):
        with patch.object(mod, "stop_embed_stack"):
            with patch.object(mod, "wait_gpu_clear"):
                with patch.object(mod, "run_chunk_bench", return_value=0):
                    with patch.object(mod, "run_scale_apply", return_value=1):
                        with patch.object(mod, "restart_query_embed"):
                            with patch.object(
                                sys,
                                "argv",
                                [
                                    "run_ingest_capacity_scale.py",
                                    "--pool-env",
                                    str(pool_env),
                                    "--scale-env",
                                    str(scale_env),
                                    "--out-dir",
                                    str(out_dir),
                                    "--skip-bench",
                                ],
                            ):
                                assert mod.main() == 1

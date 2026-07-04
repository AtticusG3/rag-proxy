"""Tests for embed pool lifecycle helpers in scale_ingest_capacity.py."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "scale_ingest_capacity.py"


def _load_scale_module():
    spec = importlib.util.spec_from_file_location("scale_ingest_capacity", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_retire_pool_units_disables_everything_outside_keep_set() -> None:
    scale = _load_scale_module()
    config = scale.EmbedPoolConfig(port_base=18089, max_instances=4)
    calls: list[tuple[str, ...]] = []

    def fake_systemctl(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        if args[:2] == ("list-units", "--all"):
            stdout = (
                "nomic-embed@18089.service loaded active running\n"
                "nomic-embed@18095.service loaded activating auto-restart\n"
            )
            return subprocess.CompletedProcess(args, 0, stdout, "")
        return subprocess.CompletedProcess(args, 0, "", "")

    with patch.object(scale, "_systemctl", side_effect=fake_systemctl):
        scale._retire_pool_units(keep_ports={18090, 18091}, config=config)

    disabled = {
        int(args[1].split("@", 1)[1].removesuffix(".service"))
        for args in calls
        if args and args[0] == "disable"
    }
    assert disabled >= {18089, 18092, 18093, 18095}


def test_systemctl_argv_uses_sudo_wrapper_for_mutating_commands() -> None:
    scale = _load_scale_module()
    with patch.object(scale, "_running_as_root", return_value=False):
        with patch.object(scale, "_pool_systemctl_wrapper_installed", return_value=True):
            argv = scale._systemctl_argv("disable", "nomic-embed@18095.service")
    assert argv == [
        "sudo",
        "-n",
        str(scale.POOL_SYSTEMCTL_WRAPPER),
        "disable",
        "nomic-embed@18095.service",
    ]


def test_systemctl_argv_skips_sudo_for_read_only_commands() -> None:
    scale = _load_scale_module()
    with patch.object(scale, "_running_as_root", return_value=False):
        argv = scale._systemctl_argv("show", "nomic-embed@18089.service", "-p", "MainPID")
    assert argv == ["systemctl", "show", "nomic-embed@18089.service", "-p", "MainPID"]


def test_is_embed_llama_requires_embedding_flag() -> None:
    scale = _load_scale_module()
    with patch.object(
        scale,
        "_process_cmdline",
        return_value="/opt/ai/bin/llama-server -m embed.gguf --embedding --port 18089",
    ):
        assert scale._is_embed_llama(123) is True
    with patch.object(
        scale,
        "_process_cmdline",
        return_value="/opt/ai/bin/llama-server -m chat.gguf --port 8080",
    ):
        assert scale._is_embed_llama(456) is False


def test_kill_stray_gpu_embeds_only_targets_embedding_processes() -> None:
    scale = _load_scale_module()
    killed: list[int] = []

    def fake_kill(pid: int, sig: int) -> None:
        killed.append(pid)

    with patch.object(scale, "_query_gpu_llama_pids", return_value={100, 200, 300}):
        with patch.object(scale, "_expected_embed_pids", return_value={100}):
            with patch.object(scale, "_is_embed_llama", side_effect=lambda pid: pid != 200):
                with patch.object(scale.os, "kill", side_effect=fake_kill):
                    result = scale._kill_stray_gpu_embeds({18090})

    assert result == [300]
    assert killed == [300]


def test_finalize_pool_units_disables_unhealthy_planned_ports() -> None:
    scale = _load_scale_module()
    config = scale.EmbedPoolConfig(port_base=18089, max_instances=8)
    calls: list[tuple[str, ...]] = []

    def fake_systemctl(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        if args[:2] == ("list-units", "--all"):
            return subprocess.CompletedProcess(args, 0, "", "")
        return subprocess.CompletedProcess(args, 0, "", "")

    with patch.object(scale, "_systemctl", side_effect=fake_systemctl):
        with patch.object(scale, "_kill_stray_gpu_embeds") as kill_mock:
            scale._finalize_pool_units(
                planned_ports={18089, 18090, 18091},
                healthy_ports={18090},
                config=config,
            )

    disabled = {
        int(args[1].split("@", 1)[1].removesuffix(".service"))
        for args in calls
        if args and args[0] == "disable"
    }
    assert 18089 in disabled
    assert 18091 in disabled
    assert 18090 not in disabled
    kill_mock.assert_called_once_with({18090})


def test_write_pool_env_uses_sudo_when_target_not_writable(tmp_path: Path) -> None:
    scale = _load_scale_module()
    target = tmp_path / "pool.env"
    plan = MagicMock()
    plan_text = "INGEST_EMBED_URLS=http://127.0.0.1:18089\n"

    real_write = scale.Path.write_text

    def write_text(self, *args, **kwargs):
        if self == target:
            raise PermissionError("denied")
        return real_write(self, *args, **kwargs)

    with patch.object(scale, "render_capacity_env", return_value=plan_text):
        with patch.object(scale.Path, "write_text", write_text):
            with patch.object(
                scale,
                "_run",
                side_effect=[
                    subprocess.CompletedProcess(["sudo"], 0, "", ""),
                    subprocess.CompletedProcess(["sudo"], 0, "", ""),
                ],
            ) as run_mock:
                scale._write_pool_env(str(target), plan)

    staging = Path(tempfile.gettempdir()) / f"nomic-embed-pool.{scale._staging_suffix()}.env"
    assert staging.read_text(encoding="utf-8") == plan_text
    assert run_mock.call_args_list[0].args[0][:3] == ["sudo", "-n", "cp"]

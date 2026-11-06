"""Sidecar lifecycle should stop sparse during ingest and wait on restart."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ingest import sidecar_lifecycle as sl

ROOT = Path(__file__).resolve().parents[1]
SIDECAR_HOST_ENV = [
    ("turbovec", "TURBOVEC_HOST"),
    ("sparse", "SPARSE_HOST"),
    ("rerank", "RERANK_HOST"),
]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop(name, None)
    return mod


def _sidecar_host(sidecar_dir: str) -> str:
    """Import a sidecar app fresh (HOST is read at import) and return its bind host."""
    prior_core = sys.modules.get("core")
    sys.modules["core"] = _load_module(
        f"{sidecar_dir}_core_host_test", ROOT / "sidecars" / sidecar_dir / "core.py"
    )
    try:
        app = _load_module(
            f"{sidecar_dir}_app_host_test", ROOT / "sidecars" / sidecar_dir / "app.py"
        )
    finally:
        if prior_core is None:
            sys.modules.pop("core", None)
        else:
            sys.modules["core"] = prior_core
    return app.HOST


@pytest.mark.parametrize(("sidecar_dir", "env_var"), SIDECAR_HOST_ENV)
def test_sidecar_default_host_is_loopback(monkeypatch, sidecar_dir, env_var) -> None:
    monkeypatch.delenv(env_var, raising=False)
    assert _sidecar_host(sidecar_dir) == "127.0.0.1"


@pytest.mark.parametrize(("sidecar_dir", "env_var"), SIDECAR_HOST_ENV)
def test_sidecar_host_override_from_env(monkeypatch, sidecar_dir, env_var) -> None:
    monkeypatch.setenv(env_var, "0.0.0.0")
    assert _sidecar_host(sidecar_dir) == "0.0.0.0"


def test_ensure_sparse_starts_unit_and_waits(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr(sl, "sidecar_on_demand_enabled", lambda: True)
    monkeypatch.setattr(sl, "_unit_active", lambda _u: False)
    monkeypatch.setattr(sl, "_start_unit", lambda u: calls.append(f"start:{u}"))
    monkeypatch.setattr(sl, "wait_for_sidecar_health", lambda url, **kw: True)

    assert sl.ensure_sparse_sidecar("http://127.0.0.1:18096") is True
    assert calls == ["start:sparse-sidecar.service"]


def test_stop_sparse_when_active(monkeypatch) -> None:
    stopped: list[str] = []
    unit_active = {"v": True}

    monkeypatch.setattr(sl, "sidecar_on_demand_enabled", lambda: True)
    monkeypatch.setattr(sl, "_unit_active", lambda _u: unit_active["v"])
    monkeypatch.setattr(sl, "probe_sidecar_health", lambda url, **kw: False)

    def stop(u: str) -> None:
        stopped.append(u)
        unit_active["v"] = False

    monkeypatch.setattr(sl, "_stop_unit", stop)
    monkeypatch.setattr(sl, "_stop_sidecar_process", lambda _p: True)

    assert sl.stop_sparse_sidecar() is True
    assert stopped == ["sparse-sidecar.service"]


def test_stop_unit_stops_without_disable(monkeypatch) -> None:
    """Stopping for ingest must not disable the unit (breaks idle restart/boot)."""
    calls: list[tuple] = []

    def fake_systemctl(*args, check=False):
        calls.append(args)
        return MagicMock(stdout="", returncode=0)

    monkeypatch.setattr(sl, "_systemctl", fake_systemctl)
    sl._stop_unit("sparse-sidecar.service")
    assert ("stop", "sparse-sidecar.service") in calls
    assert not any(c and c[0] == "disable" for c in calls)

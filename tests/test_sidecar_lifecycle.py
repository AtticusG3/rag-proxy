"""Sidecar lifecycle should stop sparse during ingest and wait on restart."""

from __future__ import annotations

from unittest.mock import MagicMock

from ingest import sidecar_lifecycle as sl


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

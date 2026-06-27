"""Tests for delayed service restart helper."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

from rag_admin.service_restart import schedule_restart


def test_schedule_restart_rejects_empty_command() -> None:
    ok, msg = schedule_restart("   ")
    assert ok is False
    assert "not configured" in msg.lower()


def test_schedule_restart_runs_command_after_delay() -> None:
    ran = threading.Event()
    mock_run = MagicMock(side_effect=lambda *a, **k: ran.set())

    with patch("rag_admin.service_restart.time.sleep", return_value=None):
        with patch("rag_admin.service_restart.subprocess.run", mock_run):
            ok, msg = schedule_restart("systemctl restart rag-proxy", delay_sec=0.01)
    assert ok is True
    assert "systemctl restart rag-proxy" in msg

    deadline = time.time() + 2.0
    while not ran.is_set() and time.time() < deadline:
        time.sleep(0.01)
    assert ran.is_set()
    mock_run.assert_called_once()
    assert mock_run.call_args.kwargs.get("shell") is True

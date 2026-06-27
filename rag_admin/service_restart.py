"""Restart systemd services after settings changes."""

from __future__ import annotations

import logging
import subprocess
import threading
import time

log = logging.getLogger("rag-admin.restart")


def schedule_restart(command: str, *, delay_sec: float = 1.5) -> tuple[bool, str]:
    """Run a shell restart command after a short delay (returns before it runs)."""
    command = command.strip()
    if not command:
        return False, "Restart command is not configured (set RAG_PROXY_RESTART_CMD or RAG_ADMIN_RESTART_CMD)."

    def _worker() -> None:
        time.sleep(delay_sec)
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            if result.returncode != 0:
                log.warning(
                    "restart command failed rc=%s cmd=%r stderr=%s",
                    result.returncode,
                    command,
                    (result.stderr or result.stdout or "").strip(),
                )
            else:
                log.info("restart command succeeded: %s", command)
        except Exception as exc:
            log.warning("restart command error cmd=%r: %s", command, exc)

    thread = threading.Thread(target=_worker, daemon=True, name="rag-admin-restart")
    thread.start()
    return True, f"Scheduled: {command}"

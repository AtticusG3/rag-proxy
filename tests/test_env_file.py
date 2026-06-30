"""Tests for env file read/write."""

from __future__ import annotations

import stat
import sys
from pathlib import Path

from rag_admin.env_file import read_env_file, write_env_file


def test_write_env_file_updates_existing_keys(tmp_path: Path) -> None:
    env_path = tmp_path / "rag-admin.env"
    env_path.write_text("FOO=1\n# comment\nBAR=old\n", encoding="utf-8")
    write_env_file(str(env_path), {"BAR": "new", "BAZ": "3"})
    values = read_env_file(str(env_path))
    assert values["FOO"] == "1"
    assert values["BAR"] == "new"
    assert values["BAZ"] == "3"
    text = env_path.read_text(encoding="utf-8")
    assert "# comment" in text


def test_write_env_file_sets_restrictive_permissions(tmp_path: Path) -> None:
    env_path = tmp_path / "rag-admin.env"
    write_env_file(str(env_path), {"SECRET": "value"})
    mode = stat.S_IMODE(env_path.stat().st_mode)
    if sys.platform == "win32":
        assert env_path.is_file()
    else:
        assert mode == 0o600

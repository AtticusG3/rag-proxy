"""Tests for env file read/write."""

from __future__ import annotations

import stat
import sys
from pathlib import Path

import pytest

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


def test_write_env_file_rejects_newline_and_does_not_create_file(tmp_path: Path) -> None:
    env_path = tmp_path / "rag-admin.env"
    with pytest.raises(ValueError):
        write_env_file(
            str(env_path),
            {"MODEL_ROUTES_JSON": '{\n "a": "b"\n}\nPROXY_INTERNAL_TOKEN='},
        )
    assert not env_path.exists()


@pytest.mark.parametrize("bad_char", ["\n", "\r", "\x00"], ids=["newline", "carriage_return_newline", "nul"])
def test_write_env_file_rejects_line_break_values(tmp_path: Path, bad_char: str) -> None:
    env_path = tmp_path / "rag-admin.env"
    original = "FOO=1\nBAR=2\n"
    env_path.write_text(original, encoding="utf-8")
    with pytest.raises(ValueError):
        write_env_file(str(env_path), {"FOO": f"a{bad_char}b"})
    assert env_path.read_text(encoding="utf-8") == original


@pytest.mark.parametrize("bad_char", ["\n", "\r", "\x00"], ids=["newline", "carriage_return_newline", "nul"])
def test_write_env_file_rejects_line_break_in_appended_key(tmp_path: Path, bad_char: str) -> None:
    env_path = tmp_path / "rag-admin.env"
    env_path.write_text("FOO=1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        write_env_file(str(env_path), {"NEW_KEY": f"value{bad_char}INJECTED=1"})
    assert env_path.read_text(encoding="utf-8") == "FOO=1\n"


def test_write_env_file_rejects_before_writing_any_file(tmp_path: Path) -> None:
    admin_env = tmp_path / "rag-admin.env"
    admin_env.write_text("FOO=1\n", encoding="utf-8")
    proxy_env = tmp_path / "rag-proxy.env"
    proxy_env.write_text("BAR=2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        write_env_file(str(admin_env), {"GOOD": "ok", "BAD": "one\ntwo"})
    assert admin_env.read_text(encoding="utf-8") == "FOO=1\n"
    assert proxy_env.read_text(encoding="utf-8") == "BAR=2\n"

"""Read and update key=value environment files."""

from __future__ import annotations

import os
import re
from pathlib import Path

_ENV_LINE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")


def read_env_file(path: str) -> dict[str, str]:
    """Parse KEY=VALUE lines; ignore comments and blanks."""
    file_path = Path(path)
    if not file_path.is_file():
        return {}
    values: dict[str, str] = {}
    for raw_line in file_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _ENV_LINE.match(line)
        if match:
            values[match.group(1)] = match.group(2)
    return values


def write_env_file(path: str, updates: dict[str, str], *, create: bool = True) -> None:
    """Merge updates into an env file and atomically replace it."""
    file_path = Path(path)
    existing = read_env_file(path) if file_path.is_file() else {}
    merged = {**existing, **updates}

    if file_path.is_file():
        lines = file_path.read_text(encoding="utf-8").splitlines()
        seen: set[str] = set()
        out_lines: list[str] = []
        for raw_line in lines:
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                out_lines.append(raw_line)
                continue
            match = _ENV_LINE.match(stripped)
            if not match:
                out_lines.append(raw_line)
                continue
            key = match.group(1)
            if key in updates:
                out_lines.append(f"{key}={updates[key]}")
                seen.add(key)
            else:
                out_lines.append(raw_line)
        for key, value in updates.items():
            if key not in seen:
                out_lines.append(f"{key}={value}")
        content = "\n".join(out_lines).rstrip() + "\n"
    elif create:
        content = "\n".join(f"{key}={value}" for key, value in sorted(merged.items())) + "\n"
    else:
        raise FileNotFoundError(path)

    file_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = file_path.with_suffix(file_path.suffix + ".tmp")
    tmp_path.write_text(content, encoding="utf-8")
    os.chmod(tmp_path, 0o600)
    os.replace(tmp_path, file_path)

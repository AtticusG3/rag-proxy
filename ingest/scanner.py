"""Scan storage directories for embeddable files."""

from __future__ import annotations

import os
from pathlib import Path

from ingest.types import determine_file_type


def scan_storage(*directories: str) -> list[str]:
    """List embeddable file paths under the given directories."""
    found: list[str] = []
    seen: set[str] = set()
    for directory in directories:
        if not directory or not os.path.isdir(directory):
            continue
        for root, _dirs, files in os.walk(directory):
            for name in sorted(files):
                full = os.path.join(root, name)
                if full in seen:
                    continue
                if determine_file_type(full) == "unknown":
                    continue
                seen.add(full)
                found.append(full)
    return sorted(found)

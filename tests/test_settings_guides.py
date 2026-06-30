"""Tests for settings UI guide metadata."""

from __future__ import annotations

from rag_admin.settings_guides import GROUP_TUNING, field_placeholder
from rag_admin.settings_schema import SETTING_FIELDS


def test_group_tuning_covers_all_settings_tabs() -> None:
    groups = {field.group for field in SETTING_FIELDS}
    assert groups <= set(GROUP_TUNING.keys())


def test_file_concurrency_placeholder_is_auto_sentinel() -> None:
    field = next(f for f in SETTING_FIELDS if f.key == "INGEST_FILE_CONCURRENCY")
    assert field_placeholder(field) == "auto (1-4 by pool size)"

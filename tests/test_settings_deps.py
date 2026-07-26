"""Tests for settings dependency warnings and select labels."""

from __future__ import annotations

from rag_admin.settings_schema import SETTING_FIELDS
from rag_admin.settings_ui import (
    SETTING_REQUIRES,
    SETTING_REQUIRES_NONEMPTY,
    evaluate_setting_warnings,
    option_labels_for,
    settings_control_plane,
)


def _defaults() -> dict[str, str]:
    return {field.key: field.default for field in SETTING_FIELDS}


def test_option_labels_for_sparse_reindex() -> None:
    field = next(f for f in SETTING_FIELDS if f.key == "INGEST_SPARSE_REINDEX")
    choices = option_labels_for(field.key, field.options)
    assert choices == [
        {"value": "off", "label": "Off"},
        {"value": "each", "label": "After each file"},
        {"value": "idle", "label": "When idle"},
    ]


def test_cognitive_child_warns_without_master() -> None:
    values = _defaults()
    values["ENABLE_COGNITIVE_PIPELINE"] = "false"
    values["ENABLE_INTENT_ROUTER"] = "true"
    warnings = evaluate_setting_warnings(values)
    assert any(
        w["key"] == "ENABLE_INTENT_ROUTER" and "Cognitive pipeline" in w["message"]
        for w in warnings
    )


def test_hybrid_warns_without_sparse_url() -> None:
    values = _defaults()
    values["ENABLE_HYBRID_RETRIEVAL"] = "true"
    values["SPARSE_INDEX_URL"] = ""
    warnings = evaluate_setting_warnings(values)
    assert any(w["key"] == "ENABLE_HYBRID_RETRIEVAL" for w in warnings)


def test_turbovec_backend_warns_without_url() -> None:
    values = _defaults()
    values["DENSE_BACKEND"] = "turbovec"
    values["TURBOVEC_URL"] = ""
    warnings = evaluate_setting_warnings(values)
    assert any(w["key"] == "DENSE_BACKEND" for w in warnings)


def test_gating_log_only_requires_gating_and_master() -> None:
    values = _defaults()
    values["ENABLE_COGNITIVE_PIPELINE"] = "true"
    values["ENABLE_RETRIEVAL_GATING"] = "false"
    values["GATING_LOG_ONLY"] = "true"
    warnings = evaluate_setting_warnings(values)
    assert any(w["key"] == "GATING_LOG_ONLY" for w in warnings)


def test_clean_cognitive_rollout_has_no_warnings() -> None:
    values = _defaults()
    values["ENABLE_COGNITIVE_PIPELINE"] = "true"
    values["ENABLE_RETRIEVAL_GATING"] = "true"
    values["GATING_LOG_ONLY"] = "true"
    values["INTENT_MODEL"] = "auto"
    values["ENABLE_INTENT_ROUTER"] = "true"
    warnings = evaluate_setting_warnings(values)
    assert warnings == []


def test_dep_keys_reference_known_settings() -> None:
    known = {field.key for field in SETTING_FIELDS}
    for key, parents in SETTING_REQUIRES.items():
        assert key in known
        for parent in parents:
            assert parent in known
    for key, deps in SETTING_REQUIRES_NONEMPTY.items():
        assert key in known
        for dep in deps:
            assert dep in known


def test_control_plane_payload_shape() -> None:
    payload = settings_control_plane(_defaults())
    assert "requires" in payload
    assert "ENABLE_MEMGRAPHRAG" in payload["requires"]
    assert payload["requires"]["ENABLE_MEMGRAPHRAG"] == ["ENABLE_COGNITIVE_PIPELINE"]
    assert isinstance(payload["warnings"], list)

"""Tests for llama-swap JSON helpers."""

from rag_proxy.clients.llama_swap import parse_json_object


def test_parse_json_object_extracts_brace_wrapped_payload():
    raw = 'Here is JSON: {"query":"kubernetes pods"} trailing text'
    data = parse_json_object(raw)
    assert data == {"query": "kubernetes pods"}


def test_parse_json_object_returns_none_for_invalid_payload():
    assert parse_json_object("not json") is None

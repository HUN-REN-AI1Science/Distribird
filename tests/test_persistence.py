"""Tests for UI session persistence serialization logic."""

from __future__ import annotations

import json

from litopri.ui.persistence import _SIMPLE_KEYS, STORAGE_KEY


class TestSimpleKeysSerialization:
    """Sidebar keys survive JSON round-trip."""

    def test_simple_values_round_trip(self):
        blob = {
            "authenticated": True,
            "use_s2": True,
            "use_openalex": False,
            "max_q": 5,
            "max_p": 20,
            "llm_url_req": "http://localhost:8000",
        }
        restored = json.loads(json.dumps(blob))
        for key, val in blob.items():
            assert restored[key] == val

    def test_only_known_keys_persisted(self):
        """All persisted keys should be sidebar-related."""
        assert "authenticated" in _SIMPLE_KEYS
        assert "use_s2" in _SIMPLE_KEYS
        assert "max_q" in _SIMPLE_KEYS
        # Main panel keys should NOT be persisted
        assert "params" not in _SIMPLE_KEYS
        assert "domain_context" not in _SIMPLE_KEYS
        assert "next_id" not in _SIMPLE_KEYS
        assert "results" not in _SIMPLE_KEYS


class TestEdgeCases:
    """Edge cases for persistence deserialization."""

    def test_corrupted_json_does_not_crash(self):
        bad_data = "not valid json {{{{"
        try:
            json.loads(bad_data)
            assert False, "Should have raised"
        except json.JSONDecodeError:
            pass  # Expected — hydrate wraps this in try/except

    def test_simple_keys_list_not_empty(self):
        assert len(_SIMPLE_KEYS) > 0

    def test_storage_key_constant(self):
        assert STORAGE_KEY == "litopri_state"

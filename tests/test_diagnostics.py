"""Tests for the opt-in debug tracer (agent/diagnostics.py) and instrumentation."""

from __future__ import annotations

import types

from distribird.agent import diagnostics
from distribird.agent.extract import _llm_json_call, reset_token_accumulator
from distribird.config import Settings
from distribird.distributions.fitting import values_to_prior
from distribird.models import ParameterInput


def _param() -> ParameterInput:
    return ParameterInput(name="permeability", description="rock permeability", unit="m2")


def test_disabled_is_noop():
    """With debug_trace off, no trace is installed and record() does nothing."""
    trace = diagnostics.start_run(_param(), Settings(debug_trace=False))
    assert trace is None
    assert diagnostics.enabled() is False
    diagnostics.record("llm_call", {"x": 1})  # must not raise
    assert diagnostics.get_trace() is None


def test_enabled_records_events():
    trace = diagnostics.start_run(_param(), Settings(debug_trace=True))
    assert trace is not None
    assert diagnostics.enabled() is True
    diagnostics.set_node("search")
    diagnostics.record("search_request", {"query": "q"})
    diagnostics.record("pdf_fetch", {"outcome": "skipped"})
    diagnostics.finish()

    d = diagnostics.get_trace().to_dict()
    assert d["n_events"] == 2
    assert [e["kind"] for e in d["events"]] == ["search_request", "pdf_fetch"]
    assert d["events"][0]["node"] == "search"
    assert d["events"][0]["seq"] == 1 and d["events"][1]["seq"] == 2
    assert d["finished_at"] is not None
    # clean up the contextvar for later tests
    diagnostics.start_run(_param(), Settings(debug_trace=False))


def test_settings_snapshot_is_redacted():
    s = Settings(
        debug_trace=True,
        llm_api_key="sk-secret",
        semantic_scholar_api_key="s2-secret",
        auth_password="hunter2",
    )
    trace = diagnostics.start_run(_param(), s)
    snap = trace.to_dict()["settings"]
    assert snap["llm_api_key"] == "***redacted***"
    assert snap["semantic_scholar_api_key"] == "***redacted***"
    assert snap["auth_password"] == "***redacted***"
    diagnostics.start_run(_param(), Settings(debug_trace=False))


def test_fitting_records_candidates():
    diagnostics.start_run(_param(), Settings(debug_trace=True))
    values_to_prior("permeability", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 0.0, 100.0)
    d = diagnostics.get_trace().to_dict()
    kinds = [e["kind"] for e in d["events"]]
    assert "fitting_input" in kinds
    assert "fitting_candidates" in kinds
    cand = [e for e in d["events"] if e["kind"] == "fitting_candidates"][0]["data"]
    assert cand["chosen_family"]
    assert len(cand["candidates"]) >= 1
    fit_in = [e for e in d["events"] if e["kind"] == "fitting_input"][0]["data"]
    assert fit_in["tier"] == "aic_selection"
    assert fit_in["values"] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    diagnostics.start_run(_param(), Settings(debug_trace=False))


class _FakeCompletions:
    def create(self, **kwargs):
        msg = types.SimpleNamespace(content='{"value": 42}')
        choice = types.SimpleNamespace(message=msg)
        usage = types.SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        return types.SimpleNamespace(choices=[choice], usage=usage)


class _FakeClient:
    def __init__(self):
        self.chat = types.SimpleNamespace(completions=_FakeCompletions())


def test_llm_call_is_recorded():
    diagnostics.start_run(_param(), Settings(debug_trace=True))
    reset_token_accumulator()
    out = _llm_json_call(
        _FakeClient(),
        "test-model",
        [{"role": "user", "content": "hi"}],
        temperature=0.0,
        label="value_extraction",
    )
    assert out == {"value": 42}
    d = diagnostics.get_trace().to_dict()
    calls = [e for e in d["events"] if e["kind"] == "llm_call"]
    assert len(calls) == 1
    data = calls[0]["data"]
    assert data["label"] == "value_extraction"
    assert data["status"] == "ok"
    assert data["model"] == "test-model"
    assert data["temperature"] == 0.0
    assert data["usage"]["total_tokens"] == 15
    assert data["parsed"] == {"value": 42}
    # full prompt captured (system message prepended + user message)
    assert any(m["content"] == "hi" for m in data["messages"])
    diagnostics.start_run(_param(), Settings(debug_trace=False))

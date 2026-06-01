"""Tests for the trace JSON/HTML exporters (export/trace_export.py)."""

from __future__ import annotations

import json

from distribird.export.trace_export import (
    _main,
    export_trace_html,
    export_trace_json,
)


def _sample_trace() -> dict:
    return {
        "run_id": "permeability_123",
        "parameter": {"name": "permeability", "description": "d"},
        "model": "test-model",
        "llm_temperature": 0.0,
        "llm_seed": 7,
        "events": [
            {
                "seq": 1,
                "kind": "search_request",
                "node": "search",
                "rel_ts": 0.1,
                "data": {
                    "source": "semantic_scholar",
                    "query": "perm",
                    "n_raw": 5,
                    "n_after_oa_filter": 2,
                    "results": [],
                },
            },
            {
                "seq": 2,
                "kind": "fitting_candidates",
                "node": "synthesize",
                "rel_ts": 1.0,
                "data": {
                    "chosen_family": "lognormal",
                    "delta_aic_to_second": 3.2,
                    "candidates": [{"family": "lognormal", "aic": 10.0}],
                },
            },
        ],
        "node_events": [{"node": "search", "duration_s": 0.1, "summary": {}}],
    }


def test_export_trace_json_roundtrip(tmp_path):
    path = tmp_path / "trace.json"
    export_trace_json(_sample_trace(), str(path))
    loaded = json.loads(path.read_text())
    assert loaded["run_id"] == "permeability_123"
    assert len(loaded["events"]) == 2


def test_export_trace_html_single(tmp_path):
    path = tmp_path / "viewer.html"
    export_trace_html(_sample_trace(), str(path))
    html = path.read_text()
    assert "__TRACE_DATA__" not in html  # token replaced
    assert "const TRACES = [" in html
    assert "permeability_123" in html
    # injected JSON must not contain a literal </ that could break the <script>
    body = html.split("const TRACES = ", 1)[1]
    assert "</script" not in body.split(";", 1)[0]


def test_export_trace_html_diff(tmp_path):
    a = _sample_trace()
    b = _sample_trace()
    b["run_id"] = "permeability_456"
    b["events"][1]["data"]["chosen_family"] = "gamma"
    path = tmp_path / "diff.html"
    export_trace_html([a, b], str(path))
    html = path.read_text()
    assert "permeability_123" in html and "permeability_456" in html


def test_cli_builds_viewer(tmp_path):
    j = tmp_path / "t.json"
    export_trace_json(_sample_trace(), str(j))
    out = tmp_path / "out.html"
    rc = _main([str(j), "-o", str(out)])
    assert rc == 0
    assert out.exists()
    assert "permeability_123" in out.read_text()

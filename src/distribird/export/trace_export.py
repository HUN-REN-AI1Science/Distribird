"""Export Distribird debug traces to JSON and a standalone interactive HTML viewer.

A "trace" here is the dict produced by ``agent.diagnostics.RunTrace.to_dict()`` and
surfaced on ``PipelineResult.debug_trace`` (or written to ``logs/traces/<run>.json``
when ``debug_trace`` is enabled).

- :func:`export_trace_json` writes one trace to disk.
- :func:`export_trace_html` injects one or more traces into a self-contained HTML
  viewer (graph + timeline + drill-down + side-by-side run diff). Pass 2+ traces to
  pre-load the comparison view; the viewer also lets you load more trace JSON files
  interactively.

CLI::

    python -m distribird.export.trace_export run_a.json [run_b.json ...] -o viewer.html
"""

from __future__ import annotations

import json
from importlib.resources import files
from typing import Any

_DATA_TOKEN = "__TRACE_DATA__"


def _load_template() -> str:
    """Load the viewer template as package data (works for installed wheels/zips)."""
    return (files("distribird.export") / "templates" / "trace_viewer.html").read_text(
        encoding="utf-8"
    )


def _safe_json(traces: list[dict[str, Any]]) -> str:
    """JSON-encode traces for safe inlining inside a <script> tag."""
    raw = json.dumps(traces, default=str)
    # Prevent premature </script> termination and HTML-comment injection.
    return raw.replace("</", "<\\/").replace("<!--", "<\\!--")


def export_trace_json(trace: dict[str, Any], path: str) -> str:
    """Write a single trace dict to ``path`` as pretty JSON. Returns the path."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(trace, fh, indent=2, default=str)
    return path


def export_trace_html(traces: dict[str, Any] | list[dict[str, Any]], path: str) -> str:
    """Render a self-contained HTML viewer embedding ``traces``. Returns the path.

    ``traces`` may be a single trace dict or a list of them (for diff mode).
    """
    if isinstance(traces, dict):
        traces = [traces]
    html = _load_template().replace(_DATA_TOKEN, _safe_json(list(traces)))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return path


def _main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Build a standalone HTML viewer from Distribird debug trace JSON file(s). "
        "Pass two or more to enable side-by-side diff."
    )
    parser.add_argument("traces", nargs="+", help="Path(s) to trace JSON file(s)")
    parser.add_argument("-o", "--output", default="trace_viewer.html", help="Output HTML path")
    args = parser.parse_args(argv)

    loaded: list[dict[str, Any]] = []
    for p in args.traces:
        with open(p, encoding="utf-8") as fh:
            loaded.append(json.load(fh))

    out = export_trace_html(loaded, args.output)
    print(f"Wrote viewer with {len(loaded)} run(s) to {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())

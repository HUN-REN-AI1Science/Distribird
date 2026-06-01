#!/usr/bin/env python
"""Run one Distribird parameter with debug tracing on, then emit a JSON trace and
a standalone interactive HTML viewer.

Usage:
    python scripts/trace_run.py "field_capacity" \
        --description "Volumetric water content at field capacity" \
        --unit "cm3/cm3" --lower 0.1 --upper 0.5 \
        --domain "maize soil hydrology" \
        --out-dir logs/traces

Connection settings (LLM host, API keys, source toggles) are read from the
environment / .env exactly like a normal run; this script only forces
``debug_trace=True``.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from distribird.agent.pipeline import run_parameter
from distribird.config import get_settings
from distribird.export.trace_export import export_trace_html, export_trace_json
from distribird.models import ConstraintSpec, ParameterInput


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("name", help="Parameter name")
    p.add_argument("--description", default="", help="What the parameter represents")
    p.add_argument("--unit", default="", help="Physical unit")
    p.add_argument("--lower", type=float, default=None, help="Lower bound")
    p.add_argument("--upper", type=float, default=None, help="Upper bound")
    p.add_argument("--domain", default="", help="Domain / application context")
    p.add_argument("--out-dir", default="logs/traces", help="Where to write trace artifacts")
    args = p.parse_args(argv)

    settings = get_settings()
    settings.debug_trace = True
    settings.trace_output_dir = args.out_dir

    parameter = ParameterInput(
        name=args.name,
        description=args.description or args.name,
        unit=args.unit,
        constraints=ConstraintSpec(lower_bound=args.lower, upper_bound=args.upper),
        domain_context=args.domain,
    )

    result = asyncio.run(run_parameter(parameter, settings))

    if not result.debug_trace:
        print("No trace captured (debug_trace did not produce a trace).", file=sys.stderr)
        return 1

    os.makedirs(args.out_dir, exist_ok=True)
    run_id = result.debug_trace.get("run_id", args.name)
    json_path = os.path.join(args.out_dir, f"{run_id}.json")
    html_path = os.path.join(args.out_dir, f"{run_id}.html")
    export_trace_json(result.debug_trace, json_path)
    export_trace_html(result.debug_trace, html_path)

    print(f"Prior: {result.prior.display_name()}  ({result.prior.confidence.value})")
    print(f"Trace JSON: {json_path}")
    print(f"Viewer:     {html_path}")
    print(f"Open in a browser:  file://{os.path.abspath(html_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

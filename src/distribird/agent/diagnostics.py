"""Opt-in structured execution tracer for a single parameter run.

This is a ``contextvars``-based sink, mirroring the per-task token accumulator in
``extract.py``. When tracing is enabled (``Settings.debug_trace`` /
``DISTRIBIRD_DEBUG_TRACE=true``), ``start_run`` installs a fresh :class:`RunTrace`
into a context variable; every instrumented chokepoint in the pipeline (LLM calls,
literature searches, PDF fetches, value extraction, distribution fitting, routing)
then calls :func:`record` to append a structured event.

Because asyncio copies context variables per task while sharing the underlying
mutable object, a single :class:`RunTrace` installed at the top of
``run_parameter_graph`` collects events from every subtask of that parameter's run —
without threading an argument through pure functions like ``distributions/fitting.py``.

When tracing is disabled the context variable stays ``None`` and every helper is a
cheap no-op, so the instrumentation is behaviour-neutral.
"""

from __future__ import annotations

import contextvars
import json
import os
import re
import time
from typing import Any

# Keys whose values must never be written to a trace artifact.
_SENSITIVE = re.compile(r"(api_key|password|secret|token|authorization)", re.IGNORECASE)

_trace_var: contextvars.ContextVar["RunTrace | None"] = contextvars.ContextVar(
    "distribird_trace", default=None
)


def _redact(value: Any) -> Any:
    """Recursively redact sensitive keys from a settings/dict snapshot."""
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            if isinstance(k, str) and _SENSITIVE.search(k):
                out[k] = "***redacted***" if v else ""
            else:
                out[k] = _redact(v)
        return out
    if isinstance(value, (list, tuple)):
        return [_redact(v) for v in value]
    return value


class RunTrace:
    """Accumulates all structured events for one parameter's pipeline run."""

    def __init__(self, run_id: str, parameter: Any, settings: Any) -> None:
        self.run_id = run_id
        self.started_at = time.time()
        self.finished_at: float | None = None
        self.current_node: str | None = None
        self._seq = 0
        self.events: list[dict[str, Any]] = []
        self.node_events: list[dict[str, Any]] = []

        # Parameter snapshot
        try:
            self.parameter = parameter.model_dump()  # pydantic
        except Exception:
            self.parameter = {"name": getattr(parameter, "name", str(parameter))}

        # Redacted settings snapshot + effective sampling knobs (for determinism analysis)
        try:
            snap = settings.model_dump()
        except Exception:
            snap = {}
        self.settings = _redact(snap)
        self.model = snap.get("llm_model", "")
        self.llm_temperature = {
            "precise": snap.get("llm_temperature_precise"),
            "creative": snap.get("llm_temperature_creative"),
            "deliberation": snap.get("llm_temperature_deliberation"),
        }
        self.llm_seed = snap.get("llm_seed")

    def add(self, kind: str, data: dict[str, Any]) -> None:
        self._seq += 1
        self.events.append(
            {
                "seq": self._seq,
                "kind": kind,
                "node": self.current_node,
                "ts": time.time(),
                "rel_ts": time.time() - self.started_at,
                "data": data,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "parameter": self.parameter,
            "settings": self.settings,
            "model": self.model,
            "llm_temperature": self.llm_temperature,
            "llm_seed": self.llm_seed,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "n_events": len(self.events),
            "events": self.events,
            "node_events": self.node_events,
        }


def start_run(parameter: Any, settings: Any) -> "RunTrace | None":
    """Install a fresh trace for this task if tracing is enabled; else no-op.

    Returns the installed :class:`RunTrace`, or ``None`` when disabled.
    """
    if not getattr(settings, "debug_trace", False):
        _trace_var.set(None)
        return None
    name = getattr(parameter, "name", "param")
    run_id = f"{name}_{int(time.time() * 1000)}"
    trace = RunTrace(run_id, parameter, settings)
    _trace_var.set(trace)
    return trace


def enabled() -> bool:
    """True when a trace is currently installed."""
    return _trace_var.get() is not None


def set_node(name: str) -> None:
    """Tag subsequent events with the currently executing graph node."""
    trace = _trace_var.get()
    if trace is not None:
        trace.current_node = name


def record(kind: str, data: dict[str, Any]) -> None:
    """Append one structured event to the active trace (no-op when disabled)."""
    trace = _trace_var.get()
    if trace is None:
        return
    trace.add(kind, data)


def get_trace() -> "RunTrace | None":
    return _trace_var.get()


def finish(node_events: list[Any] | None = None) -> "RunTrace | None":
    """Stamp the trace as finished, fold in per-node TraceEvents, and return it.

    ``node_events`` are the graph's coarse per-node timing records (pydantic
    ``TraceEvent``s or plain dicts); they are normalized to dicts. Returns None
    when tracing is disabled.
    """
    trace = _trace_var.get()
    if trace is None:
        return None
    trace.finished_at = time.time()
    if node_events:
        trace.node_events = [
            te.model_dump() if hasattr(te, "model_dump") else te for te in node_events
        ]
    return trace


def dump(trace: "RunTrace | None", path: str) -> None:
    """Serialize a trace to JSON. Non-serializable values fall back to ``str``."""
    if trace is None:
        return
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(trace.to_dict(), fh, indent=2, default=str)


def write_trace(trace: "RunTrace | None", out_dir: str) -> str | None:
    """Persist a trace to ``<out_dir>/<run_id>.json`` and return the path.

    No-op (returns None) when tracing is disabled.
    """
    if trace is None:
        return None
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{trace.run_id}.json")
    dump(trace, path)
    return path

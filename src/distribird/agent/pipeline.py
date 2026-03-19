"""Main orchestrator: LangGraph pipeline for literature search → extract → fit."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from distribird.agent.graph import NODE_META, ProgressCallback, run_parameter_graph
from distribird.config import Settings, get_settings
from distribird.models import (
    BatchResult,
    ParameterInput,
    PipelineResult,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Batch progress tracker
# ---------------------------------------------------------------------------

class BatchProgressTracker:
    """Track and log progress across multiple concurrent parameters."""

    def __init__(self, parameters: list[ParameterInput]) -> None:
        self._names = [p.name for p in parameters]
        self._total = len(parameters)
        self._status: dict[str, str] = {p.name: "queued" for p in parameters}
        self._completed = 0
        self._start = time.monotonic()
        self._lock = asyncio.Lock()

    def _format_elapsed(self) -> str:
        elapsed = int(time.monotonic() - self._start)
        m, s = divmod(elapsed, 60)
        return f"{m:02d}:{s:02d}"

    async def on_start(self, name: str) -> None:
        async with self._lock:
            self._status[name] = "enrich"
            self._log_status()

    async def on_node(self, name: str, node: str, state: dict[str, Any]) -> None:
        async with self._lock:
            label, _ = NODE_META.get(node, (node, 0))
            self._status[name] = label
            self._log_status()

    async def on_done(self, name: str, result: PipelineResult) -> None:
        async with self._lock:
            self._completed += 1
            prior = result.prior
            tag = "informative" if prior.is_informative else "uninformative"
            self._status[name] = f"DONE ({prior.display_name()}, {tag})"
            self._log_status()

    async def on_error(self, name: str, error: str) -> None:
        async with self._lock:
            self._completed += 1
            self._status[name] = f"FAILED ({error})"
            self._log_status()

    def _log_status(self) -> None:
        elapsed = self._format_elapsed()
        active = [
            f"  {name}: {status}"
            for name, status in self._status.items()
            if not status.startswith("DONE") and not status.startswith("FAILED") and status != "queued"
        ]
        done = sum(
            1 for s in self._status.values()
            if s.startswith("DONE") or s.startswith("FAILED")
        )
        header = f"[{elapsed}] Progress: {done}/{self._total} complete"
        lines = [header]
        if active:
            lines.extend(active)
        # Log each finished param that was just completed (last DONE/FAILED)
        for name, status in self._status.items():
            if status.startswith("DONE") or status.startswith("FAILED"):
                pass  # already counted
        logger.info("\n".join(lines))
        print("\n".join(lines), flush=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def run_parameter(
    parameter: ParameterInput,
    settings: Settings | None = None,
    on_node_complete: ProgressCallback = None,
) -> PipelineResult:
    """Run the LangGraph pipeline for a single parameter."""
    return await run_parameter_graph(parameter, settings, on_node_complete)


async def run_batch(
    parameters: list[ParameterInput],
    settings: Settings | None = None,
    progress: bool = True,
) -> BatchResult:
    """Run the pipeline for multiple parameters with bounded concurrency."""
    if settings is None:
        settings = get_settings()

    tracker = BatchProgressTracker(parameters) if progress else None
    semaphore = asyncio.Semaphore(settings.max_parallel_parameters)

    async def _run_one(param: ParameterInput) -> PipelineResult:
        async with semaphore:
            if tracker:
                await tracker.on_start(param.name)

            def _on_node(node_name: str, state: dict[str, Any]) -> None:
                if tracker:
                    asyncio.get_event_loop().create_task(
                        tracker.on_node(param.name, node_name, state)
                    )

            try:
                result = await run_parameter(param, settings, _on_node)
                if tracker:
                    await tracker.on_done(param.name, result)
                return result
            except Exception as e:
                logger.error(f"Pipeline failed for '{param.name}': {e}")
                if tracker:
                    await tracker.on_error(param.name, str(e)[:60])
                from distribird.distributions.uninformative import wide_normal_prior

                fallback_prior = wide_normal_prior(
                    param.name,
                    param.constraints.lower_bound,
                    param.constraints.upper_bound,
                )
                return PipelineResult(
                    parameter=param,
                    prior=fallback_prior,
                    warnings=[f"Pipeline error: {e}"],
                )

    results = await asyncio.gather(*[_run_one(p) for p in parameters])

    return BatchResult(
        results=list(results),
        metadata={"n_parameters": len(parameters)},
    )

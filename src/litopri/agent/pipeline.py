"""Main orchestrator: LangGraph pipeline for literature search → extract → fit."""

from __future__ import annotations

import asyncio
import logging

from litopri.agent.graph import run_parameter_graph
from litopri.config import Settings, get_settings
from litopri.models import (
    BatchResult,
    ParameterInput,
    PipelineResult,
)

logger = logging.getLogger(__name__)


async def run_parameter(
    parameter: ParameterInput,
    settings: Settings | None = None,
) -> PipelineResult:
    """Run the LangGraph pipeline for a single parameter."""
    return await run_parameter_graph(parameter, settings)


async def run_batch(
    parameters: list[ParameterInput],
    settings: Settings | None = None,
) -> BatchResult:
    """Run the pipeline for multiple parameters with bounded concurrency."""
    if settings is None:
        settings = get_settings()

    semaphore = asyncio.Semaphore(settings.max_parallel_parameters)

    async def _run_one(param: ParameterInput) -> PipelineResult:
        async with semaphore:
            try:
                return await run_parameter(param, settings)
            except Exception as e:
                logger.error(f"Pipeline failed for '{param.name}': {e}")
                from litopri.distributions.uninformative import wide_normal_prior

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

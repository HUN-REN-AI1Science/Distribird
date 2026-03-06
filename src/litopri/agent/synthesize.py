"""Connect extraction results to distribution fitting."""

from __future__ import annotations

import math

from litopri.distributions.constraints import filter_values_by_constraints
from litopri.distributions.fitting import values_to_prior
from litopri.models import (
    EnrichedContext,
    FittedPrior,
    LiteratureEvidence,
    ParameterInput,
    WeightedValue,
)


def collect_weighted_values(papers: list[LiteratureEvidence]) -> list[WeightedValue]:
    """Flatten all extracted values from papers into weighted values.

    Weight = sqrt(sample_size) if available, else 1.0.
    For range-only values: midpoint as value, (hi - lo) / 4 as uncertainty.
    """
    weighted: list[WeightedValue] = []
    for paper in papers:
        for ev in paper.extracted_values:
            if ev.reported_value is not None:
                has_n = ev.sample_size and ev.sample_size > 0
                weight = math.sqrt(ev.sample_size) if has_n else 1.0
                weighted.append(
                    WeightedValue(
                        value=ev.reported_value,
                        weight=weight,
                        uncertainty=ev.uncertainty,
                        source_paper=paper.title,
                    )
                )
            elif ev.reported_range is not None:
                lo, hi = ev.reported_range
                midpoint = (lo + hi) / 2.0
                uncertainty = (hi - lo) / 4.0
                weighted.append(
                    WeightedValue(
                        value=midpoint,
                        weight=1.0,
                        uncertainty=uncertainty,
                        source_paper=paper.title,
                    )
                )
    return weighted


def collect_values(papers: list[LiteratureEvidence]) -> list[float]:
    """Flatten all extracted reported_values from papers (backward-compatible)."""
    return [wv.value for wv in collect_weighted_values(papers)]


def _infer_bounds_from_enrichment(
    parameter: ParameterInput,
    enrichment: EnrichedContext | None,
) -> tuple[float | None, float | None]:
    """Derive effective bounds: user constraints first, then enrichment typical_range."""
    lb = parameter.constraints.lower_bound
    ub = parameter.constraints.upper_bound
    if lb is not None and ub is not None:
        return lb, ub

    if enrichment and enrichment.typical_range:
        from litopri.agent.extract import _parse_typical_range

        parsed = _parse_typical_range(enrichment.typical_range)
        if parsed is not None:
            low, high = parsed
            width = max(high - low, 1.0)
            # Extend the typical range by 1x width on each side to form plausible bounds
            inferred_lb = low - width
            inferred_ub = high + width
            if lb is None:
                lb = inferred_lb
            if ub is None:
                ub = inferred_ub

    return lb, ub


def synthesize_prior(
    parameter: ParameterInput,
    papers: list[LiteratureEvidence],
    enrichment: EnrichedContext | None = None,
) -> FittedPrior:
    """Synthesize a fitted prior from literature evidence."""
    weighted_values = collect_weighted_values(papers)
    raw_values = [wv.value for wv in weighted_values]

    # Filter by constraints
    valid_values, excluded = filter_values_by_constraints(raw_values, parameter.constraints)

    # Build weights and uncertainties for valid values only
    valid_set = set()
    for i, v in enumerate(raw_values):
        if v in valid_values:
            valid_set.add(i)

    weights = []
    uncertainties = []
    valid_idx = 0
    for i, wv in enumerate(weighted_values):
        if valid_idx < len(valid_values) and wv.value == valid_values[valid_idx]:
            weights.append(wv.weight)
            uncertainties.append(wv.uncertainty)
            valid_idx += 1

    # Use enrichment typical_range as fallback bounds when user provides none
    effective_lb, effective_ub = _infer_bounds_from_enrichment(parameter, enrichment)

    prior = values_to_prior(
        parameter_name=parameter.name,
        values=valid_values,
        lower_bound=effective_lb,
        upper_bound=effective_ub,
        weights=weights if weights else None,
        uncertainties=uncertainties if uncertainties else None,
    )

    # Attach evidence
    prior.evidence = [p for p in papers if p.extracted_values]
    prior.n_sources = len(prior.evidence)

    return prior

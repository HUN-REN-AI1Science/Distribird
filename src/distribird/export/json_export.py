"""JSON export for fitted priors."""

from __future__ import annotations

import json

from distribird.models import BatchResult, PipelineResult


def result_to_dict(result: PipelineResult) -> dict[str, object]:
    """Convert a pipeline result to a JSON-serializable dict."""
    d: dict[str, object] = {
        "parameter": result.parameter.name,
        "distribution": result.prior.family.value,
        "params": result.prior.params,
        "confidence": result.prior.confidence.value,
        "is_informative": result.prior.is_informative,
        "reason": result.prior.reason,
        "n_sources": result.prior.n_sources,
        "citations": [
            {
                "title": e.title,
                "doi": e.doi,
                "year": e.year,
                "authors": e.authors,
            }
            for e in result.prior.evidence
        ],
        "warnings": result.warnings,
    }
    if result.model_check is not None:
        d["model_check"] = result.model_check.model_dump()
    d["parameter_validity"] = result.parameter_validity.value
    d["validity_reason"] = result.validity_reason
    d["validity_signals"] = result.validity_signals
    d["is_empirical"] = result.is_empirical
    return d


def export_json(batch: BatchResult, indent: int = 2) -> str:
    """Export batch results as a JSON string."""
    data = {
        "distribird_version": "0.1.0",
        "parameters": [result_to_dict(r) for r in batch.results],
        "metadata": batch.metadata,
    }
    return json.dumps(data, indent=indent)


def export_single_json(result: PipelineResult, indent: int = 2) -> str:
    """Export a single parameter result as JSON."""
    return json.dumps(result_to_dict(result), indent=indent)

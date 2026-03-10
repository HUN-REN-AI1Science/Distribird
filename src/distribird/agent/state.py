"""Shared state for the LangGraph pipeline."""

from __future__ import annotations

import time
from enum import Enum
from typing import TypedDict

from pydantic import BaseModel, Field

from distribird.models import (
    AgentFinding,
    DeliberationResult,
    EnrichedContext,
    FittedPrior,
    LiteratureEvidence,
    ParameterInput,
)

# ---------------------------------------------------------------------------
# Supporting Pydantic models
# ---------------------------------------------------------------------------


class MessageKind(str, Enum):
    DISCOVERY = "discovery"
    QUERY_SUGGESTION = "query_suggestion"
    TERMINOLOGY = "terminology"
    CROSS_REF = "cross_ref"
    WARNING = "warning"


class BlackboardMessage(BaseModel):
    sender: str
    kind: MessageKind
    content: str
    references: list[str] = Field(default_factory=list)
    iteration: int = 0


class QualityMetrics(BaseModel):
    n_papers_found: int = 0
    n_total_values: int = 0
    n_high_confidence_values: int = 0
    extraction_coverage: float = 0.0
    value_cv: float | None = None

    def is_sufficient(self, min_values: int = 2) -> bool:
        return self.n_total_values >= min_values

    def needs_search_refinement(self) -> bool:
        return self.n_total_values == 0 and self.n_papers_found > 0

    def needs_extraction_refinement(self) -> bool:
        return (
            self.n_total_values > 0
            and self.n_high_confidence_values == 0
            and self.value_cv is not None
            and self.value_cv > 1.5
        )


class IterationBudget(BaseModel):
    search_refinement_max: int = 2
    search_refinement_used: int = 0
    cross_enrichment_max: int = 1
    cross_enrichment_used: int = 0
    extraction_refinement_max: int = 1
    extraction_refinement_used: int = 0
    total_llm_calls_max: int = 30
    total_llm_calls_used: int = 0
    snowball_max: int = 1
    snowball_used: int = 0

    def can_snowball(self) -> bool:
        return self.snowball_used < self.snowball_max

    def can_refine_search(self) -> bool:
        return self.search_refinement_used < self.search_refinement_max and self.has_budget()

    def can_cross_enrich(self) -> bool:
        return self.cross_enrichment_used < self.cross_enrichment_max and self.has_budget()

    def can_refine_extraction(self) -> bool:
        return (
            self.extraction_refinement_used < self.extraction_refinement_max and self.has_budget()
        )

    def has_budget(self) -> bool:
        return self.total_llm_calls_used < self.total_llm_calls_max


class TraceEvent(BaseModel):
    node: str
    timestamp: float = Field(default_factory=time.time)
    duration_s: float = 0.0
    summary: dict[str, object] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# LangGraph shared state (TypedDict)
# ---------------------------------------------------------------------------


class PipelineState(TypedDict, total=False):
    # Inputs (set once)
    parameter: ParameterInput
    settings_dict: dict[str, object]

    # Accumulated data
    enrichment: EnrichedContext | None
    search_queries: list[str]
    all_queries_tried: list[str]
    all_papers: list[LiteratureEvidence]
    papers_with_values: list[LiteratureEvidence]
    seen_dois: set[str]
    agent_findings: list[AgentFinding]
    deliberation: DeliberationResult | None
    prior: FittedPrior | None

    # Inter-agent communication
    blackboard: list[BlackboardMessage]

    # Quality & budget
    quality: QualityMetrics
    budget: IterationBudget
    warnings: list[str]
    refinement_context: list[str]

    # Observability
    trace_events: list[TraceEvent]


# ---------------------------------------------------------------------------
# Helper functions for manipulating state
# ---------------------------------------------------------------------------


def add_papers(
    state: PipelineState,
    new_papers: list[LiteratureEvidence],
) -> list[LiteratureEvidence]:
    """Add papers to state, deduplicating by DOI. Returns the new unique papers."""
    existing = state.get("all_papers", [])
    seen = state.get("seen_dois", set())
    added: list[LiteratureEvidence] = []
    for p in new_papers:
        doi = p.doi.strip().lower() if p.doi else None
        if doi and doi in seen:
            continue
        if doi:
            seen.add(doi)
        existing.append(p)
        added.append(p)
    return added


def post_message(
    state: PipelineState,
    sender: str,
    kind: MessageKind,
    content: str,
    references: list[str] | None = None,
) -> BlackboardMessage:
    """Append a message to the blackboard."""
    bb = state.get("blackboard", [])
    iteration = state.get("budget", IterationBudget()).search_refinement_used
    msg = BlackboardMessage(
        sender=sender,
        kind=kind,
        content=content,
        references=references or [],
        iteration=iteration,
    )
    bb.append(msg)
    return msg


def get_messages(
    state: PipelineState,
    kind: MessageKind | None = None,
    exclude_sender: str | None = None,
    max_iteration: int | None = None,
) -> list[BlackboardMessage]:
    """Read blackboard messages with optional filters."""
    msgs = state.get("blackboard", [])
    result = []
    for m in msgs:
        if kind is not None and m.kind != kind:
            continue
        if exclude_sender is not None and m.sender == exclude_sender:
            continue
        if max_iteration is not None and m.iteration > max_iteration:
            continue
        result.append(m)
    return result


def update_quality(state: PipelineState) -> QualityMetrics:
    """Recompute QualityMetrics from current state."""
    papers = state.get("all_papers", [])
    papers_with_vals = state.get("papers_with_values", [])

    all_values: list[float] = []
    high_conf = 0
    for p in papers_with_vals:
        for ev in p.extracted_values:
            if ev.reported_value is not None:
                all_values.append(ev.reported_value)
            ctx = ev.context or ""
            if "high" in ctx.lower():
                high_conf += 1

    n_vals = len(all_values)
    coverage = len(papers_with_vals) / len(papers) if papers else 0.0

    cv: float | None = None
    if n_vals >= 2:
        import statistics

        mean = statistics.mean(all_values)
        if mean != 0:
            cv = statistics.stdev(all_values) / abs(mean)

    qm = QualityMetrics(
        n_papers_found=len(papers),
        n_total_values=n_vals,
        n_high_confidence_values=high_conf,
        extraction_coverage=coverage,
        value_cv=cv,
    )
    return qm

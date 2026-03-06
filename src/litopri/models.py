"""Pydantic models for LitoPri data structures."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


class DistributionFamily(str, Enum):
    NORMAL = "normal"
    TRUNCATED_NORMAL = "truncated_normal"
    BETA = "beta"
    GAMMA = "gamma"
    LOGNORMAL = "lognormal"
    UNIFORM = "uniform"


class ConstraintSpec(BaseModel):
    """A physical constraint on a parameter."""

    lower_bound: float | None = None
    upper_bound: float | None = None
    description: str = ""


class ParameterInput(BaseModel):
    """User-provided parameter specification."""

    name: str = Field(..., description="Parameter name")
    description: str = Field(..., description="What the parameter represents")
    unit: str = Field("", description="Physical unit")
    constraints: ConstraintSpec = Field(default_factory=ConstraintSpec)
    domain_context: str = Field(
        "", description="Domain or application context (e.g., 'maize crop modeling')"
    )


class ExtractedValue(BaseModel):
    """A numerical value extracted from a paper."""

    reported_value: float | None = None
    reported_range: tuple[float, float] | None = None
    uncertainty: float | None = None
    sample_size: int | None = None
    context: str = ""


class LiteratureEvidence(BaseModel):
    """Evidence from a single paper."""

    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    doi: str | None = None
    abstract: str = ""
    full_text: str = Field("", description="Full paper text (if available)")
    pdf_url: str | None = None
    extracted_values: list[ExtractedValue] = Field(default_factory=list)
    relevance_score: float = Field(0.0, ge=0.0, le=1.0)
    relevance_snippet: str = Field("", description="Most relevant abstract snippet")
    verified: bool = Field(False, description="Whether verified against external API")
    source: str = Field("unknown", description="Origin: 'semantic_scholar' or 'llm_deep_research'")


class FittedPrior(BaseModel):
    """A fitted prior distribution for a parameter."""

    parameter_name: str
    family: DistributionFamily
    params: dict[str, float] = Field(
        ..., description="Distribution parameters (e.g., mu, sigma, a, b)"
    )
    confidence: ConfidenceLevel
    is_informative: bool
    reason: str = Field("", description="Why this distribution was chosen")
    evidence: list[LiteratureEvidence] = Field(default_factory=list)
    n_sources: int = 0

    def display_name(self) -> str:
        param_str = ", ".join(f"{k}={v:.4g}" for k, v in self.params.items())
        return f"{self.family.value}({param_str})"


class WeightedValue(BaseModel):
    """A value with associated weight for fitting."""

    value: float
    weight: float = 1.0
    uncertainty: float | None = None
    source_paper: str = ""


class EnrichedContext(BaseModel):
    """Enriched parameter context from LLM pre-research."""

    model_summary: str = ""
    parameter_meaning: str = ""
    common_terminology: list[str] = Field(default_factory=list)
    typical_range: str = ""
    enriched_description: str = ""
    search_hints: list[str] = Field(default_factory=list)
    application_context: str = Field(
        "",
        description="Extracted specifics from domain context (region, species, etc.)",
    )
    context_keywords: list[str] = Field(
        default_factory=list,
        description="Keywords capturing user's specific context for relevance filtering",
    )


class AgentFinding(BaseModel):
    """A source agent's search results and rationale."""

    agent_name: str
    source_type: str  # "semantic_scholar", "web_search", "llm_deep_research"
    papers: list[LiteratureEvidence] = Field(default_factory=list)
    rationale: str = ""
    search_metadata: dict[str, Any] = Field(default_factory=dict)


class DeliberationResult(BaseModel):
    """Output of the moderator deliberation."""

    consensus_papers: list[LiteratureEvidence] = Field(default_factory=list)
    excluded_papers: list[LiteratureEvidence] = Field(default_factory=list)
    moderator_rationale: str = ""
    agent_findings: list[AgentFinding] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class PipelineResult(BaseModel):
    """Complete result for one parameter."""

    parameter: ParameterInput
    prior: FittedPrior
    search_queries: list[str] = Field(default_factory=list)
    papers_found: int = 0
    values_extracted: int = 0
    warnings: list[str] = Field(default_factory=list)
    enrichment: EnrichedContext | None = None
    deliberation: DeliberationResult | None = None


class BatchResult(BaseModel):
    """Results for a batch of parameters."""

    results: list[PipelineResult] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

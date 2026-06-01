"""Pydantic models for Distribird data structures."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


class ParameterValidity(str, Enum):
    """Verdict on whether a parameter is a real, empirically-measured scientific quantity."""

    VALID = "valid"
    SUSPICIOUS = "suspicious"
    LIKELY_INVALID = "likely_invalid"
    UNKNOWN = "unknown"


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
    context: str | None = ""


class LiteratureEvidence(BaseModel):
    """Evidence from a single paper."""

    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    doi: str | None = None
    abstract: str = ""
    full_text: str = Field(default="", description="Full paper text (if available)")
    pdf_url: str | None = None
    extracted_values: list[ExtractedValue] = Field(default_factory=list)
    relevance_score: float = Field(0.0, ge=0.0, le=1.0)
    relevance_snippet: str = Field(default="", description="Most relevant abstract snippet")
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


class CredibleIntervalCoverage(BaseModel):
    """Fraction of extracted data falling within credible intervals of the fitted prior."""

    ci_50: float = Field(..., ge=0.0, le=1.0, description="Fraction within 50% CI")
    ci_90: float = Field(..., ge=0.0, le=1.0, description="Fraction within 90% CI")
    ci_95: float = Field(..., ge=0.0, le=1.0, description="Fraction within 95% CI")


class ModelCheckResult(BaseModel):
    """Goodness-of-fit diagnostics for a fitted prior distribution."""

    map_estimate: float = Field(..., description="Mode (MAP) of the fitted distribution")
    dist_mean: float = Field(..., description="Mean of the fitted distribution")
    dist_median: float = Field(..., description="Median of the fitted distribution")
    dist_variance: float = Field(..., description="Variance of the fitted distribution")
    ci_95_lower: float = Field(..., description="2.5th percentile of the fitted distribution")
    ci_95_upper: float = Field(..., description="97.5th percentile of the fitted distribution")
    ks_statistic: float = Field(..., description="Kolmogorov-Smirnov test statistic")
    ks_pvalue: float = Field(..., description="Kolmogorov-Smirnov test p-value")
    log_likelihood: float = Field(
        ..., description="Log-likelihood of data under fitted distribution"
    )
    aic: float = Field(..., description="Akaike Information Criterion (2k - 2*ll)")
    mean_absolute_cdf_deviation: float = Field(
        ..., description="Mean |F_empirical(x) - F_fitted(x)| over data points"
    )
    credible_interval_coverage: CredibleIntervalCoverage
    n_values: int = Field(..., description="Number of data points used for evaluation")


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
        default="",
        description="Extracted specifics from domain context (region, species, etc.)",
    )
    context_keywords: list[str] = Field(
        default_factory=list,
        description="Keywords capturing user's specific context for relevance filtering",
    )
    is_recognized_parameter: bool | None = Field(
        default=None,
        description="LLM self-report: did it recognize this as a real scientific parameter?",
    )
    recognition_confidence: Literal["high", "medium", "low", "none"] = Field(
        default="none",
        description="LLM self-rated confidence in recognizing the parameter",
    )
    empirically_measured: bool | None = Field(
        default=None,
        description="LLM self-report: is this parameter empirically measured (vs theoretical)?",
    )

    @field_validator("recognition_confidence", mode="before")
    @classmethod
    def _coerce_recognition_confidence(cls, v: object) -> str:
        if isinstance(v, str) and v in {"high", "medium", "low", "none"}:
            return v
        return "none"


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
    model_check: ModelCheckResult | None = None
    parameter_validity: ParameterValidity = Field(
        default=ParameterValidity.UNKNOWN,
        description="Validity verdict for the requested parameter",
    )
    validity_reason: str = Field(
        default="", description="Human-readable explanation of validity verdict"
    )
    validity_signals: dict[str, Any] = Field(
        default_factory=dict,
        description="Raw signals that fed into the validity verdict",
    )
    is_empirical: bool | None = Field(
        default=None,
        description="Whether the parameter is empirically measurable (None = not assessed)",
    )
    token_usage: dict[str, int] = Field(
        default_factory=dict,
        description="LLM token totals across all calls for this parameter "
        "(keys: prompt_tokens, completion_tokens, total_tokens, n_calls)",
    )
    debug_trace: dict[str, Any] | None = Field(
        default=None,
        description="Full structured execution trace when debug_trace is enabled "
        "(None otherwise). Schema: see agent/diagnostics.RunTrace.to_dict().",
    )


class BatchResult(BaseModel):
    """Results for a batch of parameters."""

    results: list[PipelineResult] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

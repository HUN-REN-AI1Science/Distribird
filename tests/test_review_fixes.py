"""Regression tests for the correctness fixes from the code review."""

from __future__ import annotations

import json

import numpy as np

from distribird.agent.deliberation import _as_int_list, _deduplicate_across_agents
from distribird.agent.extract import _parse_int
from distribird.agent.search import _compute_relevance
from distribird.agent.synthesize import synthesize_prior
from distribird.distributions.fitting import (
    _fit_beta,
    _fit_normal,
    _truncation_bounds,
    values_to_prior,
)
from distribird.distributions.model_check import check_model
from distribird.export._util import comment_safe, safe_identifier
from distribird.export.json_export import export_single_json
from distribird.models import (
    AgentFinding,
    ConfidenceLevel,
    ConstraintSpec,
    DistributionFamily,
    ExtractedValue,
    FittedPrior,
    LiteratureEvidence,
    ParameterInput,
    PipelineResult,
)

# --- export: identifier / comment sanitization ---------------------------------


def test_safe_identifier_handles_illegal_chars_and_leading_digit():
    assert safe_identifier("Vcmax (25C)") == "Vcmax__25C_"
    assert safe_identifier("3PG_rate") == "_3PG_rate"
    assert safe_identifier("leaf-area/index%") == "leaf_area_index_"
    # Result is always a valid Python identifier.
    assert safe_identifier("k (d^-1)").isidentifier()


def test_comment_safe_collapses_newlines():
    assert comment_safe("Maize growth\nunder drought") == "Maize growth under drought"
    assert "\n" not in comment_safe("a\r\nb\tc   d")


# --- export: JSON must not emit NaN/Infinity -----------------------------------


def test_json_export_strips_non_finite():
    prior = FittedPrior(
        parameter_name="x",
        family=DistributionFamily.NORMAL,
        params={"mu": float("inf"), "sigma": float("nan")},
        confidence=ConfidenceLevel.LOW,
        is_informative=True,
    )
    result = PipelineResult(parameter=ParameterInput(name="x", description="d"), prior=prior)
    s = export_single_json(result)
    assert "NaN" not in s and "Infinity" not in s
    parsed = json.loads(s)
    assert parsed["params"]["mu"] is None
    assert parsed["params"]["sigma"] is None


# --- extraction: sample_size coercion ------------------------------------------


def test_parse_int_tolerates_messy_values():
    assert _parse_int("approximately 30") == 30
    assert _parse_int(36.0) == 36
    assert _parse_int(36.5) == 36
    assert _parse_int("n=12") == 12
    assert _parse_int(True) is None
    assert _parse_int("none") is None
    assert _parse_int(None) is None


def test_extracted_value_accepts_string_sample_size_after_parse():
    # Building an ExtractedValue with a parsed sample size must not raise.
    ev = ExtractedValue(reported_value=5.0, sample_size=_parse_int("approximately 30"))
    assert ev.sample_size == 30


# --- fitting: truncated-normal bounds never invert -----------------------------


def test_truncation_bounds_never_invert():
    # Value below an inferred lower bound: the auto upper end must stay above lb.
    lb, ub = _truncation_bounds(-5.0, 0.1, 0.0, None)
    assert lb < ub
    lb, ub = _truncation_bounds(99.0, 0.1, None, 10.0)
    assert lb < ub


def test_single_value_outside_inferred_bound_yields_finite_model_check():
    prior = values_to_prior("x", [-5.0], lower_bound=0.0, upper_bound=None)
    assert prior.params["a"] < prior.params["b"]
    mc = check_model(prior, [-5.0])
    assert mc is not None
    assert np.isfinite([mc.dist_mean, mc.dist_median, mc.ci_95_lower, mc.ci_95_upper]).all()
    assert np.isfinite(mc.map_estimate)


# --- fitting: beta AIC is comparable to the other families ---------------------


def test_beta_aic_on_raw_scale_is_comparable():
    vals = np.array([20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
    beta = _fit_beta(vals, 0.0, 100.0)
    normal = _fit_normal(vals)
    assert beta is not None and normal is not None
    # On the raw scale the two AICs are within a few points; the old scaled-data
    # beta AIC was ~65 points too low (range width 100 -> n*log(100)).
    assert abs(beta.aic - normal.aic) < 10


# --- deliberation: index coercion + DOI-less dedup -----------------------------


def test_as_int_list_coerces_and_drops_junk():
    assert _as_int_list([1, 2, 3]) == [1, 2, 3]
    assert _as_int_list(["1", "x", 2.0, None, True]) == [1, 2]
    assert _as_int_list(3) == []
    assert _as_int_list(None) == []


def test_deduplicate_merges_doiless_papers_by_title():
    paper_a = LiteratureEvidence(title="Same Paper", doi=None)
    paper_b = LiteratureEvidence(title="same paper", doi=None)  # different casing
    findings = [
        AgentFinding(agent_name="web", source_type="web_search", papers=[paper_a]),
        AgentFinding(agent_name="deep", source_type="llm_deep_research", papers=[paper_b]),
    ]
    all_papers, sources = _deduplicate_across_agents(findings)
    assert len(all_papers) == 1
    assert set(sources[0]) == {"web", "deep"}


# --- search: recency bonus clamped for future-dated papers ---------------------


def test_recency_bonus_capped_for_future_year():
    score = _compute_relevance(citation_count=0, year=3000)  # far-future preprint
    assert abs(score - 0.2) < 1e-9  # exactly the +0.2 cap, not inflated


# --- synthesize: weight alignment survives a NaN value -------------------------


def test_synthesize_aligns_weights_without_desync():
    papers = [
        LiteratureEvidence(
            title="A", extracted_values=[ExtractedValue(reported_value=5.0, sample_size=4)]
        ),
        LiteratureEvidence(
            title="B", extracted_values=[ExtractedValue(reported_value=7.0, sample_size=9)]
        ),
    ]
    prior = synthesize_prior(
        ParameterInput(name="p", description="d", constraints=ConstraintSpec()), papers
    )
    assert prior.is_informative
    assert prior.n_sources == 2

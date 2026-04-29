"""Tests for distribird.distributions.model_check."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats  # type: ignore[import-untyped]

from distribird.distributions.model_check import (
    _build_scipy_dist,
    _compute_cdf_deviation,
    _compute_credible_coverage,
    _compute_map,
    check_batch,
    check_model,
    check_model_from_result,
)
from distribird.models import (
    BatchResult,
    ConfidenceLevel,
    ConstraintSpec,
    DistributionFamily,
    ExtractedValue,
    FittedPrior,
    LiteratureEvidence,
    ParameterInput,
    PipelineResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_prior(
    family: DistributionFamily,
    params: dict[str, float],
    informative: bool = True,
    evidence: list[LiteratureEvidence] | None = None,
) -> FittedPrior:
    return FittedPrior(
        parameter_name="test_param",
        family=family,
        params=params,
        confidence=ConfidenceLevel.HIGH if informative else ConfidenceLevel.NONE,
        is_informative=informative,
        reason="test",
        evidence=evidence or [],
        n_sources=len(evidence) if evidence else 0,
    )


# ---------------------------------------------------------------------------
# _build_scipy_dist
# ---------------------------------------------------------------------------


class TestBuildScipyDist:
    def test_normal(self):
        dist = _build_scipy_dist(DistributionFamily.NORMAL, {"mu": 5.0, "sigma": 2.0})
        assert abs(dist.mean() - 5.0) < 1e-6
        assert abs(dist.std() - 2.0) < 1e-6

    def test_truncated_normal(self):
        dist = _build_scipy_dist(
            DistributionFamily.TRUNCATED_NORMAL,
            {"mu": 0.0, "sigma": 1.0, "a": -2.0, "b": 2.0},
        )
        # Mean of symmetric truncated normal is 0
        assert abs(dist.mean()) < 0.1

    def test_gamma(self):
        dist = _build_scipy_dist(DistributionFamily.GAMMA, {"alpha": 3.0, "scale": 2.0})
        assert abs(dist.mean() - 6.0) < 1e-6

    def test_lognormal(self):
        dist = _build_scipy_dist(DistributionFamily.LOGNORMAL, {"mu": 0.0, "sigma": 1.0})
        expected_mean = math.exp(0.0 + 0.5)
        assert abs(dist.mean() - expected_mean) < 1e-4

    def test_beta(self):
        dist = _build_scipy_dist(
            DistributionFamily.BETA,
            {"alpha": 2.0, "beta": 5.0, "lower": 0.0, "upper": 1.0},
        )
        expected_mean = 2.0 / (2.0 + 5.0)
        assert abs(dist.mean() - expected_mean) < 1e-4

    def test_beta_scaled(self):
        dist = _build_scipy_dist(
            DistributionFamily.BETA,
            {"alpha": 2.0, "beta": 2.0, "lower": 10.0, "upper": 20.0},
        )
        # Mean should be midpoint for symmetric beta
        assert abs(dist.mean() - 15.0) < 1e-4

    def test_uniform(self):
        dist = _build_scipy_dist(DistributionFamily.UNIFORM, {"lower": 3.0, "upper": 7.0})
        assert abs(dist.mean() - 5.0) < 1e-6


# ---------------------------------------------------------------------------
# _compute_map
# ---------------------------------------------------------------------------


class TestComputeMap:
    def test_normal(self):
        assert _compute_map(DistributionFamily.NORMAL, {"mu": 3.14, "sigma": 1.0}) == 3.14

    def test_truncated_normal_inside(self):
        result = _compute_map(
            DistributionFamily.TRUNCATED_NORMAL,
            {"mu": 5.0, "sigma": 1.0, "a": 0.0, "b": 10.0},
        )
        assert result == 5.0

    def test_truncated_normal_clipped_low(self):
        result = _compute_map(
            DistributionFamily.TRUNCATED_NORMAL,
            {"mu": -5.0, "sigma": 1.0, "a": 0.0, "b": 10.0},
        )
        assert result == 0.0

    def test_truncated_normal_clipped_high(self):
        result = _compute_map(
            DistributionFamily.TRUNCATED_NORMAL,
            {"mu": 15.0, "sigma": 1.0, "a": 0.0, "b": 10.0},
        )
        assert result == 10.0

    def test_gamma_alpha_gt_1(self):
        result = _compute_map(DistributionFamily.GAMMA, {"alpha": 3.0, "scale": 2.0})
        assert abs(result - 4.0) < 1e-10

    def test_gamma_alpha_lt_1(self):
        # alpha < 1: mode is degenerate (PDF → ∞ at 0), so we use the median
        result = _compute_map(DistributionFamily.GAMMA, {"alpha": 0.5, "scale": 2.0})
        expected = float(stats.gamma(a=0.5, scale=2.0).median())
        assert result == pytest.approx(expected)
        assert result > 0  # must be positive and evaluable

    def test_lognormal(self):
        result = _compute_map(DistributionFamily.LOGNORMAL, {"mu": 1.0, "sigma": 0.5})
        expected = math.exp(1.0 - 0.25)
        assert abs(result - expected) < 1e-10

    def test_beta_standard(self):
        result = _compute_map(
            DistributionFamily.BETA,
            {"alpha": 3.0, "beta": 5.0, "lower": 0.0, "upper": 1.0},
        )
        expected = (3.0 - 1.0) / (3.0 + 5.0 - 2.0)
        assert abs(result - expected) < 1e-10

    def test_beta_scaled(self):
        result = _compute_map(
            DistributionFamily.BETA,
            {"alpha": 3.0, "beta": 3.0, "lower": 10.0, "upper": 20.0},
        )
        assert abs(result - 15.0) < 1e-10

    def test_beta_alpha_le_1(self):
        # alpha <= 1: mode at boundary where PDF diverges, use median instead
        result = _compute_map(
            DistributionFamily.BETA,
            {"alpha": 0.5, "beta": 2.0, "lower": 0.0, "upper": 1.0},
        )
        expected = float(stats.beta(a=0.5, b=2.0).median())
        assert result == pytest.approx(expected)
        assert 0 < result < 1  # must be interior and evaluable

    def test_beta_beta_le_1(self):
        # beta <= 1: mode at upper boundary where PDF diverges, use median instead
        result = _compute_map(
            DistributionFamily.BETA,
            {"alpha": 2.0, "beta": 0.5, "lower": 0.0, "upper": 1.0},
        )
        expected = float(stats.beta(a=2.0, b=0.5).median())
        assert result == pytest.approx(expected)
        assert 0 < result < 1

    def test_uniform(self):
        result = _compute_map(DistributionFamily.UNIFORM, {"lower": 5.0, "upper": 15.0})
        assert result == 10.0


# ---------------------------------------------------------------------------
# _compute_credible_coverage
# ---------------------------------------------------------------------------


class TestCredibleCoverage:
    def test_all_inside(self):
        dist = stats.norm(loc=0, scale=10)
        values = np.array([0.0, 1.0, -1.0])
        cov = _compute_credible_coverage(dist, values)
        assert cov.ci_50 > 0.0
        assert cov.ci_90 > 0.0
        assert cov.ci_95 > 0.0

    def test_all_outside_50(self):
        dist = stats.norm(loc=0, scale=1)
        # Values well outside the 50% CI but inside 95%
        values = np.array([-1.5, 1.5])
        cov = _compute_credible_coverage(dist, values)
        assert cov.ci_50 == 0.0
        assert cov.ci_95 == 1.0

    def test_monotonic(self):
        dist = stats.norm(loc=0, scale=1)
        rng = np.random.default_rng(42)
        values = rng.normal(0, 1, size=1000)
        cov = _compute_credible_coverage(dist, values)
        assert cov.ci_50 <= cov.ci_90 <= cov.ci_95


# ---------------------------------------------------------------------------
# _compute_cdf_deviation
# ---------------------------------------------------------------------------


class TestCdfDeviation:
    def test_perfect_fit(self):
        # Large sample from the true distribution → small deviation
        dist = stats.norm(loc=0, scale=1)
        rng = np.random.default_rng(42)
        values = rng.normal(0, 1, size=10000)
        dev = _compute_cdf_deviation(dist, values)
        assert dev < 0.02

    def test_mismatched(self):
        # Data from N(0,1) tested against N(10,1) → large deviation
        dist = stats.norm(loc=10, scale=1)
        values = np.array([0.0, 0.5, 1.0, -0.5, -1.0])
        dev = _compute_cdf_deviation(dist, values)
        assert dev > 0.5


# ---------------------------------------------------------------------------
# check_model
# ---------------------------------------------------------------------------


class TestCheckModel:
    def test_normal_good_fit(self):
        """Data drawn from the same normal → high KS p-value."""
        prior = _make_prior(DistributionFamily.NORMAL, {"mu": 5.0, "sigma": 2.0})
        rng = np.random.default_rng(42)
        values = rng.normal(5.0, 2.0, size=50).tolist()
        mc = check_model(prior, values)
        assert mc is not None
        assert mc.map_estimate == 5.0
        assert mc.ks_pvalue > 0.05
        assert mc.n_values == 50

    def test_single_value(self):
        """Single data point still produces a result."""
        prior = _make_prior(DistributionFamily.NORMAL, {"mu": 3.0, "sigma": 1.0})
        mc = check_model(prior, [3.0])
        assert mc is not None
        assert mc.n_values == 1
        assert mc.map_estimate == 3.0

    def test_empty_values_returns_none(self):
        prior = _make_prior(DistributionFamily.NORMAL, {"mu": 0, "sigma": 1})
        assert check_model(prior, []) is None

    def test_non_informative_returns_none(self):
        prior = _make_prior(
            DistributionFamily.UNIFORM,
            {"lower": 0.0, "upper": 100.0},
            informative=False,
        )
        assert check_model(prior, [1.0, 2.0, 3.0]) is None

    def test_gamma(self):
        prior = _make_prior(DistributionFamily.GAMMA, {"alpha": 3.0, "scale": 2.0})
        rng = np.random.default_rng(42)
        values = stats.gamma.rvs(a=3.0, scale=2.0, size=30, random_state=rng).tolist()
        mc = check_model(prior, values)
        assert mc is not None
        assert mc.map_estimate == pytest.approx(4.0)
        assert mc.ks_pvalue > 0.05

    def test_truncated_normal(self):
        prior = _make_prior(
            DistributionFamily.TRUNCATED_NORMAL,
            {"mu": 5.0, "sigma": 2.0, "a": 0.0, "b": 10.0},
        )
        mc = check_model(prior, [3.0, 5.0, 7.0, 4.0, 6.0])
        assert mc is not None
        assert mc.map_estimate == 5.0
        assert 0.0 <= mc.credible_interval_coverage.ci_95 <= 1.0

    def test_lognormal(self):
        prior = _make_prior(DistributionFamily.LOGNORMAL, {"mu": 1.0, "sigma": 0.5})
        rng = np.random.default_rng(42)
        values = stats.lognorm.rvs(s=0.5, scale=math.exp(1.0), size=40, random_state=rng).tolist()
        mc = check_model(prior, values)
        assert mc is not None
        assert mc.ks_pvalue > 0.05

    def test_beta(self):
        prior = _make_prior(
            DistributionFamily.BETA,
            {"alpha": 2.0, "beta": 5.0, "lower": 0.0, "upper": 1.0},
        )
        rng = np.random.default_rng(42)
        values = stats.beta.rvs(a=2.0, b=5.0, size=30, random_state=rng).tolist()
        mc = check_model(prior, values)
        assert mc is not None
        assert mc.ks_pvalue > 0.05

    def test_aic_computation(self):
        prior = _make_prior(DistributionFamily.NORMAL, {"mu": 0.0, "sigma": 1.0})
        mc = check_model(prior, [0.0, 0.5, -0.5])
        assert mc is not None
        expected_aic = 2 * 2 - 2 * mc.log_likelihood
        assert mc.aic == pytest.approx(expected_aic)

    def test_coverage_monotonic(self):
        prior = _make_prior(DistributionFamily.NORMAL, {"mu": 0.0, "sigma": 1.0})
        rng = np.random.default_rng(42)
        values = rng.normal(0, 1, size=100).tolist()
        mc = check_model(prior, values)
        assert mc is not None
        cov = mc.credible_interval_coverage
        assert cov.ci_50 <= cov.ci_90 <= cov.ci_95


# ---------------------------------------------------------------------------
# Offline utilities
# ---------------------------------------------------------------------------


class TestCheckModelFromResult:
    def test_with_evidence(self):
        evidence = [
            LiteratureEvidence(
                title="Paper A",
                extracted_values=[
                    ExtractedValue(reported_value=5.0),
                    ExtractedValue(reported_value=6.0),
                ],
            ),
            LiteratureEvidence(
                title="Paper B",
                extracted_values=[
                    ExtractedValue(reported_value=4.5),
                ],
            ),
        ]
        prior = _make_prior(
            DistributionFamily.NORMAL,
            {"mu": 5.0, "sigma": 1.0},
            evidence=evidence,
        )
        result = PipelineResult(
            parameter=ParameterInput(name="test", description="test param"),
            prior=prior,
        )
        mc = check_model_from_result(result)
        assert mc is not None
        assert mc.n_values == 3

    def test_no_evidence_returns_none(self):
        prior = _make_prior(DistributionFamily.NORMAL, {"mu": 0, "sigma": 1})
        result = PipelineResult(
            parameter=ParameterInput(name="test", description="test param"),
            prior=prior,
        )
        mc = check_model_from_result(result)
        assert mc is None

    def test_with_constraints(self):
        evidence = [
            LiteratureEvidence(
                title="Paper",
                extracted_values=[
                    ExtractedValue(reported_value=5.0),
                    ExtractedValue(reported_value=100.0),  # out of bounds
                    ExtractedValue(reported_value=7.0),
                ],
            ),
        ]
        prior = _make_prior(
            DistributionFamily.NORMAL,
            {"mu": 6.0, "sigma": 2.0},
            evidence=evidence,
        )
        result = PipelineResult(
            parameter=ParameterInput(
                name="test",
                description="test param",
                constraints=ConstraintSpec(lower_bound=0.0, upper_bound=20.0),
            ),
            prior=prior,
        )
        mc = check_model_from_result(result)
        assert mc is not None
        assert mc.n_values == 2  # 100.0 excluded


class TestCheckBatch:
    def test_batch(self):
        evidence = [
            LiteratureEvidence(
                title="Paper",
                extracted_values=[ExtractedValue(reported_value=5.0)],
            ),
        ]
        prior = _make_prior(
            DistributionFamily.NORMAL,
            {"mu": 5.0, "sigma": 1.0},
            evidence=evidence,
        )
        batch = BatchResult(
            results=[
                PipelineResult(
                    parameter=ParameterInput(name="p1", description="desc"),
                    prior=prior,
                ),
            ]
        )
        results = check_batch(batch)
        assert len(results) == 1
        name, mc = results[0]
        assert name == "p1"
        assert mc is not None

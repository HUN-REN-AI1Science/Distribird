"""Tests for distribution fitting."""

import numpy as np
import pytest

from distribird.distributions.fitting import (
    fit_distribution,
    moment_match_normal,
    values_to_prior,
)
from distribird.models import ConfidenceLevel, DistributionFamily


class TestFitDistribution:
    def test_empty_values(self):
        assert fit_distribution([]) is None

    def test_normal_data(self):
        rng = np.random.default_rng(42)
        values = rng.normal(10, 2, 50).tolist()
        result = fit_distribution(values)
        assert result is not None
        assert result.family in (DistributionFamily.NORMAL, DistributionFamily.TRUNCATED_NORMAL)

    def test_gamma_data(self):
        rng = np.random.default_rng(42)
        values = rng.gamma(2, 3, 50).tolist()
        result = fit_distribution(values, lower_bound=0)
        assert result is not None

    def test_bounded_data(self):
        rng = np.random.default_rng(42)
        values = rng.uniform(2, 8, 50).tolist()
        result = fit_distribution(values, lower_bound=0, upper_bound=10)
        assert result is not None

    def test_single_value(self):
        result = fit_distribution([5.0])
        assert result is not None

    def test_weighted_fitting(self):
        """Weighted fitting should shift the result toward heavily-weighted values."""
        values = [2.0, 8.0, 8.0, 8.0, 8.0, 8.0]
        # Without weights: mean ≈ 7.0
        unweighted = fit_distribution(values)
        # With heavy weight on 2.0
        weighted = fit_distribution(values, weights=[10.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        assert unweighted is not None
        assert weighted is not None
        # Weighted mean should be lower
        assert weighted.params.get("mu", 0) < unweighted.params.get("mu", 0)


class TestMomentMatch:
    def test_basic(self):
        result = moment_match_normal([3.0, 5.0, 7.0])
        assert result.family == DistributionFamily.TRUNCATED_NORMAL
        assert abs(result.params["mu"] - 5.0) < 1e-10

    def test_with_bounds(self):
        result = moment_match_normal([3.0, 5.0], lower_bound=0, upper_bound=10)
        assert result.params["a"] == 0
        assert result.params["b"] == 10

    def test_widened_sigma(self):
        values = [5.0, 5.1, 4.9]
        result = moment_match_normal(values, widen_factor=2.0)
        raw_sd = float(np.std(values, ddof=1))
        assert result.params["sigma"] >= raw_sd

    def test_weighted_moment_match(self):
        """Weighted moment matching should shift mean toward heavier-weighted values."""
        values = [2.0, 8.0, 8.0]
        unweighted = moment_match_normal(values)
        weighted = moment_match_normal(values, weights=[10.0, 1.0, 1.0])
        # Weighted mean should be closer to 2.0
        assert weighted.params["mu"] < unweighted.params["mu"]


class TestValuesToPrior:
    def test_zero_values(self):
        prior = values_to_prior("test", [])
        assert prior.confidence == ConfidenceLevel.NONE
        assert not prior.is_informative

    def test_one_value(self):
        prior = values_to_prior("test", [5.0])
        assert prior.confidence == ConfidenceLevel.LOW
        assert prior.is_informative

    def test_few_values(self):
        prior = values_to_prior("test", [3.0, 5.0, 7.0])
        assert prior.confidence == ConfidenceLevel.MEDIUM
        assert prior.is_informative

    def test_many_values(self):
        rng = np.random.default_rng(42)
        values = rng.normal(25, 5, 20).tolist()
        prior = values_to_prior("test", values, lower_bound=0, upper_bound=50)
        assert prior.confidence == ConfidenceLevel.HIGH
        assert prior.is_informative

    def test_with_bounds(self):
        prior = values_to_prior("test", [5.0], lower_bound=0, upper_bound=10)
        assert prior.params.get("a", prior.params.get("lower", None)) is not None

    def test_single_value_with_uncertainty(self):
        """Tier 1: when uncertainty is available, use it as sigma."""
        prior = values_to_prior(
            "test",
            [5.0],
            uncertainties=[0.3],
        )
        assert prior.params["sigma"] == 0.3
        assert prior.confidence == ConfidenceLevel.LOW

    def test_single_value_without_uncertainty_uses_default(self):
        """Tier 1: without uncertainty, fallback to abs(center) * 0.5."""
        prior = values_to_prior("test", [5.0])
        assert prior.params["sigma"] == pytest.approx(2.5)

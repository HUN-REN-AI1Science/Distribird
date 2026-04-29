"""Model checking: MAP estimation, goodness-of-fit diagnostics, and offline batch utilities."""

from __future__ import annotations

import math

import numpy as np
from scipy import stats  # type: ignore[import-untyped]

from distribird.models import (
    BatchResult,
    CredibleIntervalCoverage,
    DistributionFamily,
    FittedPrior,
    ModelCheckResult,
    PipelineResult,
)

# ---------------------------------------------------------------------------
# Build a frozen scipy distribution from FittedPrior params
# ---------------------------------------------------------------------------


def _build_scipy_dist(
    family: DistributionFamily, params: dict[str, float]
) -> stats.rv_continuous | stats.rv_discrete:
    """Convert a DistributionFamily + params dict to a frozen scipy distribution."""
    if family == DistributionFamily.NORMAL:
        return stats.norm(loc=params["mu"], scale=params["sigma"])

    if family == DistributionFamily.TRUNCATED_NORMAL:
        mu, sigma = params["mu"], params["sigma"]
        a_std = (params["a"] - mu) / sigma
        b_std = (params["b"] - mu) / sigma
        return stats.truncnorm(a_std, b_std, loc=mu, scale=sigma)

    if family == DistributionFamily.GAMMA:
        return stats.gamma(a=params["alpha"], scale=params["scale"])

    if family == DistributionFamily.LOGNORMAL:
        # scipy lognorm: s=sigma, scale=exp(mu)
        return stats.lognorm(s=params["sigma"], scale=math.exp(params["mu"]))

    if family == DistributionFamily.BETA:
        lower = params.get("lower", 0.0)
        upper = params.get("upper", 1.0)
        return stats.beta(
            a=params["alpha"],
            b=params["beta"],
            loc=lower,
            scale=upper - lower,
        )

    if family == DistributionFamily.UNIFORM:
        lower = params.get("lower", params.get("a", 0.0))
        upper = params.get("upper", params.get("b", 1.0))
        return stats.uniform(loc=lower, scale=upper - lower)

    raise ValueError(f"Unsupported distribution family: {family}")


# ---------------------------------------------------------------------------
# Analytical MAP (mode) computation
# ---------------------------------------------------------------------------


def _compute_map(family: DistributionFamily, params: dict[str, float]) -> float:
    """Compute the analytical mode (MAP estimate) for a distribution."""
    if family == DistributionFamily.NORMAL:
        return params["mu"]

    if family == DistributionFamily.TRUNCATED_NORMAL:
        return float(np.clip(params["mu"], params["a"], params["b"]))

    if family == DistributionFamily.GAMMA:
        alpha = params["alpha"]
        scale = params["scale"]
        if alpha >= 1.0:
            return (alpha - 1.0) * scale
        # alpha < 1: PDF diverges at 0, mode is degenerate.
        # Use the median as a practical point estimate.
        return float(stats.gamma(a=alpha, scale=scale).median())

    if family == DistributionFamily.LOGNORMAL:
        mu, sigma = params["mu"], params["sigma"]
        return math.exp(mu - sigma**2)

    if family == DistributionFamily.BETA:
        alpha = params["alpha"]
        beta_param = params["beta"]
        lower = params.get("lower", 0.0)
        upper = params.get("upper", 1.0)
        if alpha > 1.0 and beta_param > 1.0:
            raw_mode = (alpha - 1.0) / (alpha + beta_param - 2.0)
            return lower + (upper - lower) * raw_mode
        # When alpha <= 1 or beta <= 1, the PDF diverges at a boundary —
        # the mode is degenerate. Use the median as a practical point estimate.
        dist = stats.beta(a=alpha, b=beta_param, loc=lower, scale=upper - lower)
        return float(dist.median())

    if family == DistributionFamily.UNIFORM:
        lower = params.get("lower", params.get("a", 0.0))
        upper = params.get("upper", params.get("b", 1.0))
        return (lower + upper) / 2.0

    raise ValueError(f"Unsupported distribution family: {family}")


# ---------------------------------------------------------------------------
# Credible interval coverage
# ---------------------------------------------------------------------------


def _compute_credible_coverage(
    dist: stats.rv_continuous, values: np.ndarray
) -> CredibleIntervalCoverage:
    """Compute fraction of data within 50%, 90%, 95% credible intervals."""
    coverages = {}
    for level, label in [(0.50, "ci_50"), (0.90, "ci_90"), (0.95, "ci_95")]:
        lo = dist.ppf((1.0 - level) / 2.0)
        hi = dist.ppf((1.0 + level) / 2.0)
        frac = float(np.mean((values >= lo) & (values <= hi)))
        coverages[label] = frac
    return CredibleIntervalCoverage(**coverages)


# ---------------------------------------------------------------------------
# CDF deviation
# ---------------------------------------------------------------------------


def _compute_cdf_deviation(dist: stats.rv_continuous, values: np.ndarray) -> float:
    """Mean absolute deviation between empirical CDF and fitted CDF."""
    n = len(values)
    sorted_vals = np.sort(values)
    ecdf = np.arange(1, n + 1) / n
    fcdf = dist.cdf(sorted_vals)
    return float(np.mean(np.abs(ecdf - fcdf)))


# ---------------------------------------------------------------------------
# Main check_model entry point
# ---------------------------------------------------------------------------


def check_model(prior: FittedPrior, values: list[float]) -> ModelCheckResult | None:
    """Compute goodness-of-fit diagnostics for a fitted prior against extracted values.

    Returns None when there are no values or the prior is non-informative.
    """
    if not values or not prior.is_informative:
        return None

    arr = np.array(values, dtype=float)
    n = len(arr)

    dist = _build_scipy_dist(prior.family, prior.params)
    map_estimate = _compute_map(prior.family, prior.params)

    # Summary statistics
    dist_mean = float(dist.mean())
    dist_median = float(dist.median())
    dist_var = float(dist.var())
    ci_lower = float(dist.ppf(0.025))
    ci_upper = float(dist.ppf(0.975))

    # KS test
    ks_stat, ks_p = stats.kstest(arr, dist.cdf)

    # Log-likelihood
    logpdf_vals = dist.logpdf(arr)
    # Guard against -inf from values outside support
    logpdf_vals = np.where(np.isfinite(logpdf_vals), logpdf_vals, -1e10)
    ll = float(np.sum(logpdf_vals))

    # AIC: 2 parameters for all currently supported families
    k = 2
    aic = 2.0 * k - 2.0 * ll

    # Coverage and CDF deviation
    coverage = _compute_credible_coverage(dist, arr)
    cdf_dev = _compute_cdf_deviation(dist, arr)

    return ModelCheckResult(
        map_estimate=map_estimate,
        dist_mean=dist_mean,
        dist_median=dist_median,
        dist_variance=dist_var,
        ci_95_lower=ci_lower,
        ci_95_upper=ci_upper,
        ks_statistic=float(ks_stat),
        ks_pvalue=float(ks_p),
        log_likelihood=ll,
        aic=aic,
        mean_absolute_cdf_deviation=cdf_dev,
        credible_interval_coverage=coverage,
        n_values=n,
    )


# ---------------------------------------------------------------------------
# Offline utilities for existing BatchResult JSON
# ---------------------------------------------------------------------------


def check_model_from_result(result: PipelineResult) -> ModelCheckResult | None:
    """Compute model check from an existing PipelineResult without re-running the pipeline.

    Extracts values from the evidence attached to the fitted prior, filters by
    parameter constraints, and runs check_model.
    """
    from distribird.agent.synthesize import collect_weighted_values
    from distribird.distributions.constraints import filter_values_by_constraints

    papers = result.prior.evidence
    if not papers:
        return None

    weighted_values = collect_weighted_values(papers)
    raw_values = [wv.value for wv in weighted_values]
    valid_values, _ = filter_values_by_constraints(raw_values, result.parameter.constraints)

    return check_model(result.prior, valid_values)


def check_batch(
    batch: BatchResult,
) -> list[tuple[str, ModelCheckResult | None]]:
    """Run model checking on every result in a batch.

    Returns a list of (parameter_name, ModelCheckResult | None) tuples.
    """
    results: list[tuple[str, ModelCheckResult | None]] = []
    for r in batch.results:
        mc = check_model_from_result(r)
        results.append((r.parameter.name, mc))
    return results

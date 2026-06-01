"""Distribution fitting with AIC-based family selection."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy import stats  # type: ignore[import-untyped]

from distribird.agent import diagnostics
from distribird.models import ConfidenceLevel, DistributionFamily, FittedPrior


@dataclass
class FitCandidate:
    family: DistributionFamily
    params: dict[str, float]
    log_likelihood: float
    n_params: int
    aic: float


def _mean_std_floored(values: np.ndarray, weights: np.ndarray | None) -> tuple[float, float]:
    """Mean and standard deviation of ``values`` (weighted when ``weights`` given).

    Applies a small floor so a degenerate (near-zero) sigma never collapses a fit.
    """
    if weights is not None:
        mu = float(np.average(values, weights=weights))
        sigma = float(np.sqrt(np.average((values - mu) ** 2, weights=weights)))
    else:
        mu = float(np.mean(values))
        sigma = float(np.std(values, ddof=1)) if len(values) > 1 else float(np.std(values))
    if sigma < 1e-12:
        sigma = abs(mu) * 0.1 if abs(mu) > 1e-12 else 1.0
    return mu, sigma


def _log_likelihood(logpdf: np.ndarray, weights: np.ndarray | None) -> float:
    """Sum the per-point log-densities, weighted when ``weights`` is provided."""
    return float(np.sum(weights * logpdf)) if weights is not None else float(np.sum(logpdf))


def _truncation_bounds(
    center: float, spread: float, lower_bound: float | None, upper_bound: float | None
) -> tuple[float, float]:
    """Return ``(lb, ub)`` for a truncated normal, guaranteeing ``lb < ub``.

    Auto-computed open ends are anchored to ``center`` clamped inside any fixed
    bound, so a value lying outside a user/enrichment-inferred bound can never
    invert the support (``a_std > b_std``), which otherwise yields NaN
    mean/ppf in ``model_check`` and NaN in the JSON/plot outputs.
    """
    c = center
    if lower_bound is not None:
        c = max(c, lower_bound)
    if upper_bound is not None:
        c = min(c, upper_bound)
    lb = lower_bound if lower_bound is not None else c - 10 * spread
    ub = upper_bound if upper_bound is not None else c + 10 * spread
    if ub <= lb:
        ub = lb + max(abs(spread), 1.0)
    return lb, ub


def _fit_truncated_normal(
    values: np.ndarray, lower: float, upper: float, weights: np.ndarray | None = None
) -> FitCandidate | None:
    """Fit a truncated normal distribution."""
    mu, sigma = _mean_std_floored(values, weights)

    a_std = (lower - mu) / sigma
    b_std = (upper - mu) / sigma
    try:
        logpdf = stats.truncnorm.logpdf(values, a_std, b_std, loc=mu, scale=sigma)
        ll = _log_likelihood(logpdf, weights)
        if not np.isfinite(ll):
            return None
    except Exception:
        return None

    return FitCandidate(
        family=DistributionFamily.TRUNCATED_NORMAL,
        params={"mu": mu, "sigma": sigma, "a": lower, "b": upper},
        log_likelihood=ll,
        n_params=2,
        aic=2 * 2 - 2 * ll,
    )


def _fit_normal(values: np.ndarray, weights: np.ndarray | None = None) -> FitCandidate | None:
    mu, sigma = _mean_std_floored(values, weights)
    try:
        logpdf = stats.norm.logpdf(values, loc=mu, scale=sigma)
        ll = _log_likelihood(logpdf, weights)
        if not np.isfinite(ll):
            return None
    except Exception:
        return None
    return FitCandidate(
        family=DistributionFamily.NORMAL,
        params={"mu": mu, "sigma": sigma},
        log_likelihood=ll,
        n_params=2,
        aic=2 * 2 - 2 * ll,
    )


def _fit_gamma(values: np.ndarray, weights: np.ndarray | None = None) -> FitCandidate | None:
    """Fit a Gamma distribution (positive values only)."""
    if np.any(values <= 0):
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a_shape, loc, scale = stats.gamma.fit(values, floc=0)
        logpdf = stats.gamma.logpdf(values, a_shape, loc=0, scale=scale)
        ll = _log_likelihood(logpdf, weights)
        if not np.isfinite(ll):
            return None
    except Exception:
        return None
    return FitCandidate(
        family=DistributionFamily.GAMMA,
        params={"alpha": float(a_shape), "scale": float(scale)},
        log_likelihood=ll,
        n_params=2,
        aic=2 * 2 - 2 * ll,
    )


def _fit_lognormal(values: np.ndarray, weights: np.ndarray | None = None) -> FitCandidate | None:
    """Fit a Log-Normal distribution (positive values only)."""
    if np.any(values <= 0):
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            shape, loc, scale = stats.lognorm.fit(values, floc=0)
        logpdf = stats.lognorm.logpdf(values, shape, loc=0, scale=scale)
        ll = _log_likelihood(logpdf, weights)
        if not np.isfinite(ll):
            return None
    except Exception:
        return None
    return FitCandidate(
        family=DistributionFamily.LOGNORMAL,
        params={"mu": float(np.log(scale)), "sigma": float(shape)},
        log_likelihood=ll,
        n_params=2,
        aic=2 * 2 - 2 * ll,
    )


def _fit_beta(
    values: np.ndarray, lower: float, upper: float, weights: np.ndarray | None = None
) -> FitCandidate | None:
    """Fit a Beta distribution (values must be within [lower, upper])."""
    if upper <= lower:
        return None
    scaled = (values - lower) / (upper - lower)
    if np.any(scaled <= 0) or np.any(scaled >= 1):
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a, b, loc, scale = stats.beta.fit(scaled, floc=0, fscale=1)
        # Evaluate the log-likelihood on the RAW data scale (loc=lower,
        # scale=upper-lower) so it includes the change-of-variables Jacobian and
        # is comparable to the other families' AICs (all on the raw values).
        # Scoring on the [0,1]-scaled data instead omitted -n*log(upper-lower)
        # and systematically over/under-selected Beta by the range width.
        logpdf = stats.beta.logpdf(values, a, b, loc=lower, scale=upper - lower)
        ll = _log_likelihood(logpdf, weights)
        if not np.isfinite(ll):
            return None
    except Exception:
        return None
    return FitCandidate(
        family=DistributionFamily.BETA,
        params={"alpha": float(a), "beta": float(b), "lower": lower, "upper": upper},
        log_likelihood=ll,
        n_params=2,
        aic=2 * 2 - 2 * ll,
    )


def fit_distribution(
    values: list[float],
    lower_bound: float | None = None,
    upper_bound: float | None = None,
    weights: list[float] | None = None,
) -> FitCandidate | None:
    """Fit best distribution to extracted values using AIC selection.

    Returns the best-fitting candidate, or None if fitting fails.
    """
    arr = np.array(values, dtype=float)
    if len(arr) == 0:
        return None

    w = np.array(weights, dtype=float) if weights is not None else None

    lb = lower_bound if lower_bound is not None else -np.inf
    ub = upper_bound if upper_bound is not None else np.inf

    candidates: list[FitCandidate] = []

    for fitter in [
        lambda: _fit_normal(arr, w),
        lambda: _fit_truncated_normal(arr, lb, ub, w),
        lambda: _fit_gamma(arr, w),
        lambda: _fit_lognormal(arr, w),
        lambda: _fit_beta(arr, lb, ub, w),
    ]:
        result = fitter()  # type: ignore[no-untyped-call]
        if result is not None:
            candidates.append(result)

    if not candidates:
        if diagnostics.enabled():
            diagnostics.record("fitting_candidates", {"n_values": len(arr), "candidates": []})
        return None

    best = min(candidates, key=lambda c: c.aic)

    if diagnostics.enabled():
        ranked = sorted(candidates, key=lambda c: c.aic)
        delta = (ranked[1].aic - ranked[0].aic) if len(ranked) > 1 else None
        diagnostics.record(
            "fitting_candidates",
            {
                "n_values": len(arr),
                "candidates": [
                    {"family": c.family.value, "aic": c.aic, "log_likelihood": c.log_likelihood}
                    for c in ranked
                ],
                "chosen_family": best.family.value,
                "delta_aic_to_second": delta,
            },
        )

    return best


def moment_match_normal(
    values: list[float],
    lower_bound: float | None = None,
    upper_bound: float | None = None,
    widen_factor: float = 1.5,
    weights: list[float] | None = None,
) -> FitCandidate:
    """Moment matching to truncated Normal with widened SD (for 2-4 data points)."""
    arr = np.array(values, dtype=float)

    if weights is not None:
        w = np.array(weights, dtype=float)
        mu = float(np.average(arr, weights=w))
        if len(arr) > 1:
            sigma = float(np.sqrt(np.average((arr - mu) ** 2, weights=w)))
        else:
            sigma = abs(mu) * 0.25
    else:
        mu = float(np.mean(arr))
        sigma = float(np.std(arr, ddof=1)) if len(arr) > 1 else abs(mu) * 0.25

    # Use constraint range to set a sensible minimum sigma floor:
    # when all values cluster tightly (e.g., species-level constants like TBASE=0),
    # a tiny sigma produces an unusable Dirac-like prior.
    if lower_bound is not None and upper_bound is not None:
        range_floor = (upper_bound - lower_bound) * 0.05
    else:
        # No user bounds: use a meaningful floor based on value magnitude.
        # For zero-centered parameters (e.g., TBASE=0°C), abs(mu)*0.05 is also 0,
        # so we need an absolute minimum of 1.0 to produce a usable prior.
        range_floor = max(1.0, abs(mu) * 0.25)
    sigma = max(sigma * widen_factor, abs(mu) * 0.05, range_floor)

    lb, ub = _truncation_bounds(mu, sigma, lower_bound, upper_bound)

    a_std = (lb - mu) / sigma
    b_std = (ub - mu) / sigma
    ll = float(np.sum(stats.truncnorm.logpdf(arr, a_std, b_std, loc=mu, scale=sigma)))

    return FitCandidate(
        family=DistributionFamily.TRUNCATED_NORMAL,
        params={"mu": mu, "sigma": sigma, "a": lb, "b": ub},
        log_likelihood=ll if np.isfinite(ll) else -1e10,
        n_params=2,
        aic=2 * 2 - 2 * ll if np.isfinite(ll) else 1e10,
    )


def values_to_prior(
    parameter_name: str,
    values: list[float],
    lower_bound: float | None = None,
    upper_bound: float | None = None,
    weights: list[float] | None = None,
    uncertainties: list[float | None] | None = None,
) -> FittedPrior:
    """Convert extracted values to a fitted prior using tiered approach.

    - 5+ values: full AIC-based fitting
    - 2-4 values: moment matching with widened SD
    - 1 value: wide Normal centered on the value (uses uncertainty if available)
    - 0 values: delegated to uninformative module
    """
    from distribird.distributions.uninformative import wide_normal_prior

    n = len(values)

    if diagnostics.enabled():
        if n == 0:
            tier = "uninformative"
        elif n == 1:
            tier = "single_value_wide_normal"
        elif n <= 4:
            tier = "moment_match"
        else:
            tier = "aic_selection"
        diagnostics.record(
            "fitting_input",
            {
                "n_values": n,
                "tier": tier,
                "values": list(values),
                "weights": list(weights) if weights is not None else None,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
            },
        )

    if n == 0:
        return wide_normal_prior(parameter_name, lower_bound, upper_bound)

    if n == 1:
        center = values[0]
        # Use extracted uncertainty as sigma if available
        unc = uncertainties[0] if uncertainties and uncertainties[0] is not None else None
        if unc is not None and unc > 0:
            spread = unc
        else:
            spread = abs(center) * 0.5 if abs(center) > 1e-12 else 1.0
        lb, ub = _truncation_bounds(center, spread, lower_bound, upper_bound)
        return FittedPrior(
            parameter_name=parameter_name,
            family=DistributionFamily.TRUNCATED_NORMAL,
            params={"mu": center, "sigma": spread, "a": lb, "b": ub},
            confidence=ConfidenceLevel.LOW,
            is_informative=True,
            reason=f"Single reported value ({center}); wide Normal prior.",
            n_sources=1,
        )

    if n <= 4:
        candidate = moment_match_normal(values, lower_bound, upper_bound, weights=weights)
        return FittedPrior(
            parameter_name=parameter_name,
            family=candidate.family,
            params=candidate.params,
            confidence=ConfidenceLevel.MEDIUM,
            is_informative=True,
            reason=f"Moment matching from {n} values (widened SD).",
            n_sources=n,
        )

    # 5+ values: full fitting, falling back to moment matching if no family fits.
    candidate = fit_distribution(
        values, lower_bound, upper_bound, weights=weights
    ) or moment_match_normal(values, lower_bound, upper_bound, weights=weights)

    return FittedPrior(
        parameter_name=parameter_name,
        family=candidate.family,
        params=candidate.params,
        confidence=ConfidenceLevel.HIGH,
        is_informative=True,
        reason=f"AIC-selected {candidate.family.value} from {n} values (AIC={candidate.aic:.1f}).",
        n_sources=n,
    )

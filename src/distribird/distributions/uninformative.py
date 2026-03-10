"""Uninformative / fallback prior distributions."""

from __future__ import annotations

import numpy as np

from distribird.models import ConfidenceLevel, DistributionFamily, FittedPrior


def jeffreys_prior(
    parameter_name: str,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
) -> FittedPrior:
    """Jeffreys (uniform) prior over the given bounds."""
    lb = lower_bound if lower_bound is not None else 0.0
    ub = upper_bound if upper_bound is not None else 1.0
    return FittedPrior(
        parameter_name=parameter_name,
        family=DistributionFamily.UNIFORM,
        params={"lower": lb, "upper": ub},
        confidence=ConfidenceLevel.NONE,
        is_informative=False,
        reason="No literature evidence found; Jeffreys (uniform) prior.",
        n_sources=0,
    )


def wide_normal_prior(
    parameter_name: str,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
) -> FittedPrior:
    """Wide Normal prior spanning the feasible range."""
    lb = lower_bound if lower_bound is not None else -1e6
    ub = upper_bound if upper_bound is not None else 1e6

    if np.isfinite(lb) and np.isfinite(ub):
        mu = (lb + ub) / 2.0
        sigma = (ub - lb) / 4.0
    elif np.isfinite(lb):
        mu = lb + 100.0
        sigma = 100.0
    elif np.isfinite(ub):
        mu = ub - 100.0
        sigma = 100.0
    else:
        mu = 0.0
        sigma = 1000.0

    return FittedPrior(
        parameter_name=parameter_name,
        family=DistributionFamily.TRUNCATED_NORMAL,
        params={"mu": mu, "sigma": sigma, "a": lb, "b": ub},
        confidence=ConfidenceLevel.NONE,
        is_informative=False,
        reason="No literature evidence found; wide Normal prior.",
        n_sources=0,
    )

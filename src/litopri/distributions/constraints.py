"""Constraint encoding for parameters."""

from __future__ import annotations

from litopri.models import ConstraintSpec


def check_value_in_bounds(value: float, constraint: ConstraintSpec) -> bool:
    """Check if a value respects the given constraint."""
    if constraint.lower_bound is not None and value < constraint.lower_bound:
        return False
    if constraint.upper_bound is not None and value > constraint.upper_bound:
        return False
    return True


def filter_values_by_constraints(
    values: list[float], constraint: ConstraintSpec
) -> tuple[list[float], list[float]]:
    """Split values into those within and outside bounds.

    Returns (valid, excluded) tuple.
    """
    valid = []
    excluded = []
    for v in values:
        if check_value_in_bounds(v, constraint):
            valid.append(v)
        else:
            excluded.append(v)
    return valid, excluded


def constraint_comment(constraint: ConstraintSpec) -> str:
    """Generate a human-readable constraint comment for export."""
    parts = []
    if constraint.lower_bound is not None:
        parts.append(f"lower_bound={constraint.lower_bound}")
    if constraint.upper_bound is not None:
        parts.append(f"upper_bound={constraint.upper_bound}")
    if constraint.description:
        parts.append(constraint.description)
    return "; ".join(parts) if parts else "no constraints"

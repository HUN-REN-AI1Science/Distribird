"""LaTeX and Markdown table export for model checking results."""

from __future__ import annotations

from distribird.models import PipelineResult


def _fmt(val: float, precision: int = 4) -> str:
    return f"{val:.{precision}g}"


_LATEX_ESCAPES = {
    "\\": r"\textbackslash{}",
    "{": r"\{",
    "}": r"\}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def _latex_escape(s: str) -> str:
    """Escape characters with special meaning in LaTeX."""
    out: list[str] = []
    for ch in s:
        out.append(_LATEX_ESCAPES.get(ch, ch))
    return "".join(out)


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def batch_to_markdown_table(results: list[PipelineResult]) -> str:
    """Generate a Markdown table of model checking diagnostics."""
    header = "| Parameter | Distribution | MAP | Mean | 95% CI | KS stat | KS p | AIC | n |"
    sep = "|---|---|---|---|---|---|---|---|---|"
    rows = [header, sep]

    for r in results:
        mc = r.model_check
        if mc is None:
            rows.append(
                f"| {r.parameter.name} | {r.prior.family.value} | — | — | — | — | — | — | 0 |"
            )
            continue
        ci = f"[{_fmt(mc.ci_95_lower)}, {_fmt(mc.ci_95_upper)}]"
        rows.append(
            f"| {r.parameter.name} "
            f"| {r.prior.family.value} "
            f"| {_fmt(mc.map_estimate)} "
            f"| {_fmt(mc.dist_mean)} "
            f"| {ci} "
            f"| {_fmt(mc.ks_statistic, 3)} "
            f"| {_fmt(mc.ks_pvalue, 3)} "
            f"| {_fmt(mc.aic, 1)} "
            f"| {mc.n_values} |"
        )

    return "\n".join(rows)


# ---------------------------------------------------------------------------
# LaTeX
# ---------------------------------------------------------------------------


def batch_to_latex_table(results: list[PipelineResult]) -> str:
    """Generate a LaTeX table of model checking diagnostics for the paper."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Model checking diagnostics for fitted prior distributions.}",
        r"\label{tab:model-check}",
        r"\begin{tabular}{l l r r c r r r r}",
        r"\toprule",
        (
            r"Parameter & Distribution & MAP & Mean & 95\% CI "
            r"& KS stat & KS $p$ & AIC & $n$ \\"
        ),
        r"\midrule",
    ]

    for r in results:
        mc = r.model_check
        name = _latex_escape(r.parameter.name)
        family = _latex_escape(r.prior.family.value)
        if mc is None:
            lines.append(f"{name} & {family} & --- & --- & --- & --- & --- & --- & 0 \\\\")
            continue
        ci = f"[{_fmt(mc.ci_95_lower)}, {_fmt(mc.ci_95_upper)}]"
        lines.append(
            f"{name} & {family} & {_fmt(mc.map_estimate)} & {_fmt(mc.dist_mean)} "
            f"& {ci} & {_fmt(mc.ks_statistic, 3)} & {_fmt(mc.ks_pvalue, 3)} "
            f"& {_fmt(mc.aic, 1)} & {mc.n_values} \\\\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )

    return "\n".join(lines)

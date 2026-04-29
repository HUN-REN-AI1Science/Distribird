"""Real-LLM BullshitBench runner.

Runs a curated mix of fake, theoretical, and real parameters through the full
Distribird pipeline (with real LLM + Semantic Scholar / OpenAlex). Saves a
markdown report comparing the validity verdicts.

Usage:
    python examples/bullshitbench_run.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from distribird.agent.pipeline import run_parameter
from distribird.config import get_settings
from distribird.models import (
    ConstraintSpec,
    ParameterInput,
    ParameterValidity,
    PipelineResult,
)

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger(__name__)


@dataclass
class TestCase:
    name: str
    description: str
    domain_context: str
    expected: ParameterValidity
    category: str  # "nonsense", "theoretical", "real"
    constraints: ConstraintSpec | None = None


CASES: list[TestCase] = [
    # ── Pure nonsense ──
    TestCase(
        name="mumblesnort_factor",
        description="A fabricated coefficient that does not exist in any literature",
        domain_context="general scientific testing",
        expected=ParameterValidity.LIKELY_INVALID,
        category="nonsense",
        constraints=ConstraintSpec(lower_bound=0, upper_bound=10),
    ),
    TestCase(
        name="fake_quantum_correction_xyz",
        description="A made-up quantum correction term with no scientific basis",
        domain_context="theoretical physics",
        expected=ParameterValidity.LIKELY_INVALID,
        category="nonsense",
        constraints=ConstraintSpec(lower_bound=0, upper_bound=1),
    ),
    # ── Theoretical / non-empirical ──
    TestCase(
        name="biome_bgcmuso_carbon_pool_calibration_weight_v3",
        description=(
            "Internal calibration weight from Biome-BGCMuSo model version 3.x; "
            "purely a model-internal tuning parameter"
        ),
        domain_context="Biome-BGCMuSo crop modeling",
        expected=ParameterValidity.SUSPICIOUS,
        category="theoretical",
        constraints=ConstraintSpec(lower_bound=0, upper_bound=10),
    ),
    # ── Real parameter (control) ──
    TestCase(
        name="specific_leaf_area",
        description="Leaf area per unit dry mass of leaves",
        domain_context="maize crop modeling",
        expected=ParameterValidity.VALID,
        category="real",
        constraints=ConstraintSpec(lower_bound=5, upper_bound=50),
    ),
]


async def run_one(case: TestCase, settings) -> tuple[TestCase, PipelineResult, float]:
    """Run a single test case through the real pipeline."""
    param = ParameterInput(
        name=case.name,
        description=case.description,
        unit="",
        domain_context=case.domain_context,
        constraints=case.constraints or ConstraintSpec(),
    )
    print(f"\n{'=' * 70}", flush=True)
    print(f"Running: {case.name}  (expect: {case.expected.value})", flush=True)
    print(f"{'=' * 70}", flush=True)
    t0 = time.monotonic()
    try:
        result = await run_parameter(param, settings)
    except Exception as e:
        logger.exception("Pipeline crashed for %s", case.name)
        # Return a synthetic failure result
        from distribird.distributions.uninformative import wide_normal_prior

        result = PipelineResult(
            parameter=param,
            prior=wide_normal_prior(
                param.name,
                param.constraints.lower_bound,
                param.constraints.upper_bound,
            ),
            warnings=[f"Crash: {e}"],
        )
    elapsed = time.monotonic() - t0
    print(
        f"  → verdict: {result.parameter_validity.value}  "
        f"(papers={result.papers_found}, values={result.values_extracted}, "
        f"elapsed={elapsed:.1f}s)",
        flush=True,
    )
    return case, result, elapsed


def render_markdown(
    runs: list[tuple[TestCase, PipelineResult, float]],
    output_path: Path,
) -> None:
    lines: list[str] = []
    lines.append("# BullshitBench — Real-LLM Run\n")
    lines.append(
        "Real end-to-end pipeline runs (LLM enrichment + Semantic Scholar + extraction + validity check)\n"
    )

    # ── Summary table ──
    correct = sum(1 for c, r, _ in runs if r.parameter_validity == c.expected)
    lines.append(f"## Summary — {correct}/{len(runs)} verdicts match expected\n")
    lines.append(
        "| Parameter | Category | Expected | Verdict | Match | Papers | Values | Time |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for case, result, elapsed in runs:
        match = "OK" if result.parameter_validity == case.expected else "MISMATCH"
        lines.append(
            f"| `{case.name}` | {case.category} | {case.expected.value} | "
            f"{result.parameter_validity.value} | {match} | "
            f"{result.papers_found} | {result.values_extracted} | {elapsed:.1f}s |"
        )
    lines.append("")

    # ── Per-case detail ──
    lines.append("## Per-case details\n")
    for case, result, elapsed in runs:
        match = (
            "MATCHES expected"
            if result.parameter_validity == case.expected
            else "DOES NOT match expected"
        )
        lines.append(f"### `{case.name}` — {case.category}")
        lines.append(f"- **Description:** {case.description}")
        lines.append(f"- **Domain:** {case.domain_context}")
        lines.append(f"- **Expected:** `{case.expected.value}`")
        lines.append(
            f"- **Verdict:** `{result.parameter_validity.value}` — {match}"
        )
        lines.append(f"- **Reason:** {result.validity_reason or '(none)'}")
        lines.append(f"- **Empirical:** {result.is_empirical}")
        lines.append(
            f"- **Pipeline:** {result.papers_found} papers, "
            f"{result.values_extracted} values, "
            f"prior `{result.prior.family.value}` "
            f"(confidence: {result.prior.confidence.value}, "
            f"informative: {result.prior.is_informative})"
        )
        if result.enrichment is not None:
            lines.append(
                f"- **LLM recognized:** {result.enrichment.is_recognized_parameter} "
                f"(confidence: {result.enrichment.recognition_confidence})"
            )
            lines.append(
                f"- **LLM empirically measured:** {result.enrichment.empirically_measured}"
            )
            if result.enrichment.common_terminology:
                lines.append(
                    f"- **Terminology:** {', '.join(result.enrichment.common_terminology[:5])}"
                )
        if result.validity_signals:
            sig = result.validity_signals
            sig_short = {
                k: v
                for k, v in sig.items()
                if k != "probe_result"
            }
            lines.append("- **Signals:**")
            lines.append("  ```json")
            lines.append("  " + json.dumps(sig_short, indent=2).replace("\n", "\n  "))
            lines.append("  ```")
            if "probe_result" in sig and sig["probe_result"]:
                lines.append(
                    f"- **LLM probe:** verdict=`{sig['probe_result'].get('verdict')}`, "
                    f"reason={sig['probe_result'].get('reason')!r}"
                )
        if result.warnings:
            lines.append("- **Warnings:**")
            for w in result.warnings:
                lines.append(f"  - {w}")
        lines.append(f"- **Elapsed:** {elapsed:.1f}s\n")

    output_path.write_text("\n".join(lines))
    print(f"\nMarkdown report saved: {output_path}")


async def main() -> None:
    settings = get_settings()
    if not settings.llm_base_url or not settings.llm_api_key:
        print("ERROR: DISTRIBIRD_LLM_BASE_URL and DISTRIBIRD_LLM_API_KEY required")
        return

    print(f"LLM: {settings.llm_model} via {settings.llm_base_url}")
    print(f"Validity check enabled: {settings.enable_validity_check}")
    print(f"Validity probe enabled: {settings.enable_validity_probe}")
    print(f"Running {len(CASES)} test cases...")

    # Run sequentially to avoid hammering the LLM/search APIs
    runs = []
    for case in CASES:
        result_tuple = await run_one(case, settings)
        runs.append(result_tuple)

    output_path = Path(__file__).parent.parent / "bullshitbench_real_llm_results.md"
    render_markdown(runs, output_path)

    # Brief stdout summary
    correct = sum(1 for c, r, _ in runs if r.parameter_validity == c.expected)
    print(f"\n{'=' * 70}")
    print(f"FINAL: {correct}/{len(runs)} verdicts match expected")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    asyncio.run(main())

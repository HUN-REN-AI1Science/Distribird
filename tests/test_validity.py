"""Unit tests for the validity classification heuristics."""

from __future__ import annotations

from distribird.agent.validity import (
    apply_probe_verdict,
    classify_validity_passive,
)
from distribird.models import (
    ConfidenceLevel,
    DistributionFamily,
    EnrichedContext,
    FittedPrior,
    ParameterValidity,
)


def _make_prior(
    informative: bool = True, confidence: ConfidenceLevel = ConfidenceLevel.HIGH
) -> FittedPrior:
    return FittedPrior(
        parameter_name="test",
        family=DistributionFamily.NORMAL,
        params={"mu": 0.5, "sigma": 0.1},
        confidence=confidence,
        is_informative=informative,
        n_sources=5,
    )


def test_rule_no_literature_after_refinement_is_likely_invalid():
    enrichment = EnrichedContext(
        is_recognized_parameter=False,
        recognition_confidence="none",
    )
    verdict, reason, signals, is_empirical = classify_validity_passive(
        enrichment=enrichment,
        prior=_make_prior(informative=False, confidence=ConfidenceLevel.NONE),
        papers_found=0,
        values_extracted=0,
        queries_tried=3,
    )
    assert verdict == ParameterValidity.LIKELY_INVALID
    assert "3 refined queries" in reason
    assert signals["papers_found"] == 0
    assert signals["n_queries_tried"] == 3


def test_rule_unrecognized_no_literature_is_likely_invalid():
    enrichment = EnrichedContext(
        is_recognized_parameter=False,
        recognition_confidence="none",
        common_terminology=["something"],  # has terminology but unrecognized
    )
    verdict, reason, _, _ = classify_validity_passive(
        enrichment=enrichment,
        prior=_make_prior(informative=False, confidence=ConfidenceLevel.NONE),
        papers_found=0,
        values_extracted=0,
        queries_tried=1,  # only one query, so rule 1 doesn't fire
    )
    assert verdict == ParameterValidity.LIKELY_INVALID
    assert "did not recognize" in reason.lower()


def test_rule_no_terminology_no_papers_is_likely_invalid():
    enrichment = EnrichedContext(
        is_recognized_parameter=None,  # LLM didn't say
        common_terminology=[],
    )
    verdict, _, _, _ = classify_validity_passive(
        enrichment=enrichment,
        prior=_make_prior(informative=False, confidence=ConfidenceLevel.NONE),
        papers_found=0,
        values_extracted=0,
        queries_tried=1,
    )
    assert verdict == ParameterValidity.LIKELY_INVALID


def test_rule_theoretical_only_is_suspicious():
    enrichment = EnrichedContext(
        is_recognized_parameter=True,
        recognition_confidence="medium",
        empirically_measured=False,
        common_terminology=["model term", "theoretical quantity"],
    )
    verdict, reason, _, is_empirical = classify_validity_passive(
        enrichment=enrichment,
        prior=_make_prior(informative=False, confidence=ConfidenceLevel.NONE),
        papers_found=5,
        values_extracted=0,
        queries_tried=2,
    )
    assert verdict == ParameterValidity.SUSPICIOUS
    assert "theoretical" in reason.lower() or "not empirically" in reason.lower()
    assert is_empirical is False


def test_rule_strong_positive_signal_is_valid():
    enrichment = EnrichedContext(
        is_recognized_parameter=True,
        recognition_confidence="high",
        empirically_measured=True,
        common_terminology=["specific leaf area", "SLA", "leaf area mass ratio"],
    )
    verdict, reason, _, is_empirical = classify_validity_passive(
        enrichment=enrichment,
        prior=_make_prior(informative=True, confidence=ConfidenceLevel.HIGH),
        papers_found=8,
        values_extracted=15,
        queries_tried=1,
    )
    assert verdict == ParameterValidity.VALID
    assert "literature-backed" in reason.lower() or "recognized" in reason.lower()
    assert is_empirical is True


def test_rule_ambiguous_defaults_to_suspicious():
    """Recognized parameter with literature but only LOW confidence → suspicious."""
    enrichment = EnrichedContext(
        is_recognized_parameter=True,
        recognition_confidence="low",
        common_terminology=["maybe relevant"],
    )
    verdict, _, _, _ = classify_validity_passive(
        enrichment=enrichment,
        prior=_make_prior(informative=True, confidence=ConfidenceLevel.LOW),
        papers_found=2,
        values_extracted=1,
        queries_tried=2,
    )
    assert verdict == ParameterValidity.SUSPICIOUS


def test_apply_probe_verdict_upgrades_suspicious():
    probe = {"verdict": "valid", "is_empirical": True, "reason": "actually a real param"}
    verdict, reason, empirical = apply_probe_verdict(
        passive_verdict=ParameterValidity.SUSPICIOUS,
        passive_reason="ambiguous",
        passive_is_empirical=None,
        probe_result=probe,
    )
    assert verdict == ParameterValidity.VALID
    assert reason == "actually a real param"
    assert empirical is True


def test_apply_probe_verdict_does_not_override_likely_invalid():
    probe = {"verdict": "valid", "is_empirical": True, "reason": "false probe"}
    verdict, reason, _ = apply_probe_verdict(
        passive_verdict=ParameterValidity.LIKELY_INVALID,
        passive_reason="no literature found",
        passive_is_empirical=None,
        probe_result=probe,
    )
    assert verdict == ParameterValidity.LIKELY_INVALID
    assert reason == "no literature found"


def test_apply_probe_verdict_handles_none_probe():
    verdict, reason, empirical = apply_probe_verdict(
        passive_verdict=ParameterValidity.SUSPICIOUS,
        passive_reason="ambiguous",
        passive_is_empirical=False,
        probe_result=None,
    )
    assert verdict == ParameterValidity.SUSPICIOUS
    assert reason == "ambiguous"
    assert empirical is False


def test_apply_probe_verdict_handles_invalid_verdict_string():
    probe = {"verdict": "garbage_value", "is_empirical": True, "reason": "x"}
    verdict, reason, _ = apply_probe_verdict(
        passive_verdict=ParameterValidity.SUSPICIOUS,
        passive_reason="ambiguous",
        passive_is_empirical=None,
        probe_result=probe,
    )
    assert verdict == ParameterValidity.SUSPICIOUS
    assert reason == "ambiguous"

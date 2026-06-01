"""Parameter validity detection — passive heuristics + optional LLM probe.

This module classifies a parameter request as VALID, SUSPICIOUS, or LIKELY_INVALID
based on signals collected during the pipeline run:
- Enrichment LLM self-flags (is_recognized_parameter, empirically_measured)
- Pipeline observations (papers_found, values_extracted, queries_tried)
- Final fitted prior confidence

The classification is two-stage:
1. Passive (no LLM cost): heuristic rules combine the above signals.
2. Optional active (one LLM call): if passive verdict is SUSPICIOUS and budget
   permits, a dedicated probe asks the LLM for a second opinion.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, ValidationError

from distribird.config import Settings
from distribird.models import (
    ConfidenceLevel,
    EnrichedContext,
    FittedPrior,
    ParameterInput,
    ParameterValidity,
)

logger = logging.getLogger(__name__)

# Minimum extracted values required to consider a literature-backed prior reliable.
# Two values give the synthesizer enough signal to fit a Normal via moment matching
# (single-value priors fall back to a wide Normal with LOW confidence).
MIN_VALUES_FOR_VALID = 2

REASON_NO_LITERATURE = "No literature found across {n} refined queries"
REASON_LLM_UNRECOGNIZED = "LLM did not recognize the parameter and no literature was found"
REASON_NO_TERMINOLOGY = "No domain terminology and no literature found"
REASON_THEORETICAL_ONLY = "Theoretical/derived parameter, not empirically measured"
REASON_EMPIRICAL_UNCLEAR = (
    "Empirical status unclear; literature found but no measured values extractable"
)
REASON_LITERATURE_BACKED = "Recognized parameter with literature-backed prior"
REASON_INSUFFICIENT_EVIDENCE = "Insufficient evidence for confident classification"


class ProbeResult(BaseModel):
    """Parsed output of the LLM validity probe."""

    verdict: ParameterValidity
    reason: str = ""
    is_empirical: bool | None = None


def classify_validity_passive(
    enrichment: EnrichedContext | None,
    prior: FittedPrior | None,
    papers_found: int,
    values_extracted: int,
    queries_tried: int,
) -> tuple[ParameterValidity, str, dict[str, Any], bool | None]:
    """Classify validity using only heuristic rules (no LLM call).

    Returns:
        (verdict, reason, signals, is_empirical)
    """
    is_empirical = enrichment.empirically_measured if enrichment else None
    is_recognized = enrichment.is_recognized_parameter if enrichment else None
    recog_conf = enrichment.recognition_confidence if enrichment else "none"
    n_terms = len(enrichment.common_terminology) if enrichment else 0
    prior_confidence = prior.confidence if prior else ConfidenceLevel.NONE
    prior_informative = prior.is_informative if prior else False

    signals: dict[str, Any] = {
        "papers_found": papers_found,
        "values_extracted": values_extracted,
        "n_queries_tried": queries_tried,
        "is_recognized_parameter": is_recognized,
        "recognition_confidence": recog_conf,
        "empirically_measured": is_empirical,
        "n_terminology": n_terms,
        "prior_is_informative": prior_informative,
        "prior_confidence": prior_confidence.value,
    }

    if papers_found == 0 and values_extracted == 0 and queries_tried >= 2:
        return (
            ParameterValidity.LIKELY_INVALID,
            REASON_NO_LITERATURE.format(n=queries_tried),
            signals,
            is_empirical,
        )

    if is_recognized is False and recog_conf in {"none", "low"} and papers_found == 0:
        return ParameterValidity.LIKELY_INVALID, REASON_LLM_UNRECOGNIZED, signals, is_empirical

    if n_terms == 0 and papers_found == 0:
        return ParameterValidity.LIKELY_INVALID, REASON_NO_TERMINOLOGY, signals, is_empirical

    if is_empirical is not True and values_extracted == 0 and papers_found > 0:
        reason = REASON_THEORETICAL_ONLY if is_empirical is False else REASON_EMPIRICAL_UNCLEAR
        return ParameterValidity.SUSPICIOUS, reason, signals, is_empirical

    if (
        prior_informative
        and prior_confidence in {ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM}
        and is_recognized is True
        and is_empirical is not False
        and values_extracted >= MIN_VALUES_FOR_VALID
    ):
        return ParameterValidity.VALID, REASON_LITERATURE_BACKED, signals, is_empirical

    return ParameterValidity.SUSPICIOUS, REASON_INSUFFICIENT_EVIDENCE, signals, is_empirical


def validity_probe_llm(
    parameter: ParameterInput,
    enrichment: EnrichedContext | None,
    signals: dict[str, Any],
    settings: Settings,
) -> dict[str, Any] | None:
    """Run a dedicated LLM probe for ambiguous validity cases.

    Returns the parsed JSON dict {verdict, is_empirical, reason}, or None on failure.
    """
    from openai import OpenAI

    from distribird.agent.extract import _llm_json_call
    from distribird.agent.prompts import PARAMETER_VALIDITY_PROBE

    is_recognized = enrichment.is_recognized_parameter if enrichment else None
    recog_conf = enrichment.recognition_confidence if enrichment else "none"
    empirical = enrichment.empirically_measured if enrichment else None
    if enrichment and enrichment.common_terminology:
        terminology = ", ".join(enrichment.common_terminology)
    else:
        terminology = "(none)"

    prompt = PARAMETER_VALIDITY_PROBE.format(
        name=parameter.name,
        description=parameter.description,
        domain_context=parameter.domain_context or "(unspecified)",
        is_recognized=is_recognized,
        recognition_confidence=recog_conf,
        empirically_measured=empirical,
        terminology=terminology,
        n_queries=signals.get("n_queries_tried", 0),
        papers_found=signals.get("papers_found", 0),
        values_extracted=signals.get("values_extracted", 0),
    )

    logger.info(
        "[LLM:validity_probe] param=%r model=%s",
        parameter.name,
        settings.llm_model,
    )

    try:
        client = OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)
        raw = _llm_json_call(
            client,
            settings.llm_model,
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            label="validity_probe",
        )
    except Exception as e:
        logger.warning("[LLM:validity_probe] failed: %s", e)
        return None

    if not isinstance(raw, dict):
        logger.warning("[LLM:validity_probe] unexpected response type: %s", type(raw).__name__)
        return None

    return raw


def apply_probe_verdict(
    passive_verdict: ParameterValidity,
    passive_reason: str,
    passive_is_empirical: bool | None,
    probe_result: dict[str, Any] | None,
) -> tuple[ParameterValidity, str, bool | None]:
    """Combine passive verdict with LLM probe result.

    The probe can only refine SUSPICIOUS verdicts; clear VALID/LIKELY_INVALID
    verdicts from the passive heuristics are not overridden.
    """
    if probe_result is None or passive_verdict != ParameterValidity.SUSPICIOUS:
        return passive_verdict, passive_reason, passive_is_empirical

    try:
        probe = ProbeResult.model_validate(probe_result)
    except ValidationError:
        return passive_verdict, passive_reason, passive_is_empirical

    final_empirical = (
        probe.is_empirical if probe.is_empirical is not None else passive_is_empirical
    )
    return probe.verdict, probe.reason or passive_reason, final_empirical

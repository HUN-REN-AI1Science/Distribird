"""BullshitBench: integration tests for parameter validity detection.

These tests verify that Distribird flags fake/nonsense/theoretical-only
parameters appropriately, and does NOT misclassify real parameters.
"""

from contextlib import ExitStack
from unittest.mock import AsyncMock, patch

import pytest

from distribird.agent.pipeline import run_parameter
from distribird.config import Settings
from distribird.models import (
    ConstraintSpec,
    EnrichedContext,
    ExtractedValue,
    LiteratureEvidence,
    ParameterInput,
    ParameterValidity,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def bullshit_settings():
    """Settings with enrichment enabled (we mock it) and validity check on."""
    return Settings(
        llm_base_url="http://localhost:4000",
        llm_api_key="test",
        enable_deliberation=False,
        enable_context_enrichment=True,
        enable_llm_deep_research=False,
        llm_web_search=False,
        enable_validity_check=True,
        enable_validity_probe=True,
        search_refinement_max=1,  # allow one refinement → queries_tried can reach 2
        cross_enrichment_max=0,
        extraction_refinement_max=0,
    )


def _mk_param(name: str, description: str = "") -> ParameterInput:
    return ParameterInput(
        name=name,
        description=description or f"Some description for {name}",
        unit="",
        domain_context="general scientific testing",
        constraints=ConstraintSpec(lower_bound=0, upper_bound=100),
    )


def _mk_paper(idx: int, title_prefix: str = "Paper") -> LiteratureEvidence:
    return LiteratureEvidence(
        title=f"{title_prefix} {idx}",
        doi=f"10.1234/test{idx}",
        abstract=f"Value reported was {1.0 + idx * 0.5}.",
        year=2020 + idx,
        extracted_values=[ExtractedValue(reported_value=1.0 + idx * 0.5)],
    )


def _mk_papers_no_values(count: int) -> list[LiteratureEvidence]:
    """Papers without extracted values — for testing model-internal/calibration parameters."""
    papers = [_mk_paper(i) for i in range(count)]
    for p in papers:
        p.extracted_values = []
    return papers


# ---------------------------------------------------------------------------
# Helpers for mocking the pipeline
# ---------------------------------------------------------------------------


def _patch_pipeline(
    stack: ExitStack,
    enrichment: EnrichedContext | None,
    search_papers: list[LiteratureEvidence],
    extract_papers: list[LiteratureEvidence] | None = None,
    probe_return=None,
):
    """Enter pipeline mock contexts. Returns the probe mock for assertion."""
    if extract_papers is None:
        extract_papers = search_papers

    stack.enter_context(
        patch(
            "distribird.agent.enrich.enrich_parameter_context",
            return_value=enrichment,
        )
    )
    stack.enter_context(
        patch(
            "distribird.agent.search.generate_search_queries",
            return_value=["query1"],
        )
    )
    stack.enter_context(
        patch(
            "distribird.agent.search.search_all_queries",
            new_callable=AsyncMock,
            return_value=search_papers,
        )
    )
    stack.enter_context(
        patch(
            "distribird.agent.extract.extract_all_values",
            return_value=extract_papers,
        )
    )
    probe_mock = stack.enter_context(
        patch(
            "distribird.agent.validity.validity_probe_llm",
            return_value=probe_return,
        )
    )
    return probe_mock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pure_nonsense_mumblesnort(bullshit_settings):
    """A pure-nonsense parameter name with no literature → LIKELY_INVALID."""
    enrichment = EnrichedContext(
        is_recognized_parameter=False,
        recognition_confidence="none",
        empirically_measured=None,
        common_terminology=[],
    )
    param = _mk_param("mumblesnort_factor", "A totally fabricated parameter name")

    with ExitStack() as stack:
        probe_mock = _patch_pipeline(
            stack,
            enrichment,
            [],
            probe_return={
                "verdict": "likely_invalid",
                "is_empirical": False,
                "reason": "x",
            },
        )
        result = await run_parameter(param, bullshit_settings)

    assert result.parameter_validity == ParameterValidity.LIKELY_INVALID
    assert not result.prior.is_informative
    assert result.papers_found == 0
    # No probe call needed for clear-cut cases
    assert probe_mock.call_count == 0
    assert any("LIKELY INVALID" in w for w in result.warnings)


@pytest.mark.asyncio
async def test_pure_nonsense_fake_xyz(bullshit_settings):
    """Another pure-nonsense — verify validity_signals populated."""
    enrichment = EnrichedContext(
        is_recognized_parameter=False,
        recognition_confidence="none",
        common_terminology=[],
    )
    param = _mk_param("fake_quantum_correction_xyz")

    with ExitStack() as stack:
        _patch_pipeline(stack, enrichment, [])
        result = await run_parameter(param, bullshit_settings)

    assert result.parameter_validity == ParameterValidity.LIKELY_INVALID
    for key in (
        "papers_found",
        "values_extracted",
        "is_recognized_parameter",
        "n_queries_tried",
    ):
        assert key in result.validity_signals


@pytest.mark.asyncio
async def test_plausible_sounding_fake_uses_probe(bullshit_settings):
    """Plausible-sounding fake fools the enrichment LLM (passes early gate).

    The enrichment LLM is fooled into setting is_recognized_parameter=True with
    low confidence, so the early-skip route does NOT fire and the pipeline
    proceeds through full search/extraction. The terminal validity_check then
    invokes the probe to escalate the SUSPICIOUS verdict to LIKELY_INVALID.
    """
    enrichment = EnrichedContext(
        is_recognized_parameter=True,  # LLM fooled by plausible name
        recognition_confidence="low",
        common_terminology=["chlorophyll fluorescence"],
        empirically_measured=None,
    )
    param = _mk_param(
        "chlorophyll_resonance_index",
        "A plausible-sounding but fabricated parameter",
    )
    # Two papers, but extraction yields zero values → ambiguous
    papers = _mk_papers_no_values(2)

    probe_response = {
        "verdict": "likely_invalid",
        "is_empirical": False,
        "reason": "fabricated terminology, no real measurements exist",
    }
    with ExitStack() as stack:
        probe_mock = _patch_pipeline(
            stack,
            enrichment,
            papers,
            extract_papers=[],
            probe_return=probe_response,
        )
        result = await run_parameter(param, bullshit_settings)

    assert probe_mock.call_count == 1
    assert result.parameter_validity == ParameterValidity.LIKELY_INVALID
    assert "fabricated" in result.validity_reason.lower()


@pytest.mark.asyncio
async def test_empirical_only_theoretical_param(bullshit_settings):
    """Theoretical-only parameter (papers exist but no measurements) → SUSPICIOUS."""
    enrichment = EnrichedContext(
        is_recognized_parameter=True,
        recognition_confidence="medium",
        empirically_measured=False,
        common_terminology=["latent state", "model-internal"],
    )
    param = _mk_param(
        "hypothetical_dark_carbon_pool",
        "Purely theoretical model term, never measured",
    )
    papers = _mk_papers_no_values(5)

    with ExitStack() as stack:
        _patch_pipeline(stack, enrichment, papers, extract_papers=[])
        result = await run_parameter(param, bullshit_settings)

    assert result.parameter_validity == ParameterValidity.SUSPICIOUS
    assert result.is_empirical is False
    assert (
        "theoretical" in result.validity_reason.lower()
        or "not empirically" in result.validity_reason.lower()
    )


@pytest.mark.asyncio
async def test_misspelled_real_param_is_valid(bullshit_settings):
    """A misspelling of a real param still recognized via downstream evidence → VALID."""
    enrichment = EnrichedContext(
        is_recognized_parameter=True,
        recognition_confidence="low",
        empirically_measured=True,
        parameter_meaning="Appears to be a misspelling of maximum leaf area index",
        common_terminology=["leaf area index", "LAI"],
    )
    param = _mk_param("maxmimum_leaf_area_indxex")
    papers = [_mk_paper(i) for i in range(6)]

    with ExitStack() as stack:
        _patch_pipeline(stack, enrichment, papers)
        result = await run_parameter(param, bullshit_settings)

    assert result.parameter_validity == ParameterValidity.VALID
    assert result.prior.is_informative


@pytest.mark.asyncio
async def test_real_parameter_control_specific_leaf_area(bullshit_settings):
    """Real, well-known parameter with literature → VALID, probe NOT called."""
    enrichment = EnrichedContext(
        is_recognized_parameter=True,
        recognition_confidence="high",
        empirically_measured=True,
        common_terminology=["specific leaf area", "SLA", "leaf area mass ratio"],
        typical_range="10-30 m2/kg",
    )
    param = _mk_param("specific_leaf_area", "Leaf area per unit dry mass")
    papers = [_mk_paper(i) for i in range(6)]

    with ExitStack() as stack:
        probe_mock = _patch_pipeline(
            stack,
            enrichment,
            papers,
            probe_return={"verdict": "valid", "is_empirical": True, "reason": "x"},
        )
        result = await run_parameter(param, bullshit_settings)

    assert result.parameter_validity == ParameterValidity.VALID
    assert result.prior.is_informative
    assert result.is_empirical is True
    assert probe_mock.call_count == 0


@pytest.mark.asyncio
async def test_real_param_with_search_outage(bullshit_settings):
    """Recognized real param + transient search failure → SUSPICIOUS, not LIKELY_INVALID."""
    enrichment = EnrichedContext(
        is_recognized_parameter=True,
        recognition_confidence="high",
        empirically_measured=True,
        common_terminology=["specific leaf area", "SLA"],
    )
    param = _mk_param("specific_leaf_area")

    # Disable refinement so only 1 query attempted → rule 1 (>=2 queries) doesn't fire
    no_refine_settings = bullshit_settings.model_copy(update={"search_refinement_max": 0})

    probe_response = {
        "verdict": "suspicious",
        "is_empirical": True,
        "reason": "recognized parameter but search returned no results",
    }
    with ExitStack() as stack:
        _patch_pipeline(stack, enrichment, [], probe_return=probe_response)
        result = await run_parameter(param, no_refine_settings)

    assert result.parameter_validity != ParameterValidity.LIKELY_INVALID


@pytest.mark.asyncio
async def test_early_skip_avoids_search_and_extract(bullshit_settings):
    """Clear nonsense at enrich time short-circuits past search/extract/synthesize.

    The enrichment LLM marks is_recognized_parameter=False with none confidence;
    the route_after_enrich short-circuits to validity_check, so the search and
    extract mocks should NOT be called.
    """
    from distribird.agent import extract as extract_mod
    from distribird.agent import search as search_mod

    enrichment = EnrichedContext(
        is_recognized_parameter=False,
        recognition_confidence="none",
        common_terminology=[],
    )
    param = _mk_param("totally_fabricated_xyz")

    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "distribird.agent.enrich.enrich_parameter_context",
                return_value=enrichment,
            )
        )
        gen_mock = stack.enter_context(
            patch.object(search_mod, "generate_search_queries", return_value=["q"])
        )
        search_mock = stack.enter_context(
            patch.object(
                search_mod,
                "search_all_queries",
                new_callable=AsyncMock,
                return_value=[],
            )
        )
        extract_mock = stack.enter_context(
            patch.object(extract_mod, "extract_all_values", return_value=[])
        )
        stack.enter_context(
            patch(
                "distribird.agent.validity.validity_probe_llm",
                return_value=None,
            )
        )
        result = await run_parameter(param, bullshit_settings)

    assert result.parameter_validity == ParameterValidity.LIKELY_INVALID
    # Critical: the expensive nodes were NOT called
    assert gen_mock.call_count == 0, "query_gen should be skipped"
    assert search_mock.call_count == 0, "search should be skipped"
    assert extract_mock.call_count == 0, "extract should be skipped"
    assert result.papers_found == 0
    assert result.values_extracted == 0


@pytest.mark.asyncio
async def test_validity_check_disabled(bullshit_settings):
    """When the toggle is off, validity verdict is UNKNOWN."""
    settings = bullshit_settings.model_copy(update={"enable_validity_check": False})
    enrichment = EnrichedContext(
        is_recognized_parameter=False,
        recognition_confidence="none",
    )
    param = _mk_param("mumblesnort_factor")

    with ExitStack() as stack:
        _patch_pipeline(stack, enrichment, [])
        result = await run_parameter(param, settings)

    assert result.parameter_validity == ParameterValidity.UNKNOWN
    assert result.prior is not None


@pytest.mark.asyncio
async def test_validity_probe_skipped_when_unambiguous(bullshit_settings):
    """For clearly LIKELY_INVALID cases, the LLM probe must NOT be called."""
    enrichment = EnrichedContext(
        is_recognized_parameter=False,
        recognition_confidence="none",
        common_terminology=[],
    )
    param = _mk_param("totally_made_up_parameter_zzz")

    with ExitStack() as stack:
        probe_mock = _patch_pipeline(stack, enrichment, [])
        await run_parameter(param, bullshit_settings)

    assert probe_mock.call_count == 0


@pytest.mark.asyncio
async def test_validity_signals_in_warnings(bullshit_settings):
    """LIKELY_INVALID verdicts must surface in result.warnings."""
    enrichment = EnrichedContext(
        is_recognized_parameter=False,
        recognition_confidence="none",
        common_terminology=[],
    )
    param = _mk_param("garbage_xyz_param")

    with ExitStack() as stack:
        _patch_pipeline(stack, enrichment, [])
        result = await run_parameter(param, bullshit_settings)

    assert any(w.startswith("Parameter validity:") for w in result.warnings)


@pytest.mark.asyncio
async def test_uncertain_empirical_calibration_weight(bullshit_settings):
    """LLM uncertain about empirical status, papers exist, no values → Rule 4b → SUSPICIOUS."""
    from distribird.agent.validity import REASON_EMPIRICAL_UNCLEAR

    enrichment = EnrichedContext(
        is_recognized_parameter=True,
        recognition_confidence="medium",
        empirically_measured=None,
        common_terminology=["model parameter", "tuning weight"],
    )
    param = _mk_param(
        "biome_bgcmuso_root_decomp_q10_v3",
        "Calibration weight for root decomposition Q10 in Biome-BGCMuSo v3",
    )
    papers = _mk_papers_no_values(5)

    # Empty probe reason so we can verify the passive Rule 4b reason survives
    probe_response = {"verdict": "suspicious", "is_empirical": False, "reason": ""}
    with ExitStack() as stack:
        probe_mock = _patch_pipeline(
            stack,
            enrichment,
            papers,
            extract_papers=[],
            probe_return=probe_response,
        )
        result = await run_parameter(param, bullshit_settings)

    assert result.parameter_validity == ParameterValidity.SUSPICIOUS
    assert REASON_EMPIRICAL_UNCLEAR in result.validity_reason
    assert probe_mock.call_count == 1


@pytest.mark.asyncio
async def test_misclassified_empirical_passes_through_probe(bullshit_settings):
    """Probe corrects an LLM error: parameter marked empirical but actually non-empirical.

    Why: Rule 5 requires values_extracted >= MIN_VALUES_FOR_VALID, so 0 extracted
    values blocks VALID even when the LLM mistakenly returned empirically_measured=True.
    The verdict falls through to SUSPICIOUS where the probe corrects to LIKELY_INVALID.
    """
    enrichment = EnrichedContext(
        is_recognized_parameter=True,
        recognition_confidence="high",
        empirically_measured=True,
        common_terminology=["calibration weight"],
    )
    param = _mk_param("dssat_root_factor_v45")
    no_refine_settings = bullshit_settings.model_copy(update={"search_refinement_max": 0})
    papers = _mk_papers_no_values(3)

    probe_response = {
        "verdict": "likely_invalid",
        "is_empirical": False,
        "reason": "version-specific calibration weight; LLM claim of empirical wrong",
    }
    with ExitStack() as stack:
        probe_mock = _patch_pipeline(
            stack,
            enrichment,
            papers,
            extract_papers=[],
            probe_return=probe_response,
        )
        result = await run_parameter(param, no_refine_settings)

    assert probe_mock.call_count == 1
    assert result.parameter_validity == ParameterValidity.LIKELY_INVALID
    assert result.is_empirical is False  # probe corrected the LLM mistake

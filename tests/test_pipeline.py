"""Integration tests for the pipeline (via LangGraph)."""

from unittest.mock import AsyncMock, patch

import pytest

from litopri.agent.pipeline import run_batch, run_parameter
from litopri.config import Settings
from litopri.models import (
    AgentFinding,
    ConfidenceLevel,
    ConstraintSpec,
    DeliberationResult,
    ExtractedValue,
    LiteratureEvidence,
    ParameterInput,
)


@pytest.fixture
def parameter():
    return ParameterInput(
        name="max_lai",
        description="Maximum leaf area index of maize",
        unit="m2/m2",
        domain_context="maize crop modeling",
        constraints=ConstraintSpec(lower_bound=0, upper_bound=12),
    )


@pytest.fixture
def settings():
    return Settings(
        llm_base_url="http://localhost:4000",
        llm_api_key="test",
        enable_deliberation=False,
        enable_context_enrichment=False,
        enable_llm_deep_research=False,
        llm_web_search=False,
        search_refinement_max=0,
        cross_enrichment_max=0,
        extraction_refinement_max=0,
    )


@pytest.fixture
def mock_papers():
    return [
        LiteratureEvidence(
            title=f"Paper {i}",
            doi=f"10.1234/test{i}",
            abstract=f"LAI was {4 + i * 0.5} m2/m2.",
            year=2020 + i,
            extracted_values=[ExtractedValue(reported_value=4 + i * 0.5)],
        )
        for i in range(6)
    ]


@pytest.mark.asyncio
@patch("litopri.agent.extract.extract_all_values")
@patch("litopri.agent.search.search_all_queries", new_callable=AsyncMock)
@patch("litopri.agent.search.generate_search_queries")
async def test_pipeline_high_confidence(
    mock_queries, mock_search, mock_extract, parameter, settings, mock_papers
):
    mock_queries.return_value = ["maize LAI"]
    mock_search.return_value = mock_papers
    mock_extract.return_value = mock_papers

    result = await run_parameter(parameter, settings)

    assert result.prior.is_informative
    assert result.prior.confidence == ConfidenceLevel.HIGH
    assert result.prior.n_sources == 6
    assert result.papers_found == 6


@pytest.mark.asyncio
@patch("litopri.agent.extract.extract_all_values")
@patch("litopri.agent.search.search_all_queries", new_callable=AsyncMock)
@patch("litopri.agent.search.generate_search_queries")
async def test_pipeline_no_evidence(
    mock_queries, mock_search, mock_extract, parameter, settings
):
    mock_queries.return_value = ["maize LAI"]
    mock_search.return_value = []
    mock_extract.return_value = []

    result = await run_parameter(parameter, settings)

    assert not result.prior.is_informative
    assert result.prior.confidence == ConfidenceLevel.NONE


@pytest.mark.asyncio
@patch("litopri.agent.extract.extract_all_values")
@patch("litopri.agent.search.search_all_queries", new_callable=AsyncMock)
@patch("litopri.agent.search.generate_search_queries")
async def test_run_batch_parallel(mock_queries, mock_search, mock_extract, settings):
    """Verify run_batch processes multiple parameters concurrently."""
    params = [
        ParameterInput(
            name=f"param_{i}",
            description=f"Parameter {i}",
            unit="unit",
            domain_context="test",
            constraints=ConstraintSpec(lower_bound=0, upper_bound=100),
        )
        for i in range(3)
    ]

    mock_queries.return_value = ["query"]
    mock_search.return_value = [
        LiteratureEvidence(
            title="Paper",
            doi="10.1234/test",
            abstract="Value was 50.",
            year=2022,
            extracted_values=[ExtractedValue(reported_value=50.0)],
        )
    ]
    mock_extract.return_value = [
        LiteratureEvidence(
            title="Paper",
            extracted_values=[ExtractedValue(reported_value=50.0)],
        )
    ]

    batch_result = await run_batch(params, settings)

    assert len(batch_result.results) == 3
    assert batch_result.metadata["n_parameters"] == 3
    for result in batch_result.results:
        assert result.prior.is_informative


@pytest.mark.asyncio
@patch("litopri.agent.extract.extract_all_values")
@patch("litopri.agent.search.generate_search_queries")
async def test_pipeline_with_deliberation(
    mock_queries, mock_extract, parameter, mock_papers,
):
    """Verify pipeline uses deliberation path when enabled."""
    delib_settings = Settings(
        llm_base_url="http://localhost:4000",
        llm_api_key="test",
        enable_deliberation=True,
        enable_semantic_scholar=True,
        enable_llm_deep_research=False,
        enable_context_enrichment=False,
        llm_web_search=False,
        search_refinement_max=0,
        cross_enrichment_max=0,
        extraction_refinement_max=0,
    )
    mock_queries.return_value = ["maize LAI"]
    mock_extract.return_value = mock_papers

    mock_deliberation = DeliberationResult(
        consensus_papers=mock_papers,
        moderator_rationale="All papers selected.",
        agent_findings=[
            AgentFinding(
                agent_name="s2",
                source_type="semantic_scholar",
                papers=mock_papers,
            )
        ],
    )

    with patch(
        "litopri.agent.deliberation.run_source_agents",
        new_callable=AsyncMock,
        return_value=mock_deliberation.agent_findings,
    ), patch(
        "litopri.agent.deliberation.deliberate",
        new_callable=AsyncMock,
        return_value=mock_deliberation,
    ):
        result = await run_parameter(parameter, delib_settings)

    assert result.deliberation is not None
    assert result.deliberation.moderator_rationale == "All papers selected."
    assert result.papers_found == 6
    assert result.prior.is_informative

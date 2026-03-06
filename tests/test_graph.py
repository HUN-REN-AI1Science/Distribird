"""Tests for LangGraph pipeline graph execution."""

from unittest.mock import AsyncMock, patch

import pytest

from litopri.agent.graph import NODE_META, build_pipeline_graph, run_parameter_graph
from litopri.config import Settings
from litopri.models import (
    ConstraintSpec,
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


def test_graph_compiles():
    graph = build_pipeline_graph()
    compiled = graph.compile()
    assert compiled is not None


def test_graph_has_relevance_judge_node():
    graph = build_pipeline_graph()
    compiled = graph.compile()
    node_names = set(compiled.get_graph().nodes)
    assert "relevance_judge" in node_names


@pytest.mark.asyncio
@patch("litopri.agent.extract.extract_all_values")
@patch("litopri.agent.search.search_all_queries", new_callable=AsyncMock)
@patch("litopri.agent.search.generate_search_queries")
async def test_graph_happy_path(
    mock_queries, mock_search, mock_extract, parameter, settings,
):
    """Full graph execution with sufficient values -> straight to synthesis."""
    papers = [
        LiteratureEvidence(
            title=f"Paper {i}",
            doi=f"10.1/test{i}",
            abstract=f"LAI was {4 + i * 0.5}",
            extracted_values=[ExtractedValue(reported_value=4 + i * 0.5)],
        )
        for i in range(5)
    ]
    mock_queries.return_value = ["maize LAI"]
    mock_search.return_value = papers
    mock_extract.return_value = papers

    result = await run_parameter_graph(parameter, settings)
    assert result.prior.is_informative
    assert result.papers_found == 5
    assert result.values_extracted == 5


@pytest.mark.asyncio
@patch("litopri.agent.extract.extract_all_values")
@patch("litopri.agent.search.search_all_queries", new_callable=AsyncMock)
@patch("litopri.agent.search.generate_search_queries")
async def test_graph_no_loops_when_sufficient(
    mock_queries, mock_search, mock_extract, parameter, settings,
):
    """When enough values are found, no loops trigger."""
    papers = [
        LiteratureEvidence(
            title="P1",
            doi="10.1/a",
            extracted_values=[ExtractedValue(reported_value=5.0)],
        ),
        LiteratureEvidence(
            title="P2",
            doi="10.1/b",
            extracted_values=[ExtractedValue(reported_value=6.0)],
        ),
    ]
    mock_queries.return_value = ["maize LAI"]
    mock_search.return_value = papers
    mock_extract.return_value = papers

    result = await run_parameter_graph(parameter, settings)
    assert result.prior.is_informative
    assert not any("refinement" in w.lower() for w in result.warnings)


@pytest.mark.asyncio
@patch("litopri.agent.extract._llm_json_call")
@patch("litopri.agent.extract.extract_all_values")
@patch("litopri.agent.search.search_all_queries", new_callable=AsyncMock)
@patch("litopri.agent.search.generate_search_queries")
async def test_graph_loop_a_triggers(
    mock_queries, mock_search, mock_extract, mock_llm_json,
    parameter,
):
    """0 values with papers triggers search refinement loop."""
    loop_settings = Settings(
        llm_base_url="http://localhost:4000",
        llm_api_key="test",
        enable_deliberation=False,
        enable_context_enrichment=False,
        enable_llm_deep_research=False,
        llm_web_search=False,
        search_refinement_max=1,
        cross_enrichment_max=0,
        extraction_refinement_max=0,
    )

    papers_no_vals = [
        LiteratureEvidence(title="P1", doi="10.1/a"),
        LiteratureEvidence(title="P2", doi="10.1/b"),
    ]
    papers_with_vals = [
        LiteratureEvidence(
            title="P3",
            doi="10.1/c",
            extracted_values=[ExtractedValue(reported_value=5.0)],
        ),
    ]

    call_count = {"search": 0, "extract": 0}

    async def search_side_effect(*args, **kwargs):
        call_count["search"] += 1
        if call_count["search"] == 1:
            return papers_no_vals
        return papers_with_vals

    def extract_side_effect(*args, **kwargs):
        call_count["extract"] += 1
        if call_count["extract"] == 1:
            return []
        return papers_with_vals

    mock_queries.return_value = ["maize LAI"]
    mock_search.side_effect = search_side_effect
    mock_extract.side_effect = extract_side_effect
    mock_llm_json.return_value = {
        "diagnosis": "Queries too broad",
        "new_queries": ["maize LAI field measurement"],
        "terminology_updates": [],
    }

    result = await run_parameter_graph(parameter, loop_settings)
    assert any("refinement" in w.lower() for w in result.warnings)
    assert call_count["search"] >= 2


@pytest.mark.asyncio
@patch("litopri.agent.extract.extract_all_values")
@patch("litopri.agent.search.search_all_queries", new_callable=AsyncMock)
@patch("litopri.agent.search.generate_search_queries")
async def test_graph_loop_termination(
    mock_queries, mock_search, mock_extract, parameter,
):
    """Budget exhaustion forces synthesis even with 0 values."""
    loop_settings = Settings(
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

    mock_queries.return_value = ["maize LAI"]
    mock_search.return_value = [
        LiteratureEvidence(title="P1", doi="10.1/a"),
    ]
    mock_extract.return_value = []

    result = await run_parameter_graph(parameter, loop_settings)
    assert result.prior is not None
    assert not result.prior.is_informative


@pytest.mark.asyncio
@patch("litopri.agent.extract.extract_all_values")
@patch("litopri.agent.search.search_all_queries", new_callable=AsyncMock)
@patch("litopri.agent.search.generate_search_queries")
async def test_graph_no_papers(
    mock_queries, mock_search, mock_extract, parameter, settings,
):
    """No papers found -> uninformative prior."""
    mock_queries.return_value = ["maize LAI"]
    mock_search.return_value = []
    mock_extract.return_value = []

    result = await run_parameter_graph(parameter, settings)
    assert not result.prior.is_informative
    assert result.papers_found == 0


@pytest.mark.asyncio
@patch("litopri.agent.extract.extract_all_values")
@patch("litopri.agent.search.search_all_queries", new_callable=AsyncMock)
@patch("litopri.agent.search.generate_search_queries")
async def test_graph_streams_with_callback(
    mock_queries, mock_search, mock_extract, parameter, settings,
):
    """astream mode invokes callback for each node."""
    papers = [
        LiteratureEvidence(
            title=f"Paper {i}",
            doi=f"10.1/test{i}",
            abstract=f"LAI was {4 + i * 0.5}",
            extracted_values=[ExtractedValue(reported_value=4 + i * 0.5)],
        )
        for i in range(5)
    ]
    mock_queries.return_value = ["maize LAI"]
    mock_search.return_value = papers
    mock_extract.return_value = papers

    visited: list[str] = []

    def cb(node_name: str, state: dict):
        visited.append(node_name)

    result = await run_parameter_graph(parameter, settings, on_node_complete=cb)
    assert result.prior is not None
    assert "enrich" in visited
    assert "synthesize" in visited
    # All visited nodes should be known in NODE_META
    for name in visited:
        assert name in NODE_META, f"Unknown node {name!r} not in NODE_META"

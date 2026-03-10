"""Tests for LangGraph pipeline nodes."""

from unittest.mock import AsyncMock, patch

import pytest

from distribird.agent.nodes import (
    enrich_node,
    extract_node,
    fetch_fulltext_node,
    quality_gate_node,
    query_gen_node,
    relevance_judge_node,
    route_after_deliberation,
    route_after_quality_gate,
    search_node,
    synthesize_node,
)
from distribird.agent.state import IterationBudget, PipelineState, QualityMetrics
from distribird.config import Settings
from distribird.models import (
    ConstraintSpec,
    ExtractedValue,
    LiteratureEvidence,
    ParameterInput,
)


def _make_state(**overrides) -> PipelineState:
    settings = Settings(
        llm_base_url="http://localhost:4000",
        llm_api_key="test",
        enable_deliberation=False,
        enable_context_enrichment=False,
    )
    base: PipelineState = {
        "parameter": ParameterInput(
            name="max_lai",
            description="Maximum leaf area index",
            unit="m2/m2",
            domain_context="maize crop modeling",
            constraints=ConstraintSpec(lower_bound=0.0, upper_bound=12.0),
        ),
        "settings_dict": settings.model_dump(),
        "enrichment": None,
        "search_queries": [],
        "all_queries_tried": [],
        "all_papers": [],
        "papers_with_values": [],
        "seen_dois": set(),
        "agent_findings": [],
        "deliberation": None,
        "prior": None,
        "blackboard": [],
        "quality": QualityMetrics(),
        "budget": IterationBudget(),
        "warnings": [],
        "refinement_context": [],
        "trace_events": [],
    }
    base.update(overrides)
    return base


@pytest.mark.asyncio
async def test_enrich_node_disabled():
    state = _make_state()
    result = await enrich_node(state)
    assert result["enrichment"] is None
    assert len(result["trace_events"]) == 1
    assert result["trace_events"][0].node == "enrich"


@pytest.mark.asyncio
@patch("distribird.agent.search.generate_search_queries")
async def test_query_gen_node(mock_gen):
    mock_gen.return_value = ["maize LAI", "corn leaf area"]
    state = _make_state()
    result = await query_gen_node(state)
    assert result["search_queries"] == ["maize LAI", "corn leaf area"]
    assert len(result["all_queries_tried"]) == 2


@pytest.mark.asyncio
@patch("distribird.agent.search.search_all_queries", new_callable=AsyncMock)
async def test_search_node(mock_search):
    papers = [
        LiteratureEvidence(title="P1", doi="10.1/a"),
        LiteratureEvidence(title="P2", doi="10.1/b"),
    ]
    mock_search.return_value = papers
    state = _make_state(search_queries=["maize LAI"])
    await search_node(state)
    assert len(state["all_papers"]) == 2


@pytest.mark.asyncio
async def test_fetch_fulltext_node_no_papers():
    state = _make_state()
    result = await fetch_fulltext_node(state)
    assert any(t.node == "fetch_fulltext" for t in result["trace_events"])


@pytest.mark.asyncio
@patch("distribird.agent.extract.extract_all_values")
async def test_extract_node(mock_extract):
    papers = [
        LiteratureEvidence(
            title="P1",
            extracted_values=[ExtractedValue(reported_value=5.0)],
        ),
    ]
    mock_extract.return_value = papers
    state = _make_state(all_papers=papers)
    result = await extract_node(state)
    assert len(result["papers_with_values"]) == 1


@pytest.mark.asyncio
async def test_quality_gate_node():
    papers = [
        LiteratureEvidence(
            title="P1",
            extracted_values=[ExtractedValue(reported_value=5.0)],
        ),
    ]
    state = _make_state(
        all_papers=papers,
        papers_with_values=papers,
    )
    result = await quality_gate_node(state)
    assert result["quality"].n_total_values == 1


@pytest.mark.asyncio
async def test_synthesize_node():
    papers = [
        LiteratureEvidence(
            title="P1",
            extracted_values=[
                ExtractedValue(reported_value=5.0),
                ExtractedValue(reported_value=6.0),
            ],
        ),
        LiteratureEvidence(
            title="P2",
            extracted_values=[
                ExtractedValue(reported_value=5.5),
                ExtractedValue(reported_value=7.0),
                ExtractedValue(reported_value=4.5),
            ],
        ),
    ]
    state = _make_state(papers_with_values=papers)
    result = await synthesize_node(state)
    assert result["prior"] is not None
    assert result["prior"].is_informative


def test_route_after_quality_gate_to_synthesize():
    state = _make_state(
        quality=QualityMetrics(n_papers_found=3, n_total_values=5),
    )
    assert route_after_quality_gate(state) == "synthesize"


def test_route_after_quality_gate_to_refine_search():
    state = _make_state(
        quality=QualityMetrics(n_papers_found=5, n_total_values=0),
        budget=IterationBudget(search_refinement_max=2, search_refinement_used=0),
    )
    assert route_after_quality_gate(state) == "refine_search"


def test_route_after_quality_gate_to_refine_extraction():
    state = _make_state(
        quality=QualityMetrics(
            n_papers_found=5,
            n_total_values=3,
            n_high_confidence_values=0,
            value_cv=2.0,
        ),
        budget=IterationBudget(extraction_refinement_max=1, extraction_refinement_used=0),
    )
    assert route_after_quality_gate(state) == "refine_extraction"


def test_route_after_quality_gate_budget_exhausted():
    state = _make_state(
        quality=QualityMetrics(n_papers_found=5, n_total_values=0),
        budget=IterationBudget(search_refinement_max=2, search_refinement_used=2),
    )
    assert route_after_quality_gate(state) == "synthesize"


def test_route_after_deliberation_no_cross_enrich():
    state = _make_state(
        all_papers=[LiteratureEvidence(title="P1", relevance_score=0.3)],
    )
    assert route_after_deliberation(state) == "fetch_fulltext"


def test_route_after_deliberation_cross_enrich():
    state = _make_state(
        all_papers=[
            LiteratureEvidence(title="P1", relevance_score=0.8),
            LiteratureEvidence(title="P2", relevance_score=0.7),
        ],
        budget=IterationBudget(cross_enrichment_max=1, cross_enrichment_used=0),
    )
    assert route_after_deliberation(state) == "cross_enrich"


@pytest.mark.asyncio
@patch("distribird.agent.search.judge_paper_relevance")
async def test_relevance_judge_node(mock_judge):
    mock_judge.return_value = 1  # 1 LLM call
    papers = [
        LiteratureEvidence(title="P1", doi="10.1/a", abstract="LAI was 5.8"),
        LiteratureEvidence(title="P2", doi="10.1/b", abstract="Methodology paper"),
    ]
    state = _make_state(all_papers=papers)
    # Enable relevance judgment in settings
    settings = Settings(
        llm_base_url="http://localhost:4000",
        llm_api_key="test",
        enable_deliberation=False,
        enable_context_enrichment=False,
        enable_relevance_judgment=True,
    )
    state["settings_dict"] = settings.model_dump()
    result = await relevance_judge_node(state)
    assert any(t.node == "relevance_judge" for t in result["trace_events"])
    mock_judge.assert_called_once()


@pytest.mark.asyncio
async def test_relevance_judge_node_disabled():
    papers = [LiteratureEvidence(title="P1", doi="10.1/a")]
    settings = Settings(
        llm_base_url="http://localhost:4000",
        llm_api_key="test",
        enable_deliberation=False,
        enable_context_enrichment=False,
        enable_relevance_judgment=False,
    )
    state = _make_state(all_papers=papers)
    state["settings_dict"] = settings.model_dump()
    result = await relevance_judge_node(state)
    assert any(t.node == "relevance_judge" for t in result["trace_events"])

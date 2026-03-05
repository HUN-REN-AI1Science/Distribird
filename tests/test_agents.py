"""Tests for source agent implementations."""

from unittest.mock import AsyncMock, patch

import pytest

from litopri.config import Settings
from litopri.models import (
    AgentFinding,
    ConstraintSpec,
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
        enable_deliberation=True,
        enable_llm_deep_research=True,
    )


@pytest.fixture
def sample_papers():
    return [
        LiteratureEvidence(
            title=f"Paper {i}",
            doi=f"10.1234/test{i}",
            abstract=f"LAI was {4 + i}.",
            year=2022,
            verified=True,
            source="semantic_scholar",
        )
        for i in range(3)
    ]


@pytest.mark.asyncio
@patch("litopri.agent.agents.search_all_queries", new_callable=AsyncMock)
async def test_semantic_scholar_agent(mock_search, parameter, settings, sample_papers):
    from litopri.agent.agents import SemanticScholarAgent

    mock_search.return_value = sample_papers
    agent = SemanticScholarAgent()
    finding = await agent.search(parameter, ["maize LAI"], settings)

    assert isinstance(finding, AgentFinding)
    assert finding.agent_name == "semantic_scholar"
    assert finding.source_type == "semantic_scholar"
    assert len(finding.papers) == 3
    mock_search.assert_called_once_with(["maize LAI"], settings)


@pytest.mark.asyncio
@patch("litopri.agent.agents.verify_deep_research_papers", new_callable=AsyncMock)
@patch("litopri.agent.agents._llm_json_call")
@patch("litopri.agent.agents.OpenAI")
async def test_web_search_agent(mock_openai, mock_llm_call, mock_verify, parameter, settings):
    from litopri.agent.agents import WebSearchAgent

    mock_llm_call.return_value = [
        {
            "title": "Web Paper 1",
            "authors": ["Author A"],
            "year": 2023,
            "doi": "10.1234/web1",
            "abstract": "LAI was 5.5",
            "confidence": "high",
        }
    ]
    verified_paper = LiteratureEvidence(
        title="Web Paper 1",
        doi="10.1234/web1",
        abstract="LAI was 5.5",
        year=2023,
        verified=True,
        source="web_search",
    )
    mock_verify.return_value = ([verified_paper], 0)

    agent = WebSearchAgent()
    finding = await agent.search(parameter, ["maize LAI"], settings)

    assert isinstance(finding, AgentFinding)
    assert finding.agent_name == "web_search"
    assert finding.source_type == "web_search"
    assert len(finding.papers) == 1
    mock_verify.assert_called_once()


@pytest.mark.asyncio
@patch("litopri.agent.agents.search_openalex_all_queries", new_callable=AsyncMock)
async def test_openalex_agent(mock_search, parameter, settings, sample_papers):
    from litopri.agent.agents import OpenAlexAgent

    mock_search.return_value = sample_papers
    agent = OpenAlexAgent()
    finding = await agent.search(parameter, ["maize LAI"], settings)

    assert isinstance(finding, AgentFinding)
    assert finding.agent_name == "openalex"
    assert finding.source_type == "openalex"
    assert len(finding.papers) == 3
    mock_search.assert_called_once_with(["maize LAI"], settings)


@pytest.mark.asyncio
@patch("litopri.agent.agents.verify_deep_research_papers", new_callable=AsyncMock)
@patch("litopri.agent.agents.llm_deep_research", new_callable=AsyncMock)
async def test_deep_research_agent(mock_deep, mock_verify, parameter, settings):
    from litopri.agent.agents import DeepResearchAgent

    raw_papers = [
        LiteratureEvidence(
            title="Deep Paper 1",
            doi="10.1234/deep1",
            abstract="LAI was 6.0",
            year=2021,
            verified=False,
            source="llm_deep_research",
        )
    ]
    verified_papers = [
        LiteratureEvidence(
            title="Deep Paper 1",
            doi="10.1234/deep1",
            abstract="LAI was 6.0",
            year=2021,
            verified=True,
            source="llm_deep_research",
        )
    ]
    mock_deep.return_value = raw_papers
    mock_verify.return_value = (verified_papers, 0)

    agent = DeepResearchAgent()
    finding = await agent.search(parameter, ["maize LAI"], settings)

    assert isinstance(finding, AgentFinding)
    assert finding.agent_name == "deep_research"
    assert finding.source_type == "llm_deep_research"
    assert len(finding.papers) == 1
    mock_deep.assert_called_once()
    mock_verify.assert_called_once()


@pytest.mark.asyncio
@patch("litopri.agent.agents.verify_deep_research_papers", new_callable=AsyncMock)
@patch("litopri.agent.agents.llm_deep_research", new_callable=AsyncMock)
async def test_deep_research_agent_with_discards(mock_deep, mock_verify, parameter, settings):
    mock_deep.return_value = [
        LiteratureEvidence(title="Paper A", doi="10.1234/a", abstract="x"),
        LiteratureEvidence(title="Paper B", doi="10.1234/b", abstract="y"),
    ]
    mock_verify.return_value = (
        [LiteratureEvidence(title="Paper A", doi="10.1234/a", abstract="x", verified=True)],
        1,
    )

    from litopri.agent.agents import DeepResearchAgent

    agent = DeepResearchAgent()
    finding = await agent.search(parameter, [], settings)

    assert len(finding.papers) == 1
    assert finding.search_metadata["n_discarded"] == 1

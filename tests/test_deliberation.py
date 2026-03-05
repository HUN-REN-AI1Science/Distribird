"""Tests for the deliberation module."""

from unittest.mock import AsyncMock, patch

import pytest

from litopri.agent.deliberation import (
    _build_deliberation_prompt,
    _deduplicate_across_agents,
    deliberate,
    run_source_agents,
)
from litopri.config import Settings
from litopri.models import (
    AgentFinding,
    ConstraintSpec,
    DeliberationResult,
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
        enable_semantic_scholar=True,
        enable_openalex=False,
        enable_llm_deep_research=True,
    )


def _make_paper(title, doi, verified=True, source="semantic_scholar", relevance=0.5):
    return LiteratureEvidence(
        title=title,
        doi=doi,
        abstract=f"Abstract for {title}",
        year=2022,
        verified=verified,
        source=source,
        relevance_score=relevance,
    )


def test_deduplicate_merges_by_doi():
    findings = [
        AgentFinding(
            agent_name="agent_a",
            source_type="semantic_scholar",
            papers=[_make_paper("Paper 1", "10.1234/p1", verified=True)],
        ),
        AgentFinding(
            agent_name="agent_b",
            source_type="llm_deep_research",
            papers=[
                _make_paper(
                    "Paper 1 (variant)", "10.1234/p1",
                    verified=False, source="llm_deep_research",
                ),
                _make_paper("Paper 2", "10.1234/p2", verified=False, source="llm_deep_research"),
            ],
        ),
    ]

    papers, sources = _deduplicate_across_agents(findings)

    assert len(papers) == 2
    # Paper 1 should keep verified version
    assert papers[0].verified is True
    assert papers[0].title == "Paper 1"
    # Paper 1 found by both agents
    assert sources[0] == ["agent_a", "agent_b"]
    # Paper 2 found by only agent_b
    assert sources[1] == ["agent_b"]


def test_deduplicate_prefers_verified():
    findings = [
        AgentFinding(
            agent_name="llm",
            source_type="llm_deep_research",
            papers=[_make_paper("Unverified", "10.1234/x", verified=False)],
        ),
        AgentFinding(
            agent_name="s2",
            source_type="semantic_scholar",
            papers=[_make_paper("Verified", "10.1234/x", verified=True)],
        ),
    ]

    papers, sources = _deduplicate_across_agents(findings)

    assert len(papers) == 1
    assert papers[0].verified is True
    assert papers[0].title == "Verified"
    assert sources[0] == ["llm", "s2"]


def test_deduplicate_no_doi_not_merged():
    findings = [
        AgentFinding(
            agent_name="a",
            source_type="semantic_scholar",
            papers=[_make_paper("No DOI Paper", None)],
        ),
        AgentFinding(
            agent_name="b",
            source_type="llm_deep_research",
            papers=[_make_paper("Another No DOI", None)],
        ),
    ]

    papers, sources = _deduplicate_across_agents(findings)

    assert len(papers) == 2


def test_build_deliberation_prompt(parameter):
    papers = [
        _make_paper("Paper A", "10.1234/a"),
        _make_paper("Paper B", "10.1234/b", verified=False),
    ]
    paper_sources = {0: ["s2", "deep"], 1: ["deep"]}

    prompt = _build_deliberation_prompt(papers, paper_sources, parameter)

    assert "[1]" in prompt
    assert "[2]" in prompt
    assert "Paper A" in prompt
    assert "Paper B" in prompt
    assert "s2, deep" in prompt
    assert "max_lai" in prompt


@pytest.mark.asyncio
@patch("litopri.agent.deliberation._llm_json_call")
@patch("litopri.agent.deliberation.OpenAI")
async def test_deliberate_with_mock_llm(mock_openai, mock_llm_call, parameter, settings):
    findings = [
        AgentFinding(
            agent_name="s2",
            source_type="semantic_scholar",
            papers=[
                _make_paper("Good Paper", "10.1234/good", verified=True),
                _make_paper("Bad Paper", "10.1234/bad", verified=True, relevance=0.1),
            ],
        ),
        AgentFinding(
            agent_name="deep",
            source_type="llm_deep_research",
            papers=[
                _make_paper(
                    "Good Paper", "10.1234/good",
                    verified=False, source="llm_deep_research",
                ),
                _make_paper("New Paper", "10.1234/new", verified=True, source="llm_deep_research"),
            ],
        ),
    ]

    mock_llm_call.return_value = {
        "selected_papers": [1, 3],
        "excluded_papers": [2],
        "rationale": "Selected relevant verified papers",
        "warnings": [],
    }

    result = await deliberate(findings, parameter, settings)

    assert isinstance(result, DeliberationResult)
    assert len(result.consensus_papers) == 2
    assert len(result.excluded_papers) == 1
    assert result.moderator_rationale == "Selected relevant verified papers"


@pytest.mark.asyncio
@patch("litopri.agent.deliberation._llm_json_call")
@patch("litopri.agent.deliberation.OpenAI")
async def test_deliberate_llm_failure_fallback(mock_openai, mock_llm_call, parameter, settings):
    findings = [
        AgentFinding(
            agent_name="s2",
            source_type="semantic_scholar",
            papers=[_make_paper("Verified A", "10.1234/a", verified=True)],
        ),
        AgentFinding(
            agent_name="deep",
            source_type="llm_deep_research",
            papers=[_make_paper("Unverified B", "10.1234/b", verified=False)],
        ),
    ]

    mock_llm_call.side_effect = Exception("LLM timeout")

    result = await deliberate(findings, parameter, settings)

    # Falls back to all verified papers
    assert len(result.consensus_papers) == 1
    assert result.consensus_papers[0].title == "Verified A"
    assert len(result.excluded_papers) == 1
    assert any("failed" in w.lower() for w in result.warnings)


@pytest.mark.asyncio
async def test_deliberate_single_agent_all_verified(parameter, settings):
    findings = [
        AgentFinding(
            agent_name="s2",
            source_type="semantic_scholar",
            papers=[
                _make_paper("Paper A", "10.1234/a", verified=True),
                _make_paper("Paper B", "10.1234/b", verified=True),
            ],
        ),
    ]

    result = await deliberate(findings, parameter, settings)

    # Should skip LLM call
    assert len(result.consensus_papers) == 2
    assert "no deliberation needed" in result.moderator_rationale.lower()


@pytest.mark.asyncio
async def test_deliberate_empty_findings(parameter, settings):
    result = await deliberate([], parameter, settings)

    assert len(result.consensus_papers) == 0
    assert any("no papers" in w.lower() for w in result.warnings)


@pytest.mark.asyncio
@patch("litopri.agent.deliberation.DeepResearchAgent")
@patch("litopri.agent.deliberation.SemanticScholarAgent")
async def test_run_source_agents_parallel(mock_s2_cls, mock_deep_cls, parameter, settings):
    s2_instance = mock_s2_cls.return_value
    s2_instance.name = "semantic_scholar"
    s2_instance.search = AsyncMock(
        return_value=AgentFinding(
            agent_name="semantic_scholar",
            source_type="semantic_scholar",
            papers=[_make_paper("S2 Paper", "10.1234/s2")],
        )
    )

    deep_instance = mock_deep_cls.return_value
    deep_instance.name = "deep_research"
    deep_instance.search = AsyncMock(
        return_value=AgentFinding(
            agent_name="deep_research",
            source_type="llm_deep_research",
            papers=[_make_paper("Deep Paper", "10.1234/deep")],
        )
    )

    findings = await run_source_agents(parameter, ["query"], settings)

    assert len(findings) == 2
    s2_instance.search.assert_called_once()
    deep_instance.search.assert_called_once()


@pytest.mark.asyncio
@patch("litopri.agent.deliberation.DeepResearchAgent")
@patch("litopri.agent.deliberation.SemanticScholarAgent")
async def test_run_source_agents_one_fails(mock_s2_cls, mock_deep_cls, parameter, settings):
    s2_instance = mock_s2_cls.return_value
    s2_instance.name = "semantic_scholar"
    s2_instance.search = AsyncMock(side_effect=Exception("S2 API down"))

    deep_instance = mock_deep_cls.return_value
    deep_instance.name = "deep_research"
    deep_instance.search = AsyncMock(
        return_value=AgentFinding(
            agent_name="deep_research",
            source_type="llm_deep_research",
            papers=[_make_paper("Deep Paper", "10.1234/deep")],
        )
    )

    findings = await run_source_agents(parameter, ["query"], settings)

    # One failed, one succeeded
    assert len(findings) == 1
    assert findings[0].agent_name == "deep_research"


@pytest.mark.asyncio
async def test_run_source_agents_respects_settings(parameter):
    settings = Settings(
        llm_base_url="http://localhost:4000",
        llm_api_key="test",
        enable_semantic_scholar=False,
        enable_openalex=False,
        enable_llm_deep_research=False,
        enable_web_search_agent=False,
    )

    findings = await run_source_agents(parameter, ["query"], settings)

    assert len(findings) == 0

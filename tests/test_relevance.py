"""Tests for LLM relevance judgment."""

from unittest.mock import patch

import pytest

from distribird.agent.search import judge_paper_relevance
from distribird.config import Settings
from distribird.models import LiteratureEvidence, ParameterInput


@pytest.fixture
def parameter():
    return ParameterInput(
        name="max_lai",
        description="Maximum leaf area index",
        unit="m2/m2",
        domain_context="maize crop modeling",
    )


@pytest.fixture
def settings():
    return Settings(
        llm_base_url="http://localhost:4000",
        llm_api_key="test",
    )


def _make_papers(n):
    return [
        LiteratureEvidence(
            title=f"Paper {i}",
            doi=f"10.1/test{i}",
            abstract=f"Abstract about LAI measurement {i}",
            relevance_score=0.3,
        )
        for i in range(n)
    ]


@patch("distribird.agent.search._llm_json_call")
def test_judge_relevance_high(mock_llm, parameter, settings):
    papers = _make_papers(1)
    mock_llm.return_value = {
        "0": {"relevance": "high", "snippet": "Peak LAI was 5.8 m2/m2"},
    }
    calls = judge_paper_relevance(papers, parameter, settings)
    assert calls == 1
    assert papers[0].relevance_score == pytest.approx(0.7 * 0.9 + 0.3 * 0.3)
    assert papers[0].relevance_snippet == "Peak LAI was 5.8 m2/m2"


@patch("distribird.agent.search._llm_json_call")
def test_judge_relevance_batching(mock_llm, parameter, settings):
    papers = _make_papers(15)
    mock_llm.return_value = {
        str(i): {"relevance": "medium", "snippet": f"snippet {i}"} for i in range(10)
    }
    calls = judge_paper_relevance(papers, parameter, settings, batch_size=10)
    assert calls == 2


@patch("distribird.agent.search._llm_json_call")
def test_judge_relevance_llm_failure(mock_llm, parameter, settings):
    papers = _make_papers(3)
    original_scores = [p.relevance_score for p in papers]
    mock_llm.side_effect = Exception("LLM error")
    calls = judge_paper_relevance(papers, parameter, settings)
    assert calls == 1
    # Scores should be unchanged on failure
    for p, orig in zip(papers, original_scores):
        assert p.relevance_score == orig


@patch("distribird.agent.search._llm_json_call")
def test_score_blending(mock_llm, parameter, settings):
    papers = [
        LiteratureEvidence(
            title="P1",
            doi="10.1/a",
            abstract="...",
            relevance_score=0.6,
        ),
    ]
    mock_llm.return_value = {
        "0": {"relevance": "medium", "snippet": "some snippet"},
    }
    judge_paper_relevance(papers, parameter, settings)
    expected = 0.7 * 0.5 + 0.3 * 0.6  # 0.53
    assert papers[0].relevance_score == pytest.approx(expected)


@patch("distribird.agent.search._llm_json_call")
def test_skip_already_judged(mock_llm, parameter, settings):
    papers = [
        LiteratureEvidence(
            title="P1",
            doi="10.1/a",
            abstract="...",
            relevance_score=0.8,
            relevance_snippet="already judged",
        ),
    ]
    calls = judge_paper_relevance(papers, parameter, settings)
    assert calls == 0
    mock_llm.assert_not_called()

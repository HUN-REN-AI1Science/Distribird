"""Tests for literature search with mocked APIs."""

from unittest.mock import MagicMock, patch

import httpx
import pytest
import respx

from litopri.agent.search import (
    _compute_relevance,
    llm_deep_research,
    search_semantic_scholar,
    verify_deep_research_papers,
    verify_paper_doi,
)
from litopri.config import Settings
from litopri.models import ConstraintSpec, LiteratureEvidence, ParameterInput


@pytest.fixture
def settings():
    return Settings(
        semantic_scholar_base_url="https://api.semanticscholar.org/graph/v1",
        semantic_scholar_api_key="test-key",
    )


@pytest.fixture
def mock_s2_response():
    return {
        "total": 2,
        "data": [
            {
                "paperId": "abc123",
                "title": "Maize leaf area index measurements",
                "authors": [{"name": "Smith J"}, {"name": "Doe A"}],
                "year": 2020,
                "externalIds": {"DOI": "10.1234/test1"},
                "abstract": "We measured LAI of maize at 5.2 m2/m2.",
                "citationCount": 50,
                "openAccessPdf": {"url": "https://example.com/test1.pdf"},
            },
            {
                "paperId": "def456",
                "title": "Crop model calibration study",
                "authors": [{"name": "Jones B"}],
                "year": 2021,
                "externalIds": {"DOI": "10.1234/test2"},
                "abstract": "LAI ranged from 3.1 to 7.8.",
                "citationCount": 20,
                "openAccessPdf": None,
            },
        ],
    }


@respx.mock
@pytest.mark.asyncio
async def test_search_semantic_scholar(settings, mock_s2_response):
    respx.get("https://api.semanticscholar.org/graph/v1/paper/search").mock(
        return_value=httpx.Response(200, json=mock_s2_response)
    )

    results = await search_semantic_scholar("maize LAI", settings)
    assert len(results) == 1  # only OA paper passes filter
    assert results[0].title == "Maize leaf area index measurements"
    assert results[0].doi == "10.1234/test1"
    assert results[0].authors == ["Smith J", "Doe A"]
    assert results[0].pdf_url == "https://example.com/test1.pdf"


@respx.mock
@pytest.mark.asyncio
async def test_search_includes_all_oa_papers(settings):
    all_oa_response = {
        "total": 2,
        "data": [
            {
                "paperId": "abc123",
                "title": "Paper A",
                "authors": [{"name": "Smith J"}],
                "year": 2020,
                "externalIds": {"DOI": "10.1234/a"},
                "abstract": "Abstract A",
                "citationCount": 10,
                "openAccessPdf": {"url": "https://example.com/a.pdf"},
            },
            {
                "paperId": "def456",
                "title": "Paper B",
                "authors": [{"name": "Jones B"}],
                "year": 2021,
                "externalIds": {"DOI": "10.1234/b"},
                "abstract": "Abstract B",
                "citationCount": 5,
                "openAccessPdf": {"url": "https://example.com/b.pdf"},
            },
        ],
    }
    respx.get("https://api.semanticscholar.org/graph/v1/paper/search").mock(
        return_value=httpx.Response(200, json=all_oa_response)
    )

    results = await search_semantic_scholar("test", settings)
    assert len(results) == 2


@respx.mock
@pytest.mark.asyncio
async def test_search_empty_results(settings):
    respx.get("https://api.semanticscholar.org/graph/v1/paper/search").mock(
        return_value=httpx.Response(200, json={"total": 0, "data": []})
    )

    results = await search_semantic_scholar("nonexistent query", settings)
    assert len(results) == 0


@respx.mock
@pytest.mark.asyncio
async def test_search_handles_error(settings):
    respx.get("https://api.semanticscholar.org/graph/v1/paper/search").mock(
        return_value=httpx.Response(500)
    )

    with pytest.raises(httpx.HTTPStatusError):
        await search_semantic_scholar("test query", settings)


class TestComputeRelevance:
    """Tests for the relevance scoring function."""

    def test_zero_citations(self):
        score = _compute_relevance(0, 2020)
        assert score >= 0.0
        assert score <= 1.0

    def test_high_citations(self):
        score = _compute_relevance(100, 2020)
        assert score > 0.0
        assert score <= 1.0

    def test_recent_paper_bonus(self):
        import datetime

        current_year = datetime.datetime.now(datetime.timezone.utc).year
        recent = _compute_relevance(10, current_year)
        old = _compute_relevance(10, current_year - 10)
        assert recent > old

    def test_capped_at_one(self):
        score = _compute_relevance(10000, 2024)
        assert score == 1.0

    def test_none_year(self):
        score = _compute_relevance(50, None)
        assert 0.0 <= score <= 1.0


class TestDeepResearchDedicatedModel:
    """Test that deep research always uses the dedicated o4-mini model."""

    @pytest.fixture
    def parameter(self):
        return ParameterInput(
            name="max_lai",
            description="Maximum leaf area index",
            unit="m2/m2",
            domain_context="maize",
            constraints=ConstraintSpec(lower_bound=0, upper_bound=12),
        )

    @patch("litopri.agent.search.OpenAI")
    @pytest.mark.asyncio
    async def test_uses_dedicated_model(self, mock_openai_cls, parameter):
        """Deep research uses dedicated endpoint and web prompt."""
        import json

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        papers_data = [
            {
                "title": "Paper A",
                "authors": ["X"],
                "year": 2020,
                "doi": None,
                "abstract": "LAI=5",
                "confidence": "high",
            },
        ]
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=json.dumps(papers_data)))]
        mock_client.chat.completions.create.return_value = mock_response

        s = Settings()
        results = await llm_deep_research(parameter, s)

        mock_openai_cls.assert_called_once_with(
            base_url=s.deep_research_base_url,
            api_key=s.deep_research_api_key,
        )
        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "o4-mini-deep-research"
        messages = call_kwargs.kwargs["messages"]
        user_content = next(m["content"] for m in messages if m["role"] == "user")
        assert "Search the web" in user_content
        assert len(results) == 1


class TestDeepResearchConfidence:
    """Tests for deep research confidence → relevance mapping."""

    @pytest.fixture
    def parameter(self):
        return ParameterInput(
            name="max_lai",
            description="Maximum leaf area index",
            unit="m2/m2",
            domain_context="maize",
            constraints=ConstraintSpec(lower_bound=0, upper_bound=12),
        )

    @patch("litopri.agent.search.OpenAI")
    @pytest.mark.asyncio
    async def test_confidence_maps_to_relevance(self, mock_openai_cls, parameter, settings):
        import json

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        papers_data = [
            {
                "title": "Paper A",
                "authors": ["X"],
                "year": 2020,
                "doi": None,
                "abstract": "LAI=5",
                "confidence": "high",
            },
            {
                "title": "Paper B",
                "authors": ["Y"],
                "year": 2019,
                "doi": None,
                "abstract": "LAI=6",
                "confidence": "medium",
            },
            {
                "title": "Paper C",
                "authors": ["Z"],
                "year": 2018,
                "doi": None,
                "abstract": "LAI=7",
                "confidence": "low",
            },
        ]
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=json.dumps(papers_data)))]
        mock_client.chat.completions.create.return_value = mock_response

        results = await llm_deep_research(parameter, settings)
        assert len(results) == 3
        assert results[0].relevance_score == 0.5  # high
        assert results[1].relevance_score == 0.3  # medium
        assert results[2].relevance_score == 0.1  # low


class TestVerifyPaperDoi:
    """Tests for DOI verification against Semantic Scholar."""

    @pytest.fixture
    def s2_paper_response(self):
        return {
            "title": "Verified Title",
            "authors": [{"name": "Author A"}, {"name": "Author B"}],
            "year": 2021,
            "externalIds": {"DOI": "10.1234/verified"},
            "abstract": "Verified abstract text.",
            "citationCount": 42,
            "openAccessPdf": {"url": "https://example.com/paper.pdf"},
        }

    @respx.mock
    @pytest.mark.asyncio
    async def test_verify_paper_doi_valid(self, settings, s2_paper_response):
        paper = LiteratureEvidence(
            title="LLM Title",
            doi="10.1234/verified",
            source="llm_deep_research",
        )
        respx.get("https://api.semanticscholar.org/graph/v1/paper/DOI:10.1234/verified").mock(
            return_value=httpx.Response(200, json=s2_paper_response)
        )

        result = await verify_paper_doi(paper, settings)
        assert result is not None
        assert result.verified is True
        assert result.title == "Verified Title"
        assert result.authors == ["Author A", "Author B"]
        assert result.year == 2021
        assert result.abstract == "Verified abstract text."
        assert result.pdf_url == "https://example.com/paper.pdf"

    @respx.mock
    @pytest.mark.asyncio
    async def test_verify_paper_doi_not_found(self, settings):
        paper = LiteratureEvidence(
            title="Fake Paper",
            doi="10.9999/fake",
            source="llm_deep_research",
        )
        respx.get("https://api.semanticscholar.org/graph/v1/paper/DOI:10.9999/fake").mock(
            return_value=httpx.Response(404)
        )

        result = await verify_paper_doi(paper, settings)
        assert result is None

    @pytest.mark.asyncio
    async def test_verify_paper_no_doi(self, settings):
        paper = LiteratureEvidence(
            title="No DOI Paper",
            doi=None,
            source="llm_deep_research",
        )
        result = await verify_paper_doi(paper, settings)
        assert result is None

    @respx.mock
    @pytest.mark.asyncio
    async def test_verify_paper_doi_normalized(self, settings, s2_paper_response):
        paper = LiteratureEvidence(
            title="LLM Title",
            doi="https://doi.org/10.1234/verified",
            source="llm_deep_research",
        )
        respx.get("https://api.semanticscholar.org/graph/v1/paper/DOI:10.1234/verified").mock(
            return_value=httpx.Response(200, json=s2_paper_response)
        )

        result = await verify_paper_doi(paper, settings)
        assert result is not None
        assert result.verified is True

    @respx.mock
    @pytest.mark.asyncio
    async def test_verify_deep_research_papers_batch(self, settings, s2_paper_response):
        papers = [
            LiteratureEvidence(title="Valid", doi="10.1234/verified", source="llm_deep_research"),
            LiteratureEvidence(title="Invalid", doi="10.9999/fake", source="llm_deep_research"),
            LiteratureEvidence(title="No DOI", doi=None, source="llm_deep_research"),
        ]
        respx.get("https://api.semanticscholar.org/graph/v1/paper/DOI:10.1234/verified").mock(
            return_value=httpx.Response(200, json=s2_paper_response)
        )
        respx.get("https://api.semanticscholar.org/graph/v1/paper/DOI:10.9999/fake").mock(
            return_value=httpx.Response(404)
        )

        verified, n_discarded = await verify_deep_research_papers(papers, settings)
        assert len(verified) == 1
        assert n_discarded == 2
        assert verified[0].verified is True

    @respx.mock
    @pytest.mark.asyncio
    async def test_verify_paper_doi_timeout(self, settings):
        paper = LiteratureEvidence(
            title="Timeout Paper",
            doi="10.1234/timeout",
            source="llm_deep_research",
        )
        respx.get("https://api.semanticscholar.org/graph/v1/paper/DOI:10.1234/timeout").mock(
            side_effect=httpx.ConnectTimeout("connection timed out")
        )

        result = await verify_paper_doi(paper, settings)
        assert result is None

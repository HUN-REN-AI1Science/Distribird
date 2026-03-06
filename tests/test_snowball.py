"""Tests for citation snowballing."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litopri.agent.search import fetch_citations, snowball_papers
from litopri.config import Settings
from litopri.models import LiteratureEvidence


@pytest.fixture
def settings():
    return Settings(
        llm_base_url="http://localhost:4000",
        llm_api_key="test",
        semantic_scholar_base_url="https://api.semanticscholar.org/graph/v1",
    )


def _s2_citation_response(n, direction="citations"):
    """Build a mock S2 citations/references response."""
    key = "citingPaper" if direction == "citations" else "citedPaper"
    return {
        "data": [
            {
                key: {
                    "title": f"Citing Paper {i}",
                    "authors": [{"name": f"Author {i}"}],
                    "year": 2022 + i,
                    "externalIds": {"DOI": f"10.1/cite{i}"},
                    "abstract": f"Abstract for citing paper {i}",
                    "citationCount": 10 + i,
                    "openAccessPdf": {"url": f"https://example.com/pdf{i}"},
                }
            }
            for i in range(n)
        ]
    }


@pytest.mark.asyncio
@patch("litopri.agent.search.httpx.AsyncClient")
async def test_fetch_citations_forward(mock_client_cls, settings):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = _s2_citation_response(3, "citations")
    mock_resp.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get.return_value = mock_resp
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client_cls.return_value = mock_client

    papers = await fetch_citations("DOI:10.1/seed", settings, "citations", limit=5)
    assert len(papers) == 3
    assert all(p.source == "snowball_citation" for p in papers)
    assert all(p.verified for p in papers)


@pytest.mark.asyncio
@patch("litopri.agent.search.httpx.AsyncClient")
async def test_fetch_references_backward(mock_client_cls, settings):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = _s2_citation_response(2, "references")
    mock_resp.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get.return_value = mock_resp
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client_cls.return_value = mock_client

    papers = await fetch_citations("DOI:10.1/seed", settings, "references", limit=5)
    assert len(papers) == 2
    assert all(p.source == "snowball_reference" for p in papers)


@pytest.mark.asyncio
@patch("litopri.agent.search.fetch_citations", new_callable=AsyncMock)
@patch("litopri.agent.search.asyncio.sleep", new_callable=AsyncMock)
async def test_snowball_dedup(mock_sleep, mock_fetch, settings):
    """Snowball deduplicates by DOI across seeds and existing papers."""
    # Both seeds return some overlapping papers
    mock_fetch.side_effect = [
        # seed 1 forward
        [LiteratureEvidence(title="P1", doi="10.1/a", source="snowball_citation")],
        # seed 1 backward
        [LiteratureEvidence(title="P2", doi="10.1/b", source="snowball_reference")],
        # seed 2 forward
        [LiteratureEvidence(title="P1-dup", doi="10.1/a", source="snowball_citation")],
        # seed 2 backward
        [LiteratureEvidence(title="P3", doi="10.1/c", source="snowball_reference")],
    ]

    seeds = [
        LiteratureEvidence(title="Seed1", doi="10.1/seed1", relevance_score=0.9),
        LiteratureEvidence(title="Seed2", doi="10.1/seed2", relevance_score=0.8),
    ]
    existing = {"10.1/b"}  # P2 already exists

    result = await snowball_papers(seeds, settings, existing, max_seeds=2, limit_per_seed=5)
    dois = {p.doi for p in result}
    assert "10.1/a" in dois
    assert "10.1/c" in dois
    assert "10.1/b" not in dois  # was in existing
    assert len(result) == 2  # P1 and P3 only (P1-dup and P2 excluded)


@pytest.mark.asyncio
@patch("litopri.agent.search.httpx.AsyncClient")
async def test_snowball_oa_filter(mock_client_cls, settings):
    """Non-OA papers are excluded."""
    mock_resp = MagicMock()
    resp_data = {
        "data": [
            {
                "citingPaper": {
                    "title": "OA Paper",
                    "authors": [],
                    "year": 2023,
                    "externalIds": {"DOI": "10.1/oa"},
                    "abstract": "Has OA PDF",
                    "citationCount": 5,
                    "openAccessPdf": {"url": "https://example.com/pdf"},
                }
            },
            {
                "citingPaper": {
                    "title": "Closed Paper",
                    "authors": [],
                    "year": 2023,
                    "externalIds": {"DOI": "10.1/closed"},
                    "abstract": "No OA PDF",
                    "citationCount": 50,
                    "openAccessPdf": None,
                }
            },
        ]
    }
    mock_resp.json.return_value = resp_data
    mock_resp.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get.return_value = mock_resp
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client_cls.return_value = mock_client

    papers = await fetch_citations("DOI:10.1/seed", settings, "citations", limit=10)
    assert len(papers) == 1
    assert papers[0].title == "OA Paper"

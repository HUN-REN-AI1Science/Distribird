"""Tests for OpenAlex literature search."""

import httpx
import pytest
import respx

from litopri.agent.search_openalex import (
    _normalize_openalex_doi,
    reconstruct_abstract,
    search_openalex,
    search_openalex_all_queries,
)
from litopri.config import Settings


@pytest.fixture
def settings():
    return Settings(
        enable_openalex=True,
        openalex_email="test@example.com",
    )


@pytest.fixture
def openalex_response():
    return {
        "results": [
            {
                "id": "https://openalex.org/W1234",
                "doi": "https://doi.org/10.1234/oa1",
                "title": "Open Access Maize Study",
                "publication_year": 2022,
                "cited_by_count": 30,
                "authorships": [
                    {"author": {"display_name": "Alice Smith"}},
                    {"author": {"display_name": "Bob Jones"}},
                ],
                "abstract_inverted_index": {
                    "LAI": [0],
                    "was": [1],
                    "measured": [2],
                    "at": [3],
                    "5.2": [4],
                },
                "best_oa_location": {
                    "pdf_url": "https://example.com/oa1.pdf",
                    "landing_page_url": "https://example.com/oa1",
                },
            }
        ]
    }


class TestReconstructAbstract:
    def test_normal(self):
        index = {"Hello": [0], "world": [1], "of": [2], "science": [3]}
        assert reconstruct_abstract(index) == "Hello world of science"

    def test_empty(self):
        assert reconstruct_abstract({}) == ""

    def test_none(self):
        assert reconstruct_abstract(None) == ""


class TestNormalizeOpenalexDoi:
    def test_strips_prefix(self):
        assert _normalize_openalex_doi("https://doi.org/10.1234/test") == "10.1234/test"

    def test_no_prefix(self):
        assert _normalize_openalex_doi("10.1234/test") == "10.1234/test"

    def test_none(self):
        assert _normalize_openalex_doi(None) is None

    def test_empty(self):
        assert _normalize_openalex_doi("") is None


@respx.mock
@pytest.mark.asyncio
async def test_search_openalex(settings, openalex_response):
    respx.get("https://api.openalex.org/works").mock(
        return_value=httpx.Response(200, json=openalex_response)
    )

    results = await search_openalex("maize LAI", settings)
    assert len(results) == 1
    paper = results[0]
    assert paper.title == "Open Access Maize Study"
    assert paper.doi == "10.1234/oa1"
    assert paper.year == 2022
    assert paper.authors == ["Alice Smith", "Bob Jones"]
    assert paper.pdf_url == "https://example.com/oa1.pdf"
    assert paper.verified is True
    assert paper.source == "openalex"
    assert "LAI was measured at 5.2" in paper.abstract


@respx.mock
@pytest.mark.asyncio
async def test_search_openalex_empty(settings):
    respx.get("https://api.openalex.org/works").mock(
        return_value=httpx.Response(200, json={"results": []})
    )

    results = await search_openalex("nonexistent query", settings)
    assert len(results) == 0


@respx.mock
@pytest.mark.asyncio
async def test_search_openalex_dedup(settings):
    """Duplicate DOIs across queries are deduplicated."""
    response = {
        "results": [
            {
                "id": "https://openalex.org/W1234",
                "doi": "https://doi.org/10.1234/dup",
                "title": "Duplicate Paper",
                "publication_year": 2021,
                "cited_by_count": 10,
                "authorships": [],
                "abstract_inverted_index": {"test": [0]},
                "best_oa_location": {"pdf_url": "https://example.com/dup.pdf"},
            }
        ]
    }
    respx.get("https://api.openalex.org/works").mock(
        return_value=httpx.Response(200, json=response)
    )

    results = await search_openalex_all_queries(["query1", "query2"], settings)
    # Same DOI from two queries should appear only once
    assert len(results) == 1
    assert results[0].doi == "10.1234/dup"

"""Tests for value extraction with mocked LLM."""

import json
from unittest.mock import MagicMock, patch

import pytest

from litopri.agent.extract import (
    _llm_json_call,
    _passes_bounds_check,
    extract_all_values,
    extract_values_batch,
    extract_values_from_paper,
    extract_values_web_assisted,
)
from litopri.config import Settings
from litopri.models import ConstraintSpec, ExtractedValue, LiteratureEvidence, ParameterInput


@pytest.fixture
def parameter():
    return ParameterInput(
        name="max_lai",
        description="Maximum leaf area index",
        unit="m2/m2",
        constraints=ConstraintSpec(lower_bound=0, upper_bound=12),
    )


@pytest.fixture
def paper():
    return LiteratureEvidence(
        title="Test Paper",
        doi="10.1234/test",
        abstract="The maximum LAI was 5.2 m2/m2 with a standard deviation of 0.8.",
    )


@pytest.fixture
def settings():
    return Settings(llm_base_url="http://localhost:4000", llm_api_key="test")


def test_passes_bounds_check():
    constraint = ConstraintSpec(lower_bound=0, upper_bound=10)
    assert _passes_bounds_check(ExtractedValue(reported_value=5), constraint)
    assert not _passes_bounds_check(ExtractedValue(reported_value=-1), constraint)
    assert not _passes_bounds_check(ExtractedValue(reported_value=11), constraint)
    assert _passes_bounds_check(ExtractedValue(reported_value=None), constraint)


@patch("litopri.agent.extract.OpenAI")
def test_extract_values(mock_openai_cls, parameter, paper, settings):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client

    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                content=(
                    '[{"reported_value": 5.2, "uncertainty": 0.8,'
                    ' "context": "maize field",'
                    ' "extraction_confidence": "high"}]'
                )
            )
        )
    ]
    mock_client.chat.completions.create.return_value = mock_response

    values = extract_values_from_paper(paper, parameter, settings)
    assert len(values) == 1
    assert values[0].reported_value == 5.2


@patch("litopri.agent.extract.OpenAI")
def test_extract_filters_out_of_bounds(mock_openai_cls, parameter, paper, settings):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client

    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                content='[{"reported_value": 5.2}, {"reported_value": 15.0}]'
            )
        )
    ]
    mock_client.chat.completions.create.return_value = mock_response

    values = extract_values_from_paper(paper, parameter, settings)
    assert len(values) == 1
    assert values[0].reported_value == 5.2


def test_extract_empty_abstract(parameter, settings):
    paper = LiteratureEvidence(title="Empty", abstract="")
    values = extract_values_from_paper(paper, parameter, settings)
    assert values == []


class TestLlmJsonCall:
    """Tests for the JSON retry helper."""

    def test_valid_json_first_try(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content='[1, 2, 3]'))]
        mock_client.chat.completions.create.return_value = mock_response

        result = _llm_json_call(mock_client, "model", [{"role": "user", "content": "test"}])
        assert result == [1, 2, 3]
        assert mock_client.chat.completions.create.call_count == 1

    def test_retry_on_bad_json(self):
        mock_client = MagicMock()
        bad_response = MagicMock()
        bad_response.choices = [MagicMock(message=MagicMock(content="not json at all"))]
        good_response = MagicMock()
        good_response.choices = [MagicMock(message=MagicMock(content='{"key": "value"}'))]
        mock_client.chat.completions.create.side_effect = [bad_response, good_response]

        result = _llm_json_call(mock_client, "model", [{"role": "user", "content": "test"}])
        assert result == {"key": "value"}
        assert mock_client.chat.completions.create.call_count == 2

    def test_max_retries_exceeded(self):
        mock_client = MagicMock()
        bad_response = MagicMock()
        bad_response.choices = [MagicMock(message=MagicMock(content="not json"))]
        mock_client.chat.completions.create.return_value = bad_response

        with pytest.raises(json.JSONDecodeError):
            _llm_json_call(
                mock_client, "model", [{"role": "user", "content": "test"}], max_retries=2
            )
        # 1 initial + 2 retries = 3 calls
        assert mock_client.chat.completions.create.call_count == 3

    def test_strips_code_fences(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content='```json\n[1, 2]\n```'))
        ]
        mock_client.chat.completions.create.return_value = mock_response

        result = _llm_json_call(mock_client, "model", [{"role": "user", "content": "test"}])
        assert result == [1, 2]


class TestBatchExtraction:
    """Tests for batched extraction."""

    @patch("litopri.agent.extract.OpenAI")
    def test_batch_call(self, mock_openai_cls, parameter, settings):
        papers = [
            LiteratureEvidence(title=f"Paper {i}", abstract=f"LAI was {4+i}.")
            for i in range(3)
        ]

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content=json.dumps({
                        "0": [{"reported_value": 4.0, "context": "test"}],
                        "1": [{"reported_value": 5.0, "context": "test"}],
                        "2": [],
                    })
                )
            )
        ]
        mock_client.chat.completions.create.return_value = mock_response

        results = extract_values_batch(papers, parameter, settings)
        assert len(results) == 3
        assert len(results[0]) == 1
        assert results[0][0].reported_value == 4.0
        assert len(results[1]) == 1
        assert len(results[2]) == 0

    @patch("litopri.agent.extract.OpenAI")
    def test_batch_fallback_to_per_paper(self, mock_openai_cls, parameter, settings):
        """When batch call returns non-dict, falls back to per-paper."""
        papers = [
            LiteratureEvidence(title="Paper 0", abstract="LAI was 4.0.")
        ]

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        # First call (batch) returns invalid format, second call (per-paper fallback) works
        batch_response = MagicMock()
        batch_response.choices = [MagicMock(message=MagicMock(content="not a dict"))]
        per_paper_response = MagicMock()
        per_paper_response.choices = [
            MagicMock(
                message=MagicMock(
                    content='[{"reported_value": 4.0, "context": "fallback"}]'
                )
            )
        ]
        mock_client.chat.completions.create.side_effect = [
            batch_response,  # batch fails JSON parse
            batch_response,  # retry 1
            batch_response,  # retry 2 → gives up, falls back
            per_paper_response,  # per-paper extraction
        ]

        results = extract_values_batch(papers, parameter, settings)
        assert len(results) == 1
        assert len(results[0]) == 1
        assert results[0][0].reported_value == 4.0

    @patch("litopri.agent.extract.OpenAI")
    def test_extract_all_uses_batches(self, mock_openai_cls, parameter, settings):
        """extract_all_values should chunk into batches."""
        papers = [
            LiteratureEvidence(title=f"Paper {i}", abstract=f"LAI was {4+i*0.1}.")
            for i in range(7)
        ]

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        # Two batch calls: batch of 5, then batch of 2
        batch1_response = MagicMock()
        batch1_response.choices = [
            MagicMock(
                message=MagicMock(
                    content=json.dumps({
                        str(i): [{"reported_value": 4 + i * 0.1}] for i in range(5)
                    })
                )
            )
        ]
        batch2_response = MagicMock()
        batch2_response.choices = [
            MagicMock(
                message=MagicMock(
                    content=json.dumps({
                        "0": [{"reported_value": 4.5}],
                        "1": [{"reported_value": 4.6}],
                    })
                )
            )
        ]
        mock_client.chat.completions.create.side_effect = [batch1_response, batch2_response]

        result = extract_all_values(papers, parameter, settings)
        assert len(result) == 7  # All papers should have values


class TestWebAssistedExtraction:
    """Tests for web-assisted extraction."""

    @patch("litopri.agent.extract.OpenAI")
    def test_web_assisted_extraction(self, mock_openai_cls, parameter, settings):
        """Web-assisted extraction returns values for papers found online."""
        papers = [
            LiteratureEvidence(
                title="Paper A", doi="10.1234/a", abstract="LAI study.",
                relevance_score=0.9,
            ),
            LiteratureEvidence(
                title="Paper B", doi="10.1234/b", abstract="Another study.",
                relevance_score=0.7,
            ),
        ]

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content=json.dumps({
                        "0": [{"reported_value": 6.1, "context": "Table 2",
                               "source_url": "https://doi.org/10.1234/a",
                               "extraction_confidence": "high"}],
                        "1": [],
                    })
                )
            )
        ]
        mock_client.chat.completions.create.return_value = mock_response

        result = extract_values_web_assisted(papers, parameter, settings)
        assert len(result) == 1
        assert result[0].title == "Paper A"
        assert result[0].extracted_values[0].reported_value == 6.1

    @patch("litopri.agent.extract.OpenAI")
    def test_web_assisted_no_values(self, mock_openai_cls, parameter, settings):
        """Returns empty when LLM finds no values online."""
        papers = [
            LiteratureEvidence(title="Paper X", doi="10.1234/x", abstract="Irrelevant."),
        ]

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content=json.dumps({"0": []})))
        ]
        mock_client.chat.completions.create.return_value = mock_response

        result = extract_values_web_assisted(papers, parameter, settings)
        assert result == []

    @patch("litopri.agent.extract.OpenAI")
    def test_web_assisted_max_papers_cap(self, mock_openai_cls, parameter, settings):
        """Only top max_papers are sent to LLM."""
        papers = [
            LiteratureEvidence(
                title=f"Paper {i}", abstract=f"Abstract {i}",
                relevance_score=i / 15.0,
            )
            for i in range(15)
        ]

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        # Return empty for all — we just care about how many batches are called
        mock_response.choices = [
            MagicMock(message=MagicMock(content=json.dumps({})))
        ]
        mock_client.chat.completions.create.return_value = mock_response

        extract_values_web_assisted(papers, parameter, settings, max_papers=10)

        # 10 papers in batches of 3 → ceil(10/3) = 4 calls
        assert mock_client.chat.completions.create.call_count == 4

    @patch("litopri.agent.extract.OpenAI")
    def test_web_assisted_includes_doi(self, mock_openai_cls, parameter, settings):
        """Verify the prompt sent to LLM includes paper DOIs."""
        papers = [
            LiteratureEvidence(
                title="DOI Paper", doi="10.9999/testdoi", abstract="Test.",
            ),
        ]

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content=json.dumps({"0": []})))
        ]
        mock_client.chat.completions.create.return_value = mock_response

        extract_values_web_assisted(papers, parameter, settings)

        # Check that the prompt contains the DOI
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs.get("messages") or call_args[1].get("messages")
        prompt_text = " ".join(m["content"] for m in messages)
        assert "10.9999/testdoi" in prompt_text

        # Check that web_search_options was passed
        extra_body = call_args.kwargs.get("extra_body") or call_args[1].get("extra_body")
        assert extra_body == {"web_search_options": {"search_context_size": "high"}}

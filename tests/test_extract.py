"""Tests for value extraction with mocked LLM."""

import json
import logging
from unittest.mock import MagicMock, patch

import pytest

from distribird.agent.extract import (
    _PAGE_NOTE_RESERVE_CHARS,
    _cap_chunks,
    _char_budget,
    _chunk_text,
    _dedup_values,
    _llm_json_call,
    _passes_bounds_check,
    _value_extraction_overhead,
    extract_all_values,
    extract_values_batch,
    extract_values_from_paper,
    extract_values_web_assisted,
    reset_chunk_accumulator,
)
from distribird.config import Settings
from distribird.models import ConstraintSpec, ExtractedValue, LiteratureEvidence, ParameterInput


def _json_response(content: str) -> MagicMock:
    """A mock OpenAI chat-completion response whose message content is ``content``."""
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content=content))]
    return resp


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


@patch("distribird.agent.extract.get_client")
def test_extract_values(mock_get_client, parameter, paper, settings):
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

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


@patch("distribird.agent.extract.get_client")
def test_extract_filters_out_of_bounds(mock_get_client, parameter, paper, settings):
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content='[{"reported_value": 5.2}, {"reported_value": 15.0}]'))
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
        mock_response.choices = [MagicMock(message=MagicMock(content="[1, 2, 3]"))]
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
        mock_response.choices = [MagicMock(message=MagicMock(content="```json\n[1, 2]\n```"))]
        mock_client.chat.completions.create.return_value = mock_response

        result = _llm_json_call(mock_client, "model", [{"role": "user", "content": "test"}])
        assert result == [1, 2]


class TestBatchExtraction:
    """Tests for batched extraction."""

    @patch("distribird.agent.extract.get_client")
    def test_batch_call(self, mock_get_client, parameter, settings):
        papers = [
            LiteratureEvidence(title=f"Paper {i}", abstract=f"LAI was {4 + i}.") for i in range(3)
        ]

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content=json.dumps(
                        {
                            "0": [{"reported_value": 4.0, "context": "test"}],
                            "1": [{"reported_value": 5.0, "context": "test"}],
                            "2": [],
                        }
                    )
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

    @patch("distribird.agent.extract.get_client")
    def test_batch_fallback_to_per_paper(self, mock_get_client, parameter, settings):
        """When batch call returns non-dict, falls back to per-paper."""
        papers = [LiteratureEvidence(title="Paper 0", abstract="LAI was 4.0.")]

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # First call (batch) returns invalid format, second call (per-paper fallback) works
        batch_response = MagicMock()
        batch_response.choices = [MagicMock(message=MagicMock(content="not a dict"))]
        per_paper_response = MagicMock()
        per_paper_response.choices = [
            MagicMock(
                message=MagicMock(content='[{"reported_value": 4.0, "context": "fallback"}]')
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

    @patch("distribird.agent.extract.get_client")
    def test_extract_all_uses_batches(self, mock_get_client, parameter, settings):
        """extract_all_values should chunk into batches."""
        papers = [
            LiteratureEvidence(title=f"Paper {i}", abstract=f"LAI was {4 + i * 0.1}.")
            for i in range(7)
        ]

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Two batch calls: batch of 5, then batch of 2
        batch1_response = MagicMock()
        batch1_response.choices = [
            MagicMock(
                message=MagicMock(
                    content=json.dumps(
                        {str(i): [{"reported_value": 4 + i * 0.1}] for i in range(5)}
                    )
                )
            )
        ]
        batch2_response = MagicMock()
        batch2_response.choices = [
            MagicMock(
                message=MagicMock(
                    content=json.dumps(
                        {
                            "0": [{"reported_value": 4.5}],
                            "1": [{"reported_value": 4.6}],
                        }
                    )
                )
            )
        ]
        mock_client.chat.completions.create.side_effect = [batch1_response, batch2_response]

        result = extract_all_values(papers, parameter, settings)
        assert len(result) == 7  # All papers should have values


class TestWebAssistedExtraction:
    """Tests for web-assisted extraction."""

    @patch("distribird.agent.extract.get_client")
    def test_web_assisted_extraction(self, mock_get_client, parameter, settings):
        """Web-assisted extraction returns values for papers found online."""
        papers = [
            LiteratureEvidence(
                title="Paper A",
                doi="10.1234/a",
                abstract="LAI study.",
                relevance_score=0.9,
            ),
            LiteratureEvidence(
                title="Paper B",
                doi="10.1234/b",
                abstract="Another study.",
                relevance_score=0.7,
            ),
        ]

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content=json.dumps(
                        {
                            "0": [
                                {
                                    "reported_value": 6.1,
                                    "context": "Table 2",
                                    "source_url": "https://doi.org/10.1234/a",
                                    "extraction_confidence": "high",
                                }
                            ],
                            "1": [],
                        }
                    )
                )
            )
        ]
        mock_client.chat.completions.create.return_value = mock_response

        result = extract_values_web_assisted(papers, parameter, settings)
        assert len(result) == 1
        assert result[0].title == "Paper A"
        assert result[0].extracted_values[0].reported_value == 6.1

    @patch("distribird.agent.extract.get_client")
    def test_web_assisted_no_values(self, mock_get_client, parameter, settings):
        """Returns empty when LLM finds no values online."""
        papers = [
            LiteratureEvidence(title="Paper X", doi="10.1234/x", abstract="Irrelevant."),
        ]

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=json.dumps({"0": []})))]
        mock_client.chat.completions.create.return_value = mock_response

        result = extract_values_web_assisted(papers, parameter, settings)
        assert result == []

    @patch("distribird.agent.extract.get_client")
    def test_web_assisted_max_papers_cap(self, mock_get_client, parameter, settings):
        """Only top max_papers are sent to LLM."""
        papers = [
            LiteratureEvidence(
                title=f"Paper {i}",
                abstract=f"Abstract {i}",
                relevance_score=i / 15.0,
            )
            for i in range(15)
        ]

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_response = MagicMock()
        # Return empty for all — we just care about how many batches are called
        mock_response.choices = [MagicMock(message=MagicMock(content=json.dumps({})))]
        mock_client.chat.completions.create.return_value = mock_response

        extract_values_web_assisted(papers, parameter, settings, max_papers=10)

        # 10 papers in batches of 3 → ceil(10/3) = 4 calls
        assert mock_client.chat.completions.create.call_count == 4

    @patch("distribird.agent.extract.get_client")
    def test_web_assisted_includes_doi(self, mock_get_client, parameter, settings):
        """Verify the prompt sent to LLM includes paper DOIs."""
        papers = [
            LiteratureEvidence(
                title="DOI Paper",
                doi="10.9999/testdoi",
                abstract="Test.",
            ),
        ]

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=json.dumps({"0": []})))]
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


class TestCharBudget:
    """The per-call character budget derivation."""

    def test_budget_matches_formula(self):
        s = Settings(
            llm_max_context_tokens=10_000,
            llm_reserved_answer_tokens=2_000,
            llm_chars_per_token=4.0,
        )
        # (10000 - 2000) * 4.0 - 100 = 31_900
        assert _char_budget(s, 100) == 31_900

    def test_budget_can_go_non_positive_on_tiny_window(self):
        s = Settings(
            llm_max_context_tokens=1_000,
            llm_reserved_answer_tokens=900,
            llm_chars_per_token=1.0,
        )
        # (1000 - 900) * 1.0 - 5000 = -4900
        assert _char_budget(s, 5_000) < 0


class TestChunkText:
    """Deterministic page splitting."""

    def test_single_chunk_when_fits(self):
        text = "abc\n\ndef"
        assert _chunk_text(text, 1_000, 100) == [text]

    def test_splits_with_overlap_and_is_deterministic(self):
        # Unique content per paragraph so substring positions are unambiguous.
        text = "\n\n".join(f"P{i:03d} " + "abcdefghij " * 4 for i in range(40))
        chunks = _chunk_text(text, 600, 80)
        assert len(chunks) > 1
        assert all(len(c) <= 600 for c in chunks)
        # Deterministic: identical inputs -> identical output.
        assert _chunk_text(text, 600, 80) == chunks
        # Full coverage with no gaps: each chunk starts at or before the prior
        # chunk's end (i.e. chunks are contiguous or overlap), and together they
        # span the whole text.
        assert text.startswith(chunks[0])
        prev_end = 0
        for c in chunks:
            start = text.index(c)
            assert start <= prev_end
            prev_end = start + len(c)
        assert prev_end == len(text)

    def test_no_zero_progress_with_degenerate_overlap(self):
        text = "y" * 1_000  # no newline boundaries
        chunks = _chunk_text(text, 100, 500)  # overlap > chunk -> clamped < chunk
        assert len(chunks) > 1
        assert all(len(c) <= 100 for c in chunks)


class TestDedupValues:
    def test_overlap_duplicates_collapse_first_wins(self):
        a = ExtractedValue(reported_value=5.0, context="cond A")
        a_dup = ExtractedValue(reported_value=5.0, context="cond A", sample_size=10)
        b = ExtractedValue(reported_value=5.0, context="cond B")
        out = _dedup_values([a, a_dup, b])
        assert out == [a, b]  # a_dup dropped (same key), b kept (different context)


class TestCapChunks:
    def test_caps_and_prioritises_methods(self, caplog):
        chunks = [f"section {i}\nbody text here" for i in range(5)]
        chunks[3] = "Methods\nwe measured the parameter"  # priority section
        acc = reset_chunk_accumulator()
        with caplog.at_level(logging.WARNING):
            kept = _cap_chunks(chunks, 2, LiteratureEvidence(title="Big paper"))
        assert len(kept) == 2
        assert any("Methods" in c for c in kept)  # priority chunk retained
        assert any("> cap 2" in r.message for r in caplog.records)
        assert len(acc["cap_warnings"]) == 1  # surfaced to the node accumulator

    def test_no_cap_when_under_limit(self):
        chunks = ["a", "b"]
        assert _cap_chunks(chunks, 8, LiteratureEvidence(title="t")) == chunks


@patch("distribird.agent.extract.get_client")
def test_full_text_fits_single_call(mock_get_client, parameter, settings):
    """Full text under the (large default) budget is extracted in one call."""
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    mock_client.chat.completions.create.return_value = _json_response('[{"reported_value": 5.0}]')

    paper = LiteratureEvidence(title="T", full_text="The LAI was about 5. " * 50)
    reset_chunk_accumulator()
    values = extract_values_from_paper(paper, parameter, settings)

    assert mock_client.chat.completions.create.call_count == 1
    assert len(values) == 1


@patch("distribird.agent.extract.get_client")
def test_oversized_full_text_is_paged(mock_get_client, parameter):
    """Full text exceeding the budget is read across multiple chunks and merged."""
    s = Settings(
        llm_base_url="x",
        llm_api_key="y",
        llm_max_context_tokens=5_000,
        llm_reserved_answer_tokens=256,
        llm_chars_per_token=4.0,
        extraction_chunk_overlap_chars=100,
        extraction_max_chunks=8,
    )
    overhead = _value_extraction_overhead(LiteratureEvidence(title="T"), parameter, None)
    budget = _char_budget(s, overhead)
    assert budget > 0

    para = ("measurement " * 30).strip()
    text = "\n\n".join([para] * ((budget * 3) // len(para) + 5))
    # Chunk size reserves room for the per-page reading note.
    chunk_chars = max(budget - _PAGE_NOTE_RESERVE_CHARS, 1)
    expected_chunks = _chunk_text(text, chunk_chars, s.extraction_chunk_overlap_chars)
    n = len(expected_chunks)
    assert 1 < n <= s.extraction_max_chunks

    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    # Distinct value per chunk so dedup does not collapse them.
    mock_client.chat.completions.create.side_effect = [
        _json_response(f'[{{"reported_value": {i + 1}.0}}]') for i in range(n)
    ]

    paper = LiteratureEvidence(title="T", full_text=text)
    chunk_stats = reset_chunk_accumulator()
    values = extract_values_from_paper(paper, parameter, s)

    assert mock_client.chat.completions.create.call_count == n
    assert len(values) == n  # all chunks contributed, no spurious dedup
    assert chunk_stats["n_chunked_papers"] == 1
    assert chunk_stats["total_chunks"] == n


def _paging_settings() -> Settings:
    """Settings whose budget is positive but small enough to force page-turning."""
    return Settings(
        llm_base_url="x",
        llm_api_key="y",
        llm_max_context_tokens=5_000,
        llm_reserved_answer_tokens=256,
        llm_chars_per_token=4.0,
        extraction_chunk_overlap_chars=100,
        extraction_max_chunks=8,
    )


def _oversized_text(s: Settings, parameter) -> tuple[str, int]:
    """Build full text that pages into >1 chunk; return (text, expected_chunk_count)."""
    overhead = _value_extraction_overhead(LiteratureEvidence(title="T"), parameter, None)
    budget = _char_budget(s, overhead)
    para = ("measurement " * 30).strip()
    text = "\n\n".join([para] * ((budget * 3) // len(para) + 5))
    chunk_chars = max(budget - _PAGE_NOTE_RESERVE_CHARS, 1)
    n = len(_chunk_text(text, chunk_chars, s.extraction_chunk_overlap_chars))
    return text, n


@patch("distribird.agent.extract.get_client")
def test_paged_chunks_carry_part_markers(mock_get_client, parameter):
    """Each page's prompt tells the model which part of the paper it is reading."""
    s = _paging_settings()
    text, n = _oversized_text(s, parameter)
    assert n > 1

    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    mock_client.chat.completions.create.return_value = _json_response("[]")

    reset_chunk_accumulator()
    extract_values_from_paper(LiteratureEvidence(title="T", full_text=text), parameter, s)

    calls = mock_client.chat.completions.create.call_args_list
    assert len(calls) == n
    for part, call in enumerate(calls, start=1):
        user_msg = call.kwargs["messages"][-1]["content"]
        assert f"PART {part} of {n}" in user_msg


@patch("distribird.agent.extract.get_client")
def test_single_call_has_no_part_marker(mock_get_client, parameter, settings):
    """A paper read in one call gets the plain prompt, with no page note."""
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    mock_client.chat.completions.create.return_value = _json_response("[]")

    paper = LiteratureEvidence(title="T", full_text="The LAI was about 5. " * 20)
    extract_values_from_paper(paper, parameter, settings)

    user_msg = mock_client.chat.completions.create.call_args.kwargs["messages"][-1]["content"]
    assert "Reading note" not in user_msg


@patch("distribird.agent.extract.get_client")
def test_tiny_window_falls_back_to_single_call(mock_get_client, parameter, caplog):
    """A context window so small the budget is non-positive still makes one call."""
    s = Settings(
        llm_base_url="x",
        llm_api_key="y",
        llm_max_context_tokens=1_000,
        llm_reserved_answer_tokens=900,
        llm_chars_per_token=1.0,
    )
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    mock_client.chat.completions.create.return_value = _json_response('[{"reported_value": 5.0}]')

    paper = LiteratureEvidence(title="T", full_text="Z" * 5_000)
    reset_chunk_accumulator()
    with caplog.at_level(logging.WARNING):
        values = extract_values_from_paper(paper, parameter, s)

    assert mock_client.chat.completions.create.call_count == 1
    assert len(values) == 1
    assert any("budget <= 0" in r.message for r in caplog.records)

"""Shared test fixtures."""

import pytest

from distribird.agent import extract as _extract
from distribird.config import Settings
from distribird.models import ConstraintSpec, ParameterInput


@pytest.fixture(autouse=True)
def _isolate_llm_accumulators():
    """Reset the extract module's token/chunk ContextVars before each test.

    Both are process-wide ContextVars that accumulate LLM-call and page-turn
    counts. Without this, a test that installs an accumulator (or records calls
    through mocked clients) leaks its counts into later tests — e.g. inflating
    ``get_call_count()`` so a budget gate behaves differently depending on test
    order. Resetting to None mirrors the pristine pre-pipeline state.
    """
    _extract._token_accumulator.set(None)
    _extract._chunk_accumulator.set(None)
    yield


@pytest.fixture
def sample_parameter() -> ParameterInput:
    return ParameterInput(
        name="max_lai",
        description="Maximum leaf area index of maize",
        unit="m2/m2",
        domain_context="maize crop modeling",
        constraints=ConstraintSpec(lower_bound=0.0, upper_bound=12.0),
    )


@pytest.fixture
def test_settings() -> Settings:
    return Settings(
        llm_base_url="http://localhost:4000",
        llm_api_key="test-key",
        llm_model="test-model",
        semantic_scholar_api_key="test-s2-key",
        enable_openalex=False,
    )

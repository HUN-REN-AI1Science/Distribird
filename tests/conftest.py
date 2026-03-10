"""Shared test fixtures."""

import pytest

from distribird.config import Settings
from distribird.models import ConstraintSpec, ParameterInput


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

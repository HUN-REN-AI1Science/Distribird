"""Tests for the central LLM client factory (timeout + retry wiring)."""

from distribird.agent.llm_client import get_client, get_deep_research_client
from distribird.config import Settings


def test_get_client_forwards_timeout_and_retries():
    settings = Settings(
        llm_base_url="https://example.test/v1",
        llm_api_key="sk-test",
        llm_timeout=42.0,
        llm_max_retries=5,
    )
    client = get_client(settings)
    assert client.timeout == 42.0
    assert client.max_retries == 5


def test_get_deep_research_client_forwards_timeout_and_retries():
    settings = Settings(
        deep_research_base_url="https://research.test/v1",
        deep_research_api_key="sk-test",
        llm_timeout=99.0,
        llm_max_retries=0,
    )
    client = get_deep_research_client(settings)
    assert client.timeout == 99.0
    assert client.max_retries == 0


def test_defaults_are_explicit_not_sdk_defaults():
    # Guards against silently reverting to SDK defaults (timeout 600s, retries 2).
    settings = Settings(llm_base_url="https://example.test/v1", llm_api_key="sk-test")
    client = get_client(settings)
    assert client.timeout == 120.0
    assert client.max_retries == 3

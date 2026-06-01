"""Central factory for OpenAI-compatible LLM clients.

Every module obtains its client here instead of constructing ``OpenAI(...)``
inline. Keeping endpoint and credential wiring in one place removes the
repetition that previously lived at ~14 call sites (DRY) and gives a single seam
for future cross-cutting concerns (timeouts, retries, connection reuse) and for
injecting a stub client in tests.
"""

from __future__ import annotations

from openai import OpenAI

from distribird.config import Settings


def get_client(settings: Settings) -> OpenAI:
    """Return a client pointed at the configured default LLM endpoint."""
    return OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)


def get_deep_research_client(settings: Settings) -> OpenAI:
    """Return a client pointed at the dedicated deep-research endpoint."""
    return OpenAI(
        base_url=settings.deep_research_base_url,
        api_key=settings.deep_research_api_key,
    )

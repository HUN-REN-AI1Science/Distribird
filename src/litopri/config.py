"""Configuration via pydantic-settings."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """LitoPri configuration loaded from environment variables."""

    model_config = {"env_prefix": "LITOPRI_", "env_file": ".env", "env_file_encoding": "utf-8"}

    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = "gemini-3-pro"

    semantic_scholar_api_key: str = ""
    semantic_scholar_base_url: str = "https://api.semanticscholar.org/graph/v1"

    max_papers_per_query: int = 20
    max_search_queries: int = 5
    extraction_timeout: float = 30.0

    max_parallel_parameters: int = 3
    enable_context_enrichment: bool = True
    enable_semantic_scholar: bool = True
    enable_llm_deep_research: bool = False
    llm_web_search: bool = True

    deep_research_base_url: str = ""
    deep_research_api_key: str = ""
    deep_research_model: str = "o4-mini-deep-research"

    enable_openalex: bool = True
    openalex_email: str = ""

    enable_deliberation: bool = True
    enable_web_search_agent: bool = False
    deliberation_model: str | None = None

    enable_relevance_judgment: bool = True
    enable_snowballing: bool = True
    snowball_max_seeds: int = 3
    snowball_limit_per_seed: int = 10

    # Feedback loop settings (LangGraph pipeline)
    search_refinement_max: int = 2
    cross_enrichment_max: int = 1
    extraction_refinement_max: int = 1
    total_llm_calls_max: int = 30
    min_values_for_synthesis: int = 2

    auth_username: str = "demo"
    auth_password: str = "changeme"


def get_settings() -> Settings:
    return Settings()

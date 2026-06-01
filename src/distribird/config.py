"""Configuration via pydantic-settings."""

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Distribird configuration loaded from environment variables."""

    model_config = {"env_prefix": "DISTRIBIRD_", "env_file": ".env", "env_file_encoding": "utf-8"}

    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = "gemini-3-pro"

    # LLM sampling temperature per task class. Defaults preserve current
    # behaviour; lower all three toward 0.0 for more deterministic runs.
    # (Reasoning models ignore temperature and always run at their default.)
    # Bounded to the OpenAI-compatible [0, 2] sampling range.
    llm_temperature_precise: float = Field(
        default=0.0, ge=0.0, le=2.0
    )  # query generation, extraction, relevance, validity
    llm_temperature_creative: float = Field(default=0.3, ge=0.0, le=2.0)  # enrichment, refinement
    llm_temperature_deliberation: float = Field(
        default=0.1, ge=0.0, le=2.0
    )  # multi-agent deliberation moderator

    semantic_scholar_api_key: str = ""
    semantic_scholar_base_url: str = "https://api.semanticscholar.org/graph/v1"

    max_papers_per_query: int = Field(default=20, ge=1, le=100)
    max_search_queries: int = Field(default=5, ge=1, le=20)
    extraction_timeout: float = Field(default=30.0, gt=0.0)

    # Full-text fetch fallbacks, in order of cost, tried when a paper's primary
    # PDF URL fails (e.g. a publisher 403):
    #   (B) enable_oa_mirror_fallback — pure HTTP: resolve alternate open-access
    #       copies via Unpaywall (PMC, repositories) and follow citation_pdf_url
    #       links on landing pages. Cheap and Streamlit-safe; on by default so
    #       blocked URLs are still resolved as best as possible without a browser.
    #   (C) enable_stealth_fetch — last resort: a headless stealth browser
    #       (Camoufox) that solves JS bot walls like MDPI's Akamai bm-verify.
    #       Heavy (~1.3 GB browser, high RAM), so opt-in; requires the `stealth`
    #       extra + `python -m camoufox fetch`. Lazily imported, so off is a no-op
    #       and unsuitable hosts (e.g. Streamlit Cloud) simply skip it.
    enable_oa_mirror_fallback: bool = True
    enable_stealth_fetch: bool = False
    stealth_fetch_timeout: float = Field(default=90.0, gt=0.0)

    # When a PDF URL serves HTML instead of a PDF, extract the article text from
    # the HTML (e.g. PubMed Central full-text pages, repository and DOAJ pages).
    # A quality gate rejects bot-challenge interstitials and thin/abstract-only
    # pages so they do not pollute extraction. On by default.
    enable_html_fulltext: bool = True
    html_fulltext_min_chars: int = Field(default=2000, ge=0)

    max_parallel_parameters: int = Field(default=3, ge=1)
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
    snowball_max_seeds: int = Field(default=3, ge=0)
    snowball_limit_per_seed: int = Field(default=10, ge=0)

    # Feedback loop settings (LangGraph pipeline)
    search_refinement_max: int = Field(default=2, ge=0)
    cross_enrichment_max: int = Field(default=1, ge=0)
    extraction_refinement_max: int = Field(default=1, ge=0)
    total_llm_calls_max: int = Field(default=30, ge=1)
    min_values_for_synthesis: int = Field(default=2, ge=1)

    # BullshitBench: parameter validity detection
    enable_validity_check: bool = True
    enable_validity_probe: bool = True

    # Rate limiting
    s2_rate_limit: float = Field(
        default=0.9, gt=0.0
    )  # req/sec without API key (margin under 1/sec)
    s2_rate_limit_with_key: float = Field(
        default=9.0, gt=0.0
    )  # req/sec with API key (under 10/sec)
    openalex_rate_limit: float = Field(default=9.0, gt=0.0)  # req/sec (polite pool)
    rate_limit_max_retries: int = Field(default=3, ge=0)  # max 429 retries
    rate_limit_base_backoff: float = Field(default=2.0, gt=0.0)  # seconds, doubles each retry

    # Debug tracing: when on, run_parameter_graph captures a full structured trace
    # (LLM prompts/responses, search requests, PDF outcomes, extraction, AIC candidates)
    # to trace_output_dir and attaches it to PipelineResult.debug_trace. Off by default
    # and behaviour-neutral.
    debug_trace: bool = False
    trace_output_dir: str = "logs/traces"

    auth_username: str = "demo"
    auth_password: str = "changeme"


def get_settings() -> Settings:
    return Settings()

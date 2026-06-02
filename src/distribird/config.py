"""Configuration via pydantic-settings."""

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Distribird configuration loaded from environment variables."""

    model_config = {"env_prefix": "DISTRIBIRD_", "env_file": ".env", "env_file_encoding": "utf-8"}

    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = "gemini-3-pro"
    # Per-request timeout (seconds) and automatic retry count for LLM API calls,
    # forwarded to the OpenAI SDK client. The SDK retries connection errors,
    # timeouts, and 408/409/429/5xx responses with exponential backoff. Setting
    # these explicitly (rather than relying on SDK defaults) makes a slow/hung
    # endpoint fail fast and survives SDK default changes. max_retries=0 disables.
    llm_timeout: float = Field(default=120.0, gt=0.0)
    llm_max_retries: int = Field(default=3, ge=0)
    # Optional integer seed forwarded to the OpenAI-compatible API (`seed`
    # request field) for reproducible sampling. None = omit the field (default,
    # behaviour-neutral); pinning it aids determinism alongside the temperatures.
    llm_seed: int | None = None

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

    # Page-turning / context-budget controls for full-text extraction.
    # When a paper's text fits one LLM call it is extracted in a single call
    # (the historical behaviour). When it does not, the text is split into
    # sequential overlapping chunks ("pages"), each extracted separately, and
    # the values are merged + deduplicated so the WHOLE paper contributes to
    # the prior rather than only the first ~30k characters.
    #
    # The per-call character budget is derived as:
    #   (llm_max_context_tokens - llm_reserved_answer_tokens)
    #       * llm_chars_per_token  -  prompt_overhead_chars
    # Default 255k matches the configured cloud model and keeps typical papers
    # single-call; lower it (e.g. DISTRIBIRD_LLM_MAX_CONTEXT_TOKENS=8000) for a
    # small local host and page-turning activates automatically.
    llm_max_context_tokens: int = Field(default=255_000, ge=1_000)
    # Headroom reserved for the model's JSON answer (and any reasoning tokens).
    llm_reserved_answer_tokens: int = Field(default=4_000, ge=256)
    # Conservative chars-per-token estimate (no tokenizer in repo). English
    # prose is ~4 chars/token; 3.5 under-fills so numeric/table-dense scientific
    # text — which tokenizes denser — does not overflow the window.
    llm_chars_per_token: float = Field(default=3.5, gt=0.0)
    # Safety cap on chunks per paper so a pathological huge PDF on a small
    # window cannot explode into hundreds of LLM calls. NOT a silent drop: when
    # exceeded we warn and keep Methods/Results chunks first (see _cap_chunks).
    extraction_max_chunks: int = Field(default=8, ge=1)
    # Overlap between consecutive chunks so a value straddling a boundary is not
    # lost; the dedup pass removes the duplicates the overlap creates.
    extraction_chunk_overlap_chars: int = Field(default=1_500, ge=0)
    # Generous storage cap applied at fetch time. Decouples stored full text
    # from the per-call budget so the whole paper reaches extraction; only
    # guards against absurd PDFs eating memory. Replaces the old 30k cut.
    fulltext_storage_max_chars: int = Field(default=400_000, ge=10_000)

    semantic_scholar_api_key: str = ""
    semantic_scholar_base_url: str = "https://api.semanticscholar.org/graph/v1"

    max_papers_per_query: int = Field(default=20, ge=1, le=100)
    max_search_queries: int = Field(default=5, ge=1, le=20)
    # Cap on the *aggregated, deduplicated* corpus kept after running all
    # queries (bounds downstream full-text fetch + extraction cost). Distinct
    # from max_papers_per_query, which limits each individual query.
    max_papers_total: int = Field(default=50, ge=1)
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

    # API server bind address. Default preserves container-friendly behaviour
    # (0.0.0.0); `distribird.api.main` warns when binding a non-loopback host
    # while auth is still at the insecure defaults.
    api_host: str = "0.0.0.0"
    api_port: int = Field(default=8000, ge=1, le=65535)


def get_settings() -> Settings:
    return Settings()

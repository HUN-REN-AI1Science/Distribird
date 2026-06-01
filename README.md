<p align="center">
  <img src="https://raw.githubusercontent.com/HUN-REN-AI1Science/Distribird/main/assets/logo.svg" alt="Distribird" width="360">
</p>

<p align="center">
  <strong>Automated Bayesian prior construction from scientific literature</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/distribird/"><img src="https://img.shields.io/pypi/v/distribird?logo=pypi&logoColor=white" alt="PyPI"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/tests-223%20passed-brightgreen" alt="Tests: 223 passed">
  <img src="https://img.shields.io/badge/lint-ruff-261230?logo=ruff&logoColor=D7FF64" alt="Linting: ruff">
  <a href="https://langgraph.dev/"><img src="https://img.shields.io/badge/orchestration-LangGraph-1C3C3C?logo=langchain&logoColor=white" alt="LangGraph"></a>
  <a href="https://fastapi.tiangolo.com"><img src="https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white" alt="FastAPI"></a>
  <a href="https://distribird.streamlit.app"><img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Open in Streamlit"></a>
</p>

---

**Distribird** turns a parameter name and description into a fully cited, publication-ready prior distribution.
It searches Semantic Scholar, OpenAlex, and LLM deep-research agents in parallel, extracts numerical values from papers, and fits the best-matching `scipy.stats` distribution via AIC selection.

> *"I need a prior for maximum leaf area index of maize."*
>
> &#x279C; `truncated_normal(mu=5.2, sigma=1.5, a=0, b=12)` &mdash; fitted from 6 peer-reviewed sources with full citations.

<p align="center">
  <a href="https://distribird.streamlit.app"><img src="https://img.shields.io/badge/Try_it_now-distribird.streamlit.app-2d6a4f?style=for-the-badge" alt="Try it now"></a>
</p>

## Why Distribird?

Bayesian calibration requires informative priors, but building them from literature is tedious.
Researchers default to flat priors, losing valuable domain knowledge.
Distribird closes that gap: **describe your parameter, get a defensible prior in seconds.**

## Architecture

```
                        ┌──────────────────────────────┐
                        │        LangGraph DAG         │
                        └──────────────────────────────┘

START ─► Enrich ─► QueryGen ─► Search ─► RelevanceJudge ──┬► CrossEnrich ───┐
                                 ▲                        │                 │
                                 │                        └► FetchFulltext ◄┘
                          RefineSearch                            │
                                 ▲                            Extract
                                 │                                │
                                 │   RefineExtraction ◄─── QualityGate
                                 │         │                │     │
                                 │         └────────────────┘     │
                                 │                                ▼
                                 └────────────  Synthesize ─► ValidityCheck ─► END
```

**Multi-agent search** &mdash; Semantic Scholar, OpenAlex, and LLM deep-research agents run concurrently; a moderator LLM selects the best papers via deliberation.

**Relevance scoring** &mdash; An LLM-based relevance judge scores each paper before extraction. When multiple high-relevance papers are found, the pipeline routes through cross-enrichment (citation snowballing + follow-up queries) to discover additional sources.

**Feedback loops** &mdash; A quality gate inspects extraction results and can trigger search refinement (new queries) or extraction refinement (web-assisted re-extraction) before falling through to synthesis.

**Full-text retrieval:** FetchFulltext downloads each paper's PDF and extracts its text. When a download is blocked (for example a publisher returns 403), it falls back to open-access mirrors via Unpaywall, and optionally to a headless stealth browser for sites behind JavaScript bot walls. See "Full-text PDF fallback" under Install.

**Validity defense** &mdash; Every request is classified as `VALID`, `SUSPICIOUS`, `LIKELY_INVALID`, or `UNKNOWN`. When the enrichment LLM does not recognise the parameter, the pipeline short-circuits past search, extraction, and synthesis straight to the validity node, saving roughly 80&ndash;95% of wall-clock time and LLM tokens on out-of-scope requests. Ambiguous (`SUSPICIOUS`) cases trigger a single second-opinion LLM probe.

**Budget-bounded** &mdash; `IterationBudget` caps every loop to guarantee termination.

**Live progress** &mdash; The pipeline streams node-by-node updates to the UI, showing which step is running, paper/value counts, and per-parameter progress bars.

## Quickstart

### Install

```bash
pip install distribird
```

<details>
<summary><strong>Development install</strong></summary>

```bash
git clone https://github.com/HUN-REN-AI1Science/Distribird.git
cd distribird
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                     # 223 tests, all passing
```
</details>

<details>
<summary><strong>Optional: full-text PDF fallback (stealth browser)</strong></summary>

Some publishers (MDPI is one) serve PDFs behind a JavaScript bot wall that returns 403 to plain HTTP clients. Distribird can get past these with a headless stealth browser (Camoufox). It is off by default and kept as a separate extra because it downloads a large browser binary.

```bash
pip install "distribird[stealth]"
python -m camoufox fetch          # one-time browser download, about 1.3 GB
```

Then turn it on (see Configure):

```bash
DISTRIBIRD_ENABLE_STEALTH_FETCH=true
```

This needs a host that can run a real browser, so it does not work on Streamlit Community Cloud, where it is skipped automatically. The lighter Unpaywall mirror fallback still runs everywhere.
</details>

### Configure

Distribird reads configuration from environment variables (prefix `DISTRIBIRD_`) or a `.env` file in the project root.

```bash
# .env (or export these in your shell)
DISTRIBIRD_LLM_BASE_URL="http://localhost:4000"   # any OpenAI-compatible endpoint
DISTRIBIRD_LLM_API_KEY="your-key"
DISTRIBIRD_LLM_MODEL="gpt-4o"
DISTRIBIRD_SEMANTIC_SCHOLAR_API_KEY=""            # optional, increases rate limits
DISTRIBIRD_OPENALEX_EMAIL=""                       # your email; enables the OA mirror fallback
DISTRIBIRD_ENABLE_OA_MIRROR_FALLBACK="true"        # default; Unpaywall mirrors over plain HTTP
DISTRIBIRD_ENABLE_STEALTH_FETCH="false"            # opt-in stealth browser (see Install)
DISTRIBIRD_ENABLE_HTML_FULLTEXT="true"             # default; extract text from HTML full-text pages
```

**Full-text PDF fallback.** When a paper's PDF URL is blocked, Distribird tries the direct URL first, then open-access mirrors through Unpaywall, then the stealth browser if it is enabled. The Unpaywall step needs `DISTRIBIRD_OPENALEX_EMAIL` set to a real address and is skipped without one. The stealth step needs the `stealth` extra and a host that can run a browser; it is skipped on Streamlit Community Cloud. With both off, only the direct URL is used.

When a URL serves HTML instead of a PDF, Distribird extracts the article text from the HTML (for example PubMed Central full-text pages and many repository pages). A quality check rejects bot-challenge pages and thin abstract-only pages so they do not pollute extraction. This is on by default; set `DISTRIBIRD_ENABLE_HTML_FULLTEXT=false` to turn it off, or raise `DISTRIBIRD_HTML_FULLTEXT_MIN_CHARS` to be stricter about what counts as an article.

The stealth browser follows DOI and handle redirects to the real publisher before clearing the bot challenge, so publishers reached through a `doi.org` link (MDPI and similar) are recovered. Some publishers run enterprise bot protection that a headless browser cannot pass: ScienceDirect and Elsevier (PerimeterX), Wiley and Hindawi (Cloudflare managed challenge), and some institutional repositories that block direct file access. PDFs behind these are expected misses; the paper still contributes its title and abstract.

**Sidebar behaviour in the Streamlit UI:**

- Settings provided in `.env` are used automatically — no manual input needed.
- Settings **not** provided in `.env` appear as required fields in the sidebar; the user must fill them in before generation can start.
- An **"Override configured settings"** toggle lets users temporarily replace `.env` values without editing the file.
- Literature source toggles (Semantic Scholar, OpenAlex, LLM Web Search, LLM Deep Research) are always visible and control which connection fields are required.

### Use

**Python**

```python
import asyncio
from distribird.agent.pipeline import run_parameter
from distribird.models import ParameterInput, ConstraintSpec

result = asyncio.run(run_parameter(
    ParameterInput(
        name="max_lai",
        description="Maximum leaf area index of maize",
        unit="m2/m2",
        domain_context="Biome-BGCMuSo maize crop modeling",
        constraints=ConstraintSpec(lower_bound=0, upper_bound=12),
    )
))

print(result.prior.display_name())   # truncated_normal(mu=5.2, sigma=1.5, a=0, b=12)
print(result.prior.n_sources)        # 6
print(result.prior.confidence.value) # high
```

**REST API**

```bash
distribird-api                          # starts on :8000

curl -u demo:distribird2026 -X POST http://localhost:8000/api/v1/parameter \
  -H "Content-Type: application/json" \
  -d '{"name":"max_lai","description":"Maximum leaf area index of maize","unit":"m2/m2"}'
```

**Streamlit UI**

Try the hosted version at **[distribird.streamlit.app](https://distribird.streamlit.app)**, or run locally:

```bash
streamlit run src/distribird/ui/app.py
```

## Prior Fitting Strategy

| Evidence | Method | Confidence |
|---|---|---|
| 5+ values | AIC across Normal, Truncated Normal, Gamma, Log-Normal, Beta | **High** |
| 2 &ndash; 4 values | Moment matching with widened &sigma; | Medium |
| 1 value | Wide Normal centered on value | Low |
| 0 values | Jeffreys / wide uninformative prior | None |

All fitted distributions respect user-specified physical constraints (bounds).

## Export Formats

```python
from distribird.export.json_export import export_json
from distribird.export.r_export import export_r
from distribird.export.python_export import export_python
```

| Format | Output |
|---|---|
| **JSON** | Parameter name, family, params, citations, confidence |
| **R** | Executable R script with distribution calls |
| **Python** | `scipy.stats` code ready for MCMC samplers |

## Demo

A complete worked example using five Biome-BGCMuSo maize parameters:

```bash
python examples/maize_bgcmuso/demo.py
```

## Testing

```bash
pytest                 # 223 tests
ruff check src/ tests/ # lint
mypy src/distribird/      # type checking (strict)
```

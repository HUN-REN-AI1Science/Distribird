<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://img.shields.io/badge/LitoPri-Literature--informed%20Priors-8B5CF6?style=for-the-badge&labelColor=1a1a2e">
    <img alt="LitoPri" src="https://img.shields.io/badge/LitoPri-Literature--informed%20Priors-6D28D9?style=for-the-badge&labelColor=f5f5f5">
  </picture>
</p>

<p align="center">
  <strong>Automated Bayesian prior construction from scientific literature</strong>
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/tests-130%20passed-brightgreen" alt="Tests: 130 passed">
  <img src="https://img.shields.io/badge/lint-ruff-261230?logo=ruff&logoColor=D7FF64" alt="Linting: ruff">
  <a href="https://langgraph.dev/"><img src="https://img.shields.io/badge/orchestration-LangGraph-1C3C3C?logo=langchain&logoColor=white" alt="LangGraph"></a>
  <a href="https://fastapi.tiangolo.com"><img src="https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white" alt="FastAPI"></a>
</p>

---

**LitoPri** turns a parameter name and description into a fully cited, publication-ready prior distribution.
It searches Semantic Scholar, OpenAlex, and LLM deep-research agents in parallel, extracts numerical values from papers, and fits the best-matching `scipy.stats` distribution via AIC selection.

> *"I need a prior for maximum leaf area index of maize."*
>
> &rarr; `truncated_normal(mu=5.2, sigma=1.5, a=0, b=12)` &mdash; fitted from 6 peer-reviewed sources with full citations.

## Why LitoPri?

Bayesian calibration requires informative priors, but building them from literature is tedious.
Researchers default to flat priors, losing valuable domain knowledge.
LitoPri closes that gap: **describe your parameter, get a defensible prior in seconds.**

## Architecture

```
                           ┌─────────────────────────────┐
                           │        LangGraph DAG        │
                           └─────────────────────────────┘

 START ─► Enrich ─► QueryGen ─► Search ───┬──► CrossEnrich ──┐
                                  ▲       │                  │
                                  │       └─► FetchFulltext ◄┘
                          RefineSearch         │
                                  ▲        Extract
                                  │            │
                                  │      QualityGate ──► RefineExtraction
                                  │        │       │            │
                                  └────────┘       ▼            │
                                              Synthesize ◄──────┘
                                                   │
                                                  END
```

**Multi-agent search** &mdash; Semantic Scholar, OpenAlex, and LLM deep-research agents run concurrently; a moderator LLM selects the best papers via deliberation.

**Feedback loops** &mdash; A quality gate inspects extraction results and can trigger search refinement (new queries), cross-enrichment (follow-up from key papers), or extraction refinement (web-assisted re-extraction) before falling through to synthesis.

**Budget-bounded** &mdash; `IterationBudget` caps every loop to guarantee termination.

## Quickstart

### Install

```bash
pip install litopri
```

<details>
<summary><strong>Development install</strong></summary>

```bash
git clone https://github.com/HUN-REN-AI1Science/LitoPri.git
cd litopri
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                     # 130 tests, all passing
```
</details>

### Configure

LitoPri reads configuration from environment variables (prefix `LITOPRI_`) or a `.env` file in the project root.

```bash
# .env (or export these in your shell)
LITOPRI_LLM_BASE_URL="http://localhost:4000"   # any OpenAI-compatible endpoint
LITOPRI_LLM_API_KEY="your-key"
LITOPRI_LLM_MODEL="gpt-4o"
LITOPRI_SEMANTIC_SCHOLAR_API_KEY=""            # optional, increases rate limits
```

**Sidebar behaviour in the Streamlit UI:**

- Settings provided in `.env` are used automatically — no manual input needed.
- Settings **not** provided in `.env` appear as required fields in the sidebar; the user must fill them in before generation can start.
- An **"Override configured settings"** toggle lets users temporarily replace `.env` values without editing the file.
- Literature source toggles (Semantic Scholar, OpenAlex, LLM Web Search, LLM Deep Research) are always visible and control which connection fields are required.

### Use

**Python**

```python
import asyncio
from litopri.agent.pipeline import run_parameter
from litopri.models import ParameterInput, ConstraintSpec

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
litopri-api                          # starts on :8000

curl -u demo:litopri2026 -X POST http://localhost:8000/api/v1/parameter \
  -H "Content-Type: application/json" \
  -d '{"name":"max_lai","description":"Maximum leaf area index of maize","unit":"m2/m2"}'
```

**Streamlit UI**

```bash
streamlit run src/litopri/ui/app.py
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
from litopri.export.json_export import export_json
from litopri.export.r_export import export_r
from litopri.export.python_export import export_python
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
pytest                 # 130 tests
ruff check src/ tests/ # lint
mypy src/litopri/      # type checking (strict)
```

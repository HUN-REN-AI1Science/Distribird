"""Streamlit application for LitoPri."""

from __future__ import annotations

import asyncio

import streamlit as st

from litopri.agent.pipeline import run_parameter
from litopri.config import Settings, get_settings
from litopri.export.json_export import export_single_json
from litopri.export.python_export import export_single_python
from litopri.export.r_export import export_single_r
from litopri.models import ConstraintSpec, ParameterInput, PipelineResult


def inject_custom_css():
    """Inject professional styling."""
    st.markdown(
        """
        <style>
        /* Parameter row cards */
        div[data-testid="stHorizontalBlock"].param-row {
            background: var(--background-secondary, #f8f9fa);
            border-radius: 8px;
            padding: 0.5rem;
            margin-bottom: 0.25rem;
        }
        /* Remove button styling */
        button[kind="secondary"].remove-btn {
            color: #dc3545;
            border-color: #dc3545;
        }
        /* Result confidence borders */
        .result-high {
            border-left: 4px solid #28a745;
            padding-left: 1rem;
        }
        .result-medium {
            border-left: 4px solid #ffc107;
            padding-left: 1rem;
        }
        .result-low {
            border-left: 4px solid #dc3545;
            padding-left: 1rem;
        }
        /* Tighter spacing for param rows */
        .param-header .stCaption {
            margin-bottom: 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _mask_key(key: str) -> str:
    """Show first 4 and last 4 chars of a key, mask the rest."""
    if len(key) <= 10:
        return key
    return key[:4] + "*" * (len(key) - 8) + key[-4:]


def get_settings_from_sidebar() -> Settings:
    """Render sidebar settings, using config.py defaults with optional overrides."""
    defaults = get_settings()

    st.sidebar.title("Configuration")

    st.sidebar.info(f"Model: **{defaults.llm_model}**")

    # Literature source toggles
    st.sidebar.subheader("Literature Sources")
    use_s2 = st.sidebar.toggle(
        "Semantic Scholar",
        value=defaults.enable_semantic_scholar,
        help="Search Semantic Scholar for papers with abstracts + open access full text",
        key="use_s2",
    )
    use_openalex = st.sidebar.toggle(
        "OpenAlex",
        value=defaults.enable_openalex,
        help="Search OpenAlex for open-access papers (no API key required)",
        key="use_openalex",
    )
    web_search = st.sidebar.toggle(
        "LLM Web Search",
        value=defaults.llm_web_search,
        help="Enable Gemini Google Search grounding for context enrichment",
        key="web_search",
    )
    use_deep = st.sidebar.toggle(
        "LLM Deep Research",
        value=defaults.enable_llm_deep_research,
        help="Use o4-mini-deep-research model with built-in web search",
        key="use_deep",
    )
    if not use_s2 and not use_deep and not use_openalex:
        st.sidebar.error("Enable at least one literature source.")

    st.sidebar.markdown("---")

    # Override toggle
    override = st.sidebar.toggle("Override defaults", value=False)

    if override:
        st.sidebar.markdown("---")
        st.sidebar.subheader("LLM")
        base_url = st.sidebar.text_input(
            "Base URL", value="", placeholder="Enter LLM base URL", key="llm_url"
        )
        api_key = st.sidebar.text_input(
            "API Key", value="", type="password", placeholder="Enter API key", key="llm_key"
        )
        model = st.sidebar.text_input(
            "Model", value="", placeholder="Enter model name", key="llm_model"
        )

        st.sidebar.markdown("---")
        st.sidebar.subheader("Semantic Scholar")
        s2_key = st.sidebar.text_input(
            "API Key", value="", placeholder="Enter S2 API key", key="s2_key"
        )

        st.sidebar.markdown("---")
        st.sidebar.subheader("Deep Research Model")
        dr_base_url = st.sidebar.text_input(
            "Base URL", value="", placeholder="Enter deep research base URL", key="dr_url"
        )
        dr_api_key = st.sidebar.text_input(
            "API Key", value="", type="password",
            placeholder="Enter deep research API key", key="dr_key",
        )
        dr_model = st.sidebar.text_input(
            "Model", value="", placeholder="Enter deep research model name",
            key="dr_model",
        )

        st.sidebar.markdown("---")
        st.sidebar.subheader("Search")
        max_queries = st.sidebar.slider(
            "Max search queries", 1, 10, defaults.max_search_queries, key="max_q"
        )
        max_papers = st.sidebar.slider(
            "Max papers per query", 5, 50, defaults.max_papers_per_query, key="max_p"
        )

        return Settings(
            llm_base_url=base_url or defaults.llm_base_url,
            llm_api_key=api_key or defaults.llm_api_key,
            llm_model=model or defaults.llm_model,
            semantic_scholar_api_key=s2_key or defaults.semantic_scholar_api_key,
            max_search_queries=max_queries,
            max_papers_per_query=max_papers,
            enable_semantic_scholar=use_s2,
            enable_openalex=use_openalex,
            enable_llm_deep_research=use_deep,
            llm_web_search=web_search,
            deep_research_base_url=dr_base_url or defaults.deep_research_base_url,
            deep_research_api_key=dr_api_key or defaults.deep_research_api_key,
            deep_research_model=dr_model or defaults.deep_research_model,
        )

    return Settings(
        **{
            **defaults.model_dump(),
            "enable_semantic_scholar": use_s2,
            "enable_openalex": use_openalex,
            "enable_llm_deep_research": use_deep,
            "llm_web_search": web_search,
        },
    )


def check_login() -> bool:
    """Show login form and return True if authenticated."""
    if st.session_state.get("authenticated"):
        return True

    defaults = get_settings()
    st.markdown("### Login")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

    if submitted:
        import hmac

        if hmac.compare_digest(username, defaults.auth_username) and hmac.compare_digest(
            password, defaults.auth_password
        ):
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Invalid username or password.")
    return False


def _collect_references(result: PipelineResult) -> list:
    """Build the full reference list matching deliberation paper indices.

    The deliberation moderator numbers papers [0]...[N] and may reference
    them in warnings. We use 1-based numbering in the UI, so [0] → [1].
    Show all consensus papers (not just those with extracted values) so
    that bracket references in warnings resolve correctly.
    """
    if result.deliberation and result.deliberation.consensus_papers:
        return result.deliberation.consensus_papers
    # Fallback: only papers with extracted values (no deliberation)
    prior = result.prior
    return prior.evidence if prior.evidence else []


def render_result(result: PipelineResult):
    """Render a pipeline result."""
    prior = result.prior

    # Confidence-colored container
    confidence_class = f"result-{prior.confidence.value}"
    st.markdown(f'<div class="{confidence_class}">', unsafe_allow_html=True)

    # Summary
    col1, col2, col3 = st.columns(3)
    col1.metric("Distribution", prior.display_name())
    col2.metric("Confidence", prior.confidence.value.title())
    col3.metric("Sources", prior.n_sources)

    if result.warnings:
        for w in result.warnings:
            st.warning(w)

    # Enrichment info
    if result.enrichment and result.enrichment.common_terminology:
        with st.expander("Parameter Context Enrichment"):
            terms = ", ".join(result.enrichment.common_terminology)
            st.write(f"**Scientific terminology:** {terms}")
            if result.enrichment.enriched_description:
                st.write(f"**Enriched description:** {result.enrichment.enriched_description}")
            if result.enrichment.typical_range:
                st.write(f"**Typical range:** {result.enrichment.typical_range}")
            if result.enrichment.search_hints:
                st.write(f"**Search hints:** {', '.join(result.enrichment.search_hints)}")

    # Details
    with st.expander("Details"):
        st.write(f"**Reason:** {prior.reason}")
        st.write(f"**Search queries:** {', '.join(result.search_queries)}")
        st.write(f"**Papers found:** {result.papers_found}")
        st.write(f"**Values extracted:** {result.values_extracted}")

    # References — show all consensus papers so bracket indices from
    # deliberation warnings (e.g. [5], [12]) match the displayed list.
    all_refs = _collect_references(result)
    if all_refs:
        st.subheader("References")
        for i, e in enumerate(all_refs, 1):
            authors = ", ".join(e.authors[:3])
            if len(e.authors) > 3:
                authors += " et al."
            year = e.year or "n.d."
            doi_link = f"https://doi.org/{e.doi}" if e.doi else None

            title_md = f"**[{i}]** {authors} ({year}). *{e.title}*."
            if doi_link:
                title_md += f" [DOI]({doi_link})"
            st.markdown(title_md)

            values_strs = []
            for v in e.extracted_values:
                if v.reported_value is not None:
                    s = f"{v.reported_value}"
                    if v.uncertainty is not None:
                        s += f" ± {v.uncertainty}"
                    if v.context:
                        s += f" ({v.context})"
                    values_strs.append(s)
                elif v.reported_range is not None:
                    s = f"{v.reported_range[0]}–{v.reported_range[1]}"
                    if v.context:
                        s += f" ({v.context})"
                    values_strs.append(s)
            if values_strs:
                st.caption(f"Extracted values: {'; '.join(values_strs)}")

    # Distribution plot
    with st.expander("Distribution Plot"):
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            from scipy import stats

            fig, ax = plt.subplots(figsize=(8, 4))
            p = prior.params

            if prior.family.value == "truncated_normal":
                a_std = (p["a"] - p["mu"]) / p["sigma"]
                b_std = (p["b"] - p["mu"]) / p["sigma"]
                dist = stats.truncnorm(a_std, b_std, loc=p["mu"], scale=p["sigma"])
            elif prior.family.value == "normal":
                dist = stats.norm(loc=p["mu"], scale=p["sigma"])
            elif prior.family.value == "gamma":
                dist = stats.gamma(a=p["alpha"], scale=p["scale"])
            elif prior.family.value == "lognormal":
                dist = stats.lognorm(s=p["sigma"], scale=np.exp(p["mu"]))
            elif prior.family.value == "uniform":
                dist = stats.uniform(loc=p["lower"], scale=p["upper"] - p["lower"])
            elif prior.family.value == "beta":
                dist = stats.beta(a=p["alpha"], b=p["beta"],
                                  loc=p["lower"], scale=p["upper"] - p["lower"])
            else:
                dist = None

            if dist is not None:
                x = np.linspace(dist.ppf(0.001), dist.ppf(0.999), 500)
                ax.plot(x, dist.pdf(x), "b-", lw=2)
                ax.fill_between(x, dist.pdf(x), alpha=0.2)
                ax.set_xlabel(result.parameter.unit or "Value")
                ax.set_ylabel("Density")
                ax.set_title(f"Prior: {prior.display_name()}")
                st.pyplot(fig)
            else:
                st.info("Plot not available for this distribution type.")
            plt.close(fig)
        except Exception as e:
            st.error(f"Could not generate plot: {e}")

    # Export
    st.subheader("Export")
    tab_json, tab_r, tab_py = st.tabs(["JSON", "R", "Python"])
    with tab_json:
        st.code(export_single_json(result), language="json")
    with tab_r:
        st.code(export_single_r(result), language="r")
    with tab_py:
        st.code(export_single_python(result), language="python")

    st.markdown("</div>", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state keys for dynamic parameter rows."""
    if "params" not in st.session_state:
        st.session_state.params = [
            {"id": 0, "name": "", "description": "", "unit": "",
             "lower_bound": None, "upper_bound": None}
        ]
    if "next_id" not in st.session_state:
        st.session_state.next_id = 1
    if "results" not in st.session_state:
        st.session_state.results = {}
    if "is_running" not in st.session_state:
        st.session_state.is_running = False


def add_parameter_row():
    """Callback to add a new parameter row."""
    row_id = st.session_state.next_id
    st.session_state.next_id += 1
    st.session_state.params.append(
        {"id": row_id, "name": "", "description": "", "unit": "",
         "lower_bound": None, "upper_bound": None}
    )


def remove_parameter_row(row_id: int):
    """Callback to remove a parameter row by ID."""
    st.session_state.params = [p for p in st.session_state.params if p["id"] != row_id]
    st.session_state.results.pop(row_id, None)


def sync_param_values():
    """Read current widget values back into the params list."""
    for param in st.session_state.params:
        rid = param["id"]
        param["name"] = st.session_state.get(f"name_{rid}", "")
        param["description"] = st.session_state.get(f"desc_{rid}", "")
        param["unit"] = st.session_state.get(f"unit_{rid}", "")
        param["lower_bound"] = st.session_state.get(f"lower_{rid}")
        param["upper_bound"] = st.session_state.get(f"upper_{rid}")


def render_parameter_rows():
    """Render the dynamic parameter table with column headers."""
    is_running = st.session_state.is_running

    # Column headers
    hcols = st.columns([3, 4, 1.5, 1.5, 1.5, 0.5])
    hcols[0].caption("Name")
    hcols[1].caption("Description")
    hcols[2].caption("Unit")
    hcols[3].caption("Lower bound (opt.)")
    hcols[4].caption("Upper bound (opt.)")
    hcols[5].caption("")

    for param in st.session_state.params:
        rid = param["id"]
        cols = st.columns([3, 4, 1.5, 1.5, 1.5, 0.5])

        cols[0].text_input(
            "Name", value=param["name"], key=f"name_{rid}",
            placeholder="e.g., leaf_area_index",
            label_visibility="collapsed", disabled=is_running,
        )
        cols[1].text_input(
            "Description", value=param["description"], key=f"desc_{rid}",
            placeholder="e.g., Maximum leaf area index of maize",
            label_visibility="collapsed", disabled=is_running,
        )
        cols[2].text_input(
            "Unit", value=param["unit"], key=f"unit_{rid}",
            placeholder="e.g., m2/m2",
            label_visibility="collapsed", disabled=is_running,
        )
        cols[3].number_input(
            "Lower", value=param["lower_bound"], key=f"lower_{rid}",
            format="%f", label_visibility="collapsed", disabled=is_running,
        )
        cols[4].number_input(
            "Upper", value=param["upper_bound"], key=f"upper_{rid}",
            format="%f", label_visibility="collapsed", disabled=is_running,
        )

        # Only show remove button if more than one row
        if len(st.session_state.params) > 1:
            cols[5].button(
                "X", key=f"rm_{rid}",
                on_click=remove_parameter_row, args=(rid,),
                disabled=is_running,
            )


def render_results_section():
    """Show all completed results."""
    st.header("Results")
    for param in st.session_state.params:
        rid = param["id"]
        if rid in st.session_state.results:
            result = st.session_state.results[rid]
            st.subheader(result.parameter.name)
            render_result(result)
            st.divider()


def process_all_parameters(settings: Settings):
    """Process all valid parameters sequentially with progress."""
    sync_param_values()
    domain_context = st.session_state.get("domain_context", "")

    valid_params = [
        p for p in st.session_state.params
        if p["name"].strip() and p["description"].strip()
    ]

    if not valid_params:
        st.warning("Please fill in at least one parameter with name and description.")
        return

    st.session_state.is_running = True
    st.session_state.results = {}
    n = len(valid_params)
    progress_bar = st.progress(0, text=f"Processing 0/{n} parameters...")

    for i, param in enumerate(valid_params):
        progress_bar.progress(i / n, text=f"Parameter {i + 1}/{n}: {param['name']}")

        parameter = ParameterInput(
            name=param["name"],
            description=param["description"],
            unit=param["unit"],
            domain_context=domain_context,
            constraints=ConstraintSpec(
                lower_bound=param["lower_bound"],
                upper_bound=param["upper_bound"],
            ),
        )

        with st.status(f"Processing: {param['name']}", expanded=True) as status:
            st.write("Searching literature and fitting distribution...")
            try:
                result = asyncio.run(run_parameter(parameter, settings))
            except Exception as e:
                st.error(f"Error processing {param['name']}: {e}")
                status.update(label=f"Error: {param['name']}", state="error")
                continue
            status.update(label=f"Complete: {param['name']}", state="complete")

        st.session_state.results[param["id"]] = result

    progress_bar.progress(1.0, text="All parameters processed!")
    st.session_state.is_running = False


def main():
    st.set_page_config(page_title="LitoPri", page_icon="📊", layout="wide")
    inject_custom_css()

    if not check_login():
        return

    st.title("LitoPri: Literature-informed Priors")
    st.markdown(
        "Automatically search scientific literature and synthesize "
        "informative prior distributions for Bayesian model calibration."
    )

    settings = get_settings_from_sidebar()
    init_session_state()

    # Domain context (shared across all parameters)
    st.header("Domain Context")
    st.text_input(
        "Domain context shared by all parameters below",
        placeholder="e.g., maize crop modeling with Biome-BGCMuSo",
        key="domain_context",
        disabled=st.session_state.is_running,
    )

    # Dynamic parameter rows
    st.header("Parameters")
    render_parameter_rows()

    st.button(
        "+ Add Parameter",
        on_click=add_parameter_row,
        disabled=st.session_state.is_running,
    )

    st.divider()

    # Check if any valid parameters exist
    sync_param_values()
    has_valid = any(
        p["name"].strip() and p["description"].strip()
        for p in st.session_state.params
    )

    if st.button(
        "Generate Priors",
        type="primary",
        disabled=(not has_valid or st.session_state.is_running),
    ):
        process_all_parameters(settings)

    # Show persisted results (from previous runs, visible after rerun)
    if st.session_state.results and not st.session_state.is_running:
        render_results_section()


if __name__ == "__main__":
    main()

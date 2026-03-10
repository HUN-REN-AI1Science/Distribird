"""Streamlit application for Distribird."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import streamlit as st

from distribird.agent.graph import NODE_META
from distribird.agent.pipeline import run_parameter
from distribird.config import Settings, get_settings
from distribird.export.json_export import export_single_json
from distribird.export.python_export import export_single_python
from distribird.export.r_export import export_single_r
from distribird.models import ConstraintSpec, LiteratureEvidence, ParameterInput, PipelineResult
from distribird.ui.persistence import (
    clear_persisted_state,
    hydrate_session_state,
    save_session_state,
)

_ASSETS_DIR_FROM_FILE = Path(__file__).resolve().parent.parent.parent.parent / "assets"
_ASSETS_DIR_FROM_CWD = Path.cwd() / "assets"
_ASSETS_DIR = _ASSETS_DIR_FROM_FILE if _ASSETS_DIR_FROM_FILE.is_dir() else _ASSETS_DIR_FROM_CWD


def inject_custom_css() -> None:
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


def _check_missing_secrets(settings: Settings) -> list[str]:
    """Return list of missing required secret descriptions."""
    missing = []
    if not settings.llm_base_url:
        missing.append("LLM Base URL")
    if not settings.llm_api_key:
        missing.append("LLM API Key")
    if not settings.llm_model:
        missing.append("LLM Model")
    if settings.enable_semantic_scholar and not settings.semantic_scholar_api_key:
        missing.append("Semantic Scholar API Key")
    if settings.enable_llm_deep_research:
        if not settings.deep_research_base_url:
            missing.append("Deep Research Base URL")
        if not settings.deep_research_api_key:
            missing.append("Deep Research API Key")
        if not settings.deep_research_model:
            missing.append("Deep Research Model")
    return missing


def _secret_input(label: str, key: str, default: str, *, password: bool = False) -> str:
    """Render a sidebar input that shows pre-filled status or required warning."""
    has_default = bool(default)
    if password and has_default:
        placeholder = f"Configured ({_mask_key(default)})"
    elif has_default:
        placeholder = default
    else:
        placeholder = f"Required — enter {label}"
    help_text = None if has_default else "Not configured — you must provide this value"
    value = st.sidebar.text_input(
        label,
        value=default if not password else "",
        placeholder=placeholder,
        type="password" if password else "default",
        key=key,
        help=help_text,
    )
    return value if value else default


def _render_required_fields(
    defaults: Settings,
    use_s2: bool,
    use_deep: bool,
) -> dict[str, str]:
    """Render inputs for connection fields NOT provided in .env.

    These are always visible because the user must fill them in.
    Returns a dict of field_name -> value for any fields rendered.
    """
    overrides: dict[str, str] = {}
    missing_llm: list[tuple[str, str, str, bool]] = []  # (label, key, field, password)

    if not defaults.llm_base_url:
        missing_llm.append(("LLM Base URL", "llm_url_req", "llm_base_url", False))
    if not defaults.llm_api_key:
        missing_llm.append(("LLM API Key", "llm_key_req", "llm_api_key", True))
    if not defaults.llm_model:
        missing_llm.append(("LLM Model", "llm_model_req", "llm_model", False))

    if use_s2 and not defaults.semantic_scholar_api_key:
        missing_llm.append(
            (
                "Semantic Scholar API Key",
                "s2_key_req",
                "semantic_scholar_api_key",
                True,
            )
        )

    if use_deep:
        if not defaults.deep_research_base_url:
            missing_llm.append(
                (
                    "Deep Research Base URL",
                    "dr_url_req",
                    "deep_research_base_url",
                    False,
                )
            )
        if not defaults.deep_research_api_key:
            missing_llm.append(
                (
                    "Deep Research API Key",
                    "dr_key_req",
                    "deep_research_api_key",
                    True,
                )
            )
        if not defaults.deep_research_model:
            missing_llm.append(
                (
                    "Deep Research Model",
                    "dr_model_req",
                    "deep_research_model",
                    False,
                )
            )

    if missing_llm:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Required Settings")
        for label, key, field, password in missing_llm:
            value = st.sidebar.text_input(
                label,
                value="",
                key=key,
                type="password" if password else "default",
                placeholder=f"Enter {label}",
            )
            overrides[field] = value

    return overrides


def _render_override_fields(
    defaults: Settings,
    use_s2: bool,
    use_deep: bool,
    use_openalex: bool,
) -> dict[str, str]:
    """Render override inputs for connection fields that ARE provided in .env.

    Only shown when the user enables the override toggle.
    Returns a dict of field_name -> value for any overridden fields.
    """
    # Collect which fields have .env values
    has_llm = bool(defaults.llm_base_url and defaults.llm_api_key and defaults.llm_model)
    has_s2 = bool(defaults.semantic_scholar_api_key)
    has_deep = bool(
        defaults.deep_research_base_url
        and defaults.deep_research_api_key
        and defaults.deep_research_model
    )
    has_oa = bool(defaults.openalex_email)

    # Only show the toggle if there's something to override
    has_any_configured = (
        has_llm or (use_s2 and has_s2) or (use_deep and has_deep) or (use_openalex and has_oa)
    )
    if not has_any_configured:
        return {}

    st.sidebar.markdown("---")
    if "override_toggle" not in st.session_state:
        st.session_state["override_toggle"] = False
    override = st.sidebar.toggle("Override configured settings", key="override_toggle")
    if not override:
        return {}

    overrides: dict[str, str] = {}

    if has_llm:
        st.sidebar.subheader("LLM Settings")
        st.sidebar.caption("OpenAI-compatible Chat Completions endpoint")
        overrides["llm_base_url"] = _secret_input("Base URL", "llm_url_ov", defaults.llm_base_url)
        overrides["llm_api_key"] = _secret_input(
            "API Key",
            "llm_key_ov",
            defaults.llm_api_key,
            password=True,
        )
        overrides["llm_model"] = _secret_input("Model", "llm_model_ov", defaults.llm_model)

    if use_s2 and has_s2:
        st.sidebar.subheader("Semantic Scholar")
        overrides["semantic_scholar_api_key"] = _secret_input(
            "API Key", "s2_key_ov", defaults.semantic_scholar_api_key, password=True
        )

    if use_deep and has_deep:
        st.sidebar.subheader("Deep Research Model")
        st.sidebar.caption("OpenAI-compatible Chat Completions endpoint")
        overrides["deep_research_base_url"] = _secret_input(
            "Base URL",
            "dr_url_ov",
            defaults.deep_research_base_url,
        )
        overrides["deep_research_api_key"] = _secret_input(
            "API Key",
            "dr_key_ov",
            defaults.deep_research_api_key,
            password=True,
        )
        overrides["deep_research_model"] = _secret_input(
            "Model",
            "dr_model_ov",
            defaults.deep_research_model,
        )

    if use_openalex and has_oa:
        st.sidebar.subheader("OpenAlex")
        overrides["openalex_email"] = st.sidebar.text_input(
            "Email",
            value=defaults.openalex_email,
            key="oa_email_ov",
        )

    return overrides


def get_settings_from_sidebar() -> Settings:
    """Render sidebar settings.

    Fields provided in .env are used automatically and hidden behind an
    "Override" toggle.  Fields NOT provided in .env are shown as required
    inputs the user must fill in before running.
    """
    defaults = get_settings()

    # Set defaults in session_state (only if not already set, e.g. by
    # hydration from localStorage).  This avoids the Streamlit warning
    # that fires when both ``value=`` and a session_state key are present.
    _widget_defaults = {
        "use_s2": defaults.enable_semantic_scholar,
        "use_openalex": defaults.enable_openalex,
        "web_search": defaults.llm_web_search,
        "use_deep": defaults.enable_llm_deep_research,
        "override_toggle": False,
        "max_q": defaults.max_search_queries,
        "max_p": defaults.max_papers_per_query,
    }
    for key, val in _widget_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    st.sidebar.title("Configuration")

    # --- Literature Sources (always shown) ---
    st.sidebar.subheader("Literature Sources")
    use_s2 = st.sidebar.toggle(
        "Semantic Scholar",
        help="Search Semantic Scholar for papers with abstracts + open access full text",
        key="use_s2",
    )
    use_openalex = st.sidebar.toggle(
        "OpenAlex",
        help="Search OpenAlex for open-access papers (no API key required)",
        key="use_openalex",
    )
    web_search = st.sidebar.toggle(
        "LLM Web Search",
        help=(
            "Passes web_search_options in the request body for grounded context "
            "enrichment. Your LLM endpoint must support this (e.g. OpenRouter, "
            "LiteLLM proxy with a web-search-capable model). "
            "Disable if your provider does not support it."
        ),
        key="web_search",
    )
    use_deep = st.sidebar.toggle(
        "LLM Deep Research",
        help=(
            "Use a dedicated deep research model with built-in web search. "
            "Uses the OpenAI Chat Completions API via a separate endpoint."
        ),
        key="use_deep",
    )
    if not use_s2 and not use_deep and not use_openalex:
        st.sidebar.error("Enable at least one literature source.")
    if web_search and not defaults.llm_model:
        st.sidebar.warning(
            "Web Search requires an LLM endpoint that supports "
            "`web_search_options` in the request body. "
            "Disable this if your provider returns errors.",
            icon="⚠️",
        )

    # --- Required fields (missing from .env) ---
    required = _render_required_fields(defaults, use_s2, use_deep)

    # --- Override toggle for .env-provided fields ---
    overrides = _render_override_fields(defaults, use_s2, use_deep, use_openalex)

    # --- Search Settings ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("Search Settings")
    max_queries = st.sidebar.slider("Max search queries", 1, 10, key="max_q")
    max_papers = st.sidebar.slider("Max papers per query", 5, 50, key="max_p")

    # Merge: defaults ← required ← overrides
    merged = {
        **defaults.model_dump(),
        "enable_semantic_scholar": use_s2,
        "enable_openalex": use_openalex,
        "enable_llm_deep_research": use_deep,
        "llm_web_search": web_search,
        "max_search_queries": max_queries,
        "max_papers_per_query": max_papers,
        **required,
        **overrides,
    }

    return Settings(**merged)


def check_login() -> bool:
    """Show login form and return True if authenticated.

    Skip login entirely if credentials are at defaults or empty.
    """
    if st.session_state.get("authenticated"):
        return True

    defaults = get_settings()

    # Skip login if not configured (defaults or empty)
    if defaults.auth_username in ("demo", "") and defaults.auth_password in ("changeme", ""):
        return True

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


def _collect_references(result: PipelineResult) -> list[LiteratureEvidence]:
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


def _render_single_export(safe_name: str, result: PipelineResult) -> None:
    """Per-result export with tabs for each format."""
    st.subheader("Export")
    tab_json, tab_r, tab_py = st.tabs(["JSON", "R", "Python"])
    with tab_json:
        json_str = export_single_json(result)
        st.code(json_str, language="json")
        st.download_button(
            f"Download {safe_name}.json",
            json_str,
            file_name=f"{safe_name}.json",
            mime="application/json",
            key=f"dl_json_{safe_name}",
            use_container_width=True,
        )
    with tab_r:
        r_str = export_single_r(result)
        st.code(r_str, language="r")
        st.download_button(
            f"Download {safe_name}.R",
            r_str,
            file_name=f"{safe_name}.R",
            mime="text/plain",
            key=f"dl_r_{safe_name}",
            use_container_width=True,
        )
    with tab_py:
        py_str = export_single_python(result)
        st.code(py_str, language="python")
        st.download_button(
            f"Download {safe_name}.py",
            py_str,
            file_name=f"{safe_name}.py",
            mime="text/x-python",
            key=f"dl_py_{safe_name}",
            use_container_width=True,
        )


def render_result(result: PipelineResult) -> None:
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

    # References
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

    # Excluded papers (from deliberation) shown in a collapsed, muted section
    excluded = result.deliberation.excluded_papers if result.deliberation else []
    if excluded:
        with st.expander(f"Excluded papers ({len(excluded)} reviewed but not used)"):
            for e in excluded:
                authors = ", ".join(e.authors[:3])
                if len(e.authors) > 3:
                    authors += " et al."
                year = e.year or "n.d."
                doi_link = f"https://doi.org/{e.doi}" if e.doi else None
                line = f"{authors} ({year}). *{e.title}*."
                if doi_link:
                    line += f" [DOI]({doi_link})"
                st.caption(line)

    # Distribution plot
    with st.expander("Distribution Plot"):
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            from scipy import stats  # type: ignore[import-untyped]

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
                dist = stats.beta(
                    a=p["alpha"], b=p["beta"], loc=p["lower"], scale=p["upper"] - p["lower"]
                )
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
    safe_name = result.parameter.name.replace(" ", "_").lower()
    _render_single_export(safe_name, result)

    st.markdown("</div>", unsafe_allow_html=True)


def init_session_state() -> None:
    """Initialize session state keys for dynamic parameter rows."""
    if "params" not in st.session_state:
        st.session_state.params = [
            {
                "id": 0,
                "name": "",
                "description": "",
                "unit": "",
                "lower_bound": None,
                "upper_bound": None,
            }
        ]
    if "next_id" not in st.session_state:
        st.session_state.next_id = 1
    if "results" not in st.session_state:
        st.session_state.results = {}
    if "is_running" not in st.session_state:
        st.session_state.is_running = False


def add_parameter_row() -> None:
    """Callback to add a new parameter row."""
    row_id = st.session_state.next_id
    st.session_state.next_id += 1
    st.session_state.params.append(
        {
            "id": row_id,
            "name": "",
            "description": "",
            "unit": "",
            "lower_bound": None,
            "upper_bound": None,
        }
    )


def remove_parameter_row(row_id: int) -> None:
    """Callback to remove a parameter row by ID."""
    st.session_state.params = [p for p in st.session_state.params if p["id"] != row_id]
    st.session_state.results.pop(row_id, None)


def sync_param_values() -> None:
    """Read current widget values back into the params list."""
    for param in st.session_state.params:
        rid = param["id"]
        param["name"] = st.session_state.get(f"name_{rid}", "")
        param["description"] = st.session_state.get(f"desc_{rid}", "")
        param["unit"] = st.session_state.get(f"unit_{rid}", "")
        param["lower_bound"] = st.session_state.get(f"lower_{rid}")
        param["upper_bound"] = st.session_state.get(f"upper_{rid}")


def render_parameter_rows() -> None:
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
            "Name",
            value=param["name"],
            key=f"name_{rid}",
            placeholder="e.g., leaf_area_index",
            label_visibility="collapsed",
            disabled=is_running,
        )
        cols[1].text_input(
            "Description",
            value=param["description"],
            key=f"desc_{rid}",
            placeholder="e.g., Maximum leaf area index of maize",
            label_visibility="collapsed",
            disabled=is_running,
        )
        cols[2].text_input(
            "Unit",
            value=param["unit"],
            key=f"unit_{rid}",
            placeholder="e.g., m2/m2",
            label_visibility="collapsed",
            disabled=is_running,
        )
        cols[3].number_input(
            "Lower",
            value=param["lower_bound"],
            key=f"lower_{rid}",
            format="%f",
            label_visibility="collapsed",
            disabled=is_running,
        )
        cols[4].number_input(
            "Upper",
            value=param["upper_bound"],
            key=f"upper_{rid}",
            format="%f",
            label_visibility="collapsed",
            disabled=is_running,
        )

        # Only show remove button if more than one row
        if len(st.session_state.params) > 1:
            cols[5].button(
                "X",
                key=f"rm_{rid}",
                on_click=remove_parameter_row,
                args=(rid,),
                disabled=is_running,
            )


def render_results_section() -> None:
    """Show all completed results."""
    st.header("Results")
    for param in st.session_state.params:
        rid = param["id"]
        if rid in st.session_state.results:
            result = st.session_state.results[rid]
            st.subheader(result.parameter.name)
            render_result(result)
            st.divider()


def process_all_parameters(settings: Settings) -> None:
    """Process all valid parameters sequentially with progress."""
    sync_param_values()
    domain_context = st.session_state.get("domain_context", "")

    valid_params = [
        p for p in st.session_state.params if p["name"].strip() and p["description"].strip()
    ]

    if not valid_params:
        st.warning("Please fill in at least one parameter with name and description.")
        return

    st.session_state.is_running = True
    st.session_state.results = {}
    n = len(valid_params)
    overall_bar = st.progress(0, text=f"Processing 0/{n} parameters...")

    for i, param in enumerate(valid_params):
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
            step_label = st.empty()
            step_detail = st.empty()
            step_progress = st.empty()

            completed_nodes: set[str] = set()

            def on_node_complete(
                node_name: str,
                state: dict[str, Any],
                _completed: set[str] = completed_nodes,
                _step_label: Any = step_label,
                _step_detail: Any = step_detail,
                _step_progress: Any = step_progress,
                _overall_bar: Any = overall_bar,
                _i: int = i,
                _n: int = n,
                _param: dict[str, Any] = param,
            ) -> None:
                label, weight = NODE_META.get(node_name, (node_name, 0.0))
                is_retry = node_name in _completed
                display = f"{label} (retry)" if is_retry else label
                _step_label.markdown(f"**{display}...**")

                # Build detail text from accumulated state
                parts = []
                papers: Any = state.get("all_papers", [])
                if papers:
                    parts.append(f"{len(papers)} papers")
                pwv: Any = state.get("papers_with_values", [])
                if pwv:
                    nv = sum(len(p.extracted_values) for p in pwv)
                    parts.append(f"{nv} values")
                queries: Any = state.get("all_queries_tried", [])
                if queries:
                    parts.append(f"{len(queries)} queries")
                _step_detail.text(" | ".join(parts) if parts else "")

                # Update per-parameter progress (never decreases)
                if not is_retry and weight > 0:
                    _completed.add(node_name)
                pct = sum(NODE_META[nd][1] for nd in _completed if nd in NODE_META)
                _step_progress.progress(min(pct, 1.0))

                # Update overall bar
                param_pct = (_i + pct) / _n
                _overall_bar.progress(
                    min(param_pct, 1.0),
                    text=f"Parameter {_i + 1}/{_n}: {_param['name']} — {display}",
                )

            try:
                result = asyncio.run(run_parameter(parameter, settings, on_node_complete))
            except Exception as e:
                st.error(f"Error processing {param['name']}: {e}")
                status.update(label=f"Error: {param['name']}", state="error")
                continue

            step_label.markdown("**Complete**")
            step_progress.progress(1.0)
            status.update(label=f"Complete: {param['name']}", state="complete")

        st.session_state.results[param["id"]] = result

    overall_bar.progress(1.0, text="All parameters processed!")
    st.session_state.is_running = False


@st.dialog("Documentation", width="large")
def _show_docs(docs_path: Path) -> None:
    """Render the standalone docs HTML inside a Streamlit dialog."""
    import streamlit.components.v1 as components

    html_content = docs_path.read_text(encoding="utf-8")
    components.html(html_content, height=700, scrolling=True)


def main() -> None:
    _logo_path = _ASSETS_DIR / "logo.svg"
    _logo_svg = _logo_path.read_text() if _logo_path.exists() else None
    st.set_page_config(page_title="Distribird", page_icon=_logo_svg, layout="wide")
    inject_custom_css()

    if _logo_svg:
        st.logo(_logo_svg, size="large")

    ls = hydrate_session_state()

    if not check_login():
        return

    st.title("Distribird: Literature Informed Prior Generator")
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
        p["name"].strip() and p["description"].strip() for p in st.session_state.params
    )

    missing = _check_missing_secrets(settings)
    if missing:
        st.error(f"Missing required settings: {', '.join(missing)}")

    if st.button(
        "Generate Priors",
        type="primary",
        disabled=(not has_valid or st.session_state.is_running or bool(missing)),
    ):
        process_all_parameters(settings)

    # Show persisted results (from previous runs, visible after rerun)
    if st.session_state.results and not st.session_state.is_running:
        render_results_section()

    # Sidebar: documentation link (auto-open on first visit)
    _docs_path = _ASSETS_DIR / "docs.html"
    if _docs_path.exists():
        if not st.session_state.get("_docs_shown"):
            st.session_state["_docs_shown"] = True
            _show_docs(_docs_path)
        st.sidebar.markdown("---")
        if st.sidebar.button(
            "Documentation", icon=":material/menu_book:", use_container_width=True
        ):
            _show_docs(_docs_path)

    # Sidebar: clear saved settings
    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Your settings and results are saved locally in your browser. "
        "We do not store any of your data on our servers."
    )
    if st.sidebar.button("Clear saved settings"):
        clear_persisted_state(ls)
        st.rerun()

    save_session_state(ls)


if __name__ == "__main__":
    main()

"""LangGraph node functions wrapping existing pipeline logic."""

from __future__ import annotations

import logging
import time

from openai import OpenAI

from litopri.agent.state import (
    IterationBudget,
    MessageKind,
    PipelineState,
    QualityMetrics,
    TraceEvent,
    add_papers,
    post_message,
    update_quality,
)
from litopri.config import Settings
from litopri.models import EnrichedContext

logger = logging.getLogger(__name__)


def _settings_from_state(state: PipelineState) -> Settings:
    return Settings(**state["settings_dict"])


def _trace(node: str, start: float, summary: dict | None = None) -> TraceEvent:
    return TraceEvent(
        node=node,
        timestamp=start,
        duration_s=time.time() - start,
        summary=summary or {},
    )


# ---------------------------------------------------------------------------
# Core pipeline nodes
# ---------------------------------------------------------------------------


async def enrich_node(state: PipelineState) -> dict:
    t0 = time.time()
    settings = _settings_from_state(state)
    parameter = state["parameter"]
    warnings = list(state.get("warnings", []))
    traces = list(state.get("trace_events", []))

    enrichment: EnrichedContext | None = None
    if settings.enable_context_enrichment:
        from litopri.agent.enrich import enrich_parameter_context

        try:
            enrichment = enrich_parameter_context(parameter, settings)
        except Exception as e:
            logger.warning("[node:enrich] failed: %s", e)
            warnings.append(f"Context enrichment failed: {e}")

    traces.append(_trace("enrich", t0, {"has_enrichment": enrichment is not None}))
    return {
        "enrichment": enrichment,
        "warnings": warnings,
        "trace_events": traces,
    }


async def query_gen_node(state: PipelineState) -> dict:
    t0 = time.time()
    settings = _settings_from_state(state)
    parameter = state["parameter"]
    enrichment = state.get("enrichment")
    traces = list(state.get("trace_events", []))

    from litopri.agent.search import generate_search_queries

    queries = generate_search_queries(parameter, settings, enrichment=enrichment)

    all_tried = list(state.get("all_queries_tried", []))
    all_tried.extend(queries)

    traces.append(_trace("query_gen", t0, {"n_queries": len(queries)}))
    return {
        "search_queries": queries,
        "all_queries_tried": all_tried,
        "trace_events": traces,
    }


async def search_node(state: PipelineState) -> dict:
    t0 = time.time()
    settings = _settings_from_state(state)
    parameter = state["parameter"]
    queries = state.get("search_queries", [])
    enrichment = state.get("enrichment")
    blackboard = list(state.get("blackboard", []))
    warnings = list(state.get("warnings", []))
    traces = list(state.get("trace_events", []))

    if settings.enable_deliberation:
        from litopri.agent.deliberation import deliberate, run_source_agents

        agent_findings = await run_source_agents(
            parameter,
            queries,
            settings,
            enrichment,
        )
        deliberation = await deliberate(
            agent_findings,
            parameter,
            settings,
            enrichment,
        )
        papers = deliberation.consensus_papers
        warnings.extend(deliberation.warnings)

        # Post discovery messages for high-relevance papers
        for p in papers:
            if p.relevance_score > 0.6:
                post_message(
                    state,
                    "search_node",
                    MessageKind.DISCOVERY,
                    f"High-relevance paper: {p.title}",
                    references=[p.doi] if p.doi else [],
                )

        new_added = add_papers(state, papers)
        traces.append(
            _trace(
                "search",
                t0,
                {
                    "n_papers": len(papers),
                    "n_new": len(new_added),
                    "deliberation": True,
                },
            )
        )
        return {
            "all_papers": state.get("all_papers", []),
            "seen_dois": state.get("seen_dois", set()),
            "agent_findings": agent_findings,
            "deliberation": deliberation,
            "blackboard": blackboard,
            "warnings": warnings,
            "trace_events": traces,
        }
    else:
        from litopri.agent.search import search_all_queries

        papers = await search_all_queries(queries, settings)
        new_added = add_papers(state, papers)

        traces.append(
            _trace(
                "search",
                t0,
                {
                    "n_papers": len(papers),
                    "n_new": len(new_added),
                    "deliberation": False,
                },
            )
        )
        return {
            "all_papers": state.get("all_papers", []),
            "seen_dois": state.get("seen_dois", set()),
            "blackboard": blackboard,
            "warnings": warnings,
            "trace_events": traces,
        }


async def relevance_judge_node(state: PipelineState) -> dict:
    t0 = time.time()
    settings = _settings_from_state(state)
    parameter = state["parameter"]
    papers = state.get("all_papers", [])
    enrichment = state.get("enrichment")
    budget = state.get("budget", IterationBudget())
    traces = list(state.get("trace_events", []))

    n_judged = 0
    if settings.enable_relevance_judgment and papers and budget.has_budget():
        from litopri.agent.search import judge_paper_relevance

        llm_calls = judge_paper_relevance(papers, parameter, settings, enrichment)
        budget.total_llm_calls_used += llm_calls
        n_judged = sum(1 for p in papers if p.relevance_snippet)

    n_high = sum(1 for p in papers if p.relevance_score > 0.7)
    traces.append(
        _trace(
            "relevance_judge",
            t0,
            {
                "n_judged": n_judged,
                "n_high": n_high,
            },
        )
    )
    return {
        "all_papers": papers,
        "budget": budget,
        "trace_events": traces,
    }


async def fetch_fulltext_node(state: PipelineState) -> dict:
    t0 = time.time()
    settings = _settings_from_state(state)
    papers = state.get("all_papers", [])
    traces = list(state.get("trace_events", []))

    n_fulltext = 0
    if papers:
        from litopri.agent.fulltext import fetch_all_fulltexts

        n_fulltext = await fetch_all_fulltexts(papers, settings)

    traces.append(_trace("fetch_fulltext", t0, {"n_fulltext": n_fulltext}))
    return {
        "all_papers": papers,
        "trace_events": traces,
    }


async def extract_node(state: PipelineState) -> dict:
    t0 = time.time()
    settings = _settings_from_state(state)
    parameter = state["parameter"]
    papers = list(state.get("all_papers", []))
    enrichment = state.get("enrichment")
    warnings = list(state.get("warnings", []))
    traces = list(state.get("trace_events", []))

    from litopri.agent.extract import (
        extract_all_values,
        extract_consensus_values,
        extract_values_web_assisted,
    )

    papers_with_values = extract_all_values(
        papers,
        parameter,
        settings,
        enrichment=enrichment,
    )
    total_values = sum(len(p.extracted_values) for p in papers_with_values)

    # Consensus extraction fallback
    if total_values == 0 and papers:
        consensus_values = extract_consensus_values(
            papers,
            parameter,
            settings,
            enrichment=enrichment,
        )
        if consensus_values:
            best_paper = max(papers, key=lambda p: p.relevance_score)
            best_paper.extracted_values = consensus_values
            papers_with_values = [best_paper]
            total_values = len(consensus_values)
            warnings.append(
                "No per-paper values found; used consensus extraction across abstracts."
            )

    # Web-assisted extraction fallback
    if total_values == 0 and papers and settings.llm_web_search:
        web_papers = extract_values_web_assisted(
            papers,
            parameter,
            settings,
            enrichment=enrichment,
        )
        if web_papers:
            papers_with_values.extend(p for p in web_papers if p not in papers_with_values)
            total_values = sum(len(p.extracted_values) for p in papers_with_values)
            warnings.append("Used web-assisted extraction to look up paper content online.")

    traces.append(
        _trace(
            "extract",
            t0,
            {
                "n_papers": len(papers),
                "n_with_values": len(papers_with_values),
                "total_values": total_values,
            },
        )
    )
    return {
        "papers_with_values": papers_with_values,
        "warnings": warnings,
        "trace_events": traces,
    }


async def quality_gate_node(state: PipelineState) -> dict:
    t0 = time.time()
    traces = list(state.get("trace_events", []))
    quality = update_quality(state)
    traces.append(
        _trace(
            "quality_gate",
            t0,
            {
                "n_values": quality.n_total_values,
                "n_papers": quality.n_papers_found,
                "sufficient": quality.is_sufficient(),
            },
        )
    )
    return {
        "quality": quality,
        "trace_events": traces,
    }


async def synthesize_node(state: PipelineState) -> dict:
    t0 = time.time()
    parameter = state["parameter"]
    papers_with_values = state.get("papers_with_values", [])
    warnings = list(state.get("warnings", []))
    traces = list(state.get("trace_events", []))

    from litopri.agent.synthesize import synthesize_prior

    enrichment = state.get("enrichment")
    prior = synthesize_prior(parameter, papers_with_values, enrichment=enrichment)

    if not prior.is_informative:
        warnings.append("No informative evidence found; using uninformative prior.")
    if prior.n_sources == 1:
        warnings.append("Prior based on a single source; low confidence.")

    traces.append(
        _trace(
            "synthesize",
            t0,
            {
                "family": prior.family.value,
                "is_informative": prior.is_informative,
                "n_sources": prior.n_sources,
            },
        )
    )
    return {
        "prior": prior,
        "warnings": warnings,
        "trace_events": traces,
    }


# ---------------------------------------------------------------------------
# Feedback loop nodes
# ---------------------------------------------------------------------------


async def refine_search_node(state: PipelineState) -> dict:
    t0 = time.time()
    settings = _settings_from_state(state)
    parameter = state["parameter"]
    budget = state.get("budget", IterationBudget())
    all_tried = list(state.get("all_queries_tried", []))
    papers = state.get("all_papers", [])
    blackboard = list(state.get("blackboard", []))
    warnings = list(state.get("warnings", []))
    traces = list(state.get("trace_events", []))

    from litopri.agent.extract import _llm_json_call
    from litopri.agent.prompts import SEARCH_REFINEMENT

    paper_summaries = "\n".join(
        f"- {p.title} ({p.year}): {(p.abstract or '')[:150]}" for p in papers[:10]
    )

    high_rel = sorted(
        [p for p in papers if p.relevance_score > 0.7],
        key=lambda p: p.relevance_score,
        reverse=True,
    )[:5]
    high_rel_text = (
        "\n".join(
            f"- {p.title} ({p.year}): {p.relevance_snippet or (p.abstract or '')[:150]}"
            for p in high_rel
        )
        or "(none found yet)"
    )

    bb_text = (
        "\n".join(f"- [{m.kind.value}] {m.sender}: {m.content}" for m in blackboard[-10:])
        or "(none)"
    )

    prompt = SEARCH_REFINEMENT.format(
        name=parameter.name,
        description=parameter.description,
        unit=parameter.unit,
        domain_context=parameter.domain_context,
        previous_queries="\n".join(f"- {q}" for q in all_tried),
        paper_summaries=paper_summaries,
        high_relevance_papers=high_rel_text,
        blackboard_messages=bb_text,
        n_queries=settings.max_search_queries,
    )

    client = OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)
    try:
        raw = _llm_json_call(
            client,
            settings.llm_model,
            [{"role": "user", "content": prompt}],
            temperature=0.3,
        )
    except Exception as e:
        logger.warning("[node:refine_search] LLM call failed: %s", e)
        raw = {}

    new_queries: list[str] = []
    if isinstance(raw, dict):
        diagnosis = raw.get("diagnosis", "")
        new_queries = [
            q for q in raw.get("new_queries", []) if isinstance(q, str) and q not in all_tried
        ]
        term_updates = raw.get("terminology_updates", [])

        if diagnosis:
            post_message(
                state,
                "refine_search",
                MessageKind.WARNING,
                f"Search refinement diagnosis: {diagnosis}",
            )
        for term in term_updates:
            post_message(
                state,
                "refine_search",
                MessageKind.TERMINOLOGY,
                term,
            )

    if not new_queries:
        new_queries = [f"{parameter.name} {parameter.unit} measured"]

    all_tried.extend(new_queries)
    budget.search_refinement_used += 1
    budget.total_llm_calls_used += 1

    warnings.append(
        f"Search refinement round {budget.search_refinement_used}: "
        f"generated {len(new_queries)} new queries."
    )

    traces.append(
        _trace(
            "refine_search",
            t0,
            {
                "n_new_queries": len(new_queries),
                "round": budget.search_refinement_used,
            },
        )
    )
    return {
        "search_queries": new_queries,
        "all_queries_tried": all_tried,
        "budget": budget,
        "blackboard": blackboard,
        "warnings": warnings,
        "trace_events": traces,
    }


async def cross_enrich_node(state: PipelineState) -> dict:
    t0 = time.time()
    settings = _settings_from_state(state)
    parameter = state["parameter"]
    papers = state.get("all_papers", [])
    budget = state.get("budget", IterationBudget())
    blackboard = list(state.get("blackboard", []))
    traces = list(state.get("trace_events", []))

    # --- Snowballing (pure S2 API, no LLM cost) ---
    if settings.enable_snowballing and budget.can_snowball():
        from litopri.agent.search import snowball_papers

        existing_dois = {p.doi.strip().lower() for p in papers if p.doi}
        seed_papers = sorted(papers, key=lambda p: p.relevance_score, reverse=True)
        snowballed = await snowball_papers(
            seed_papers,
            settings,
            existing_dois,
            max_seeds=settings.snowball_max_seeds,
            limit_per_seed=settings.snowball_limit_per_seed,
        )
        add_papers(state, snowballed)
        budget.snowball_used += 1
        papers = state.get("all_papers", [])

    from litopri.agent.extract import _llm_json_call
    from litopri.agent.prompts import CROSS_ENRICHMENT_QUERIES

    # Identify key papers (high relevance)
    key_papers = [p for p in papers if p.relevance_score > 0.6]
    if not key_papers:
        key_papers = sorted(papers, key=lambda p: p.relevance_score, reverse=True)[:3]

    key_papers_text = "\n".join(
        f"- {p.title} ({p.year}, DOI: {p.doi}): {(p.abstract or '')[:200]}" for p in key_papers
    )

    # Post cross-ref messages
    for p in key_papers:
        if p.doi:
            post_message(
                state,
                "cross_enrich",
                MessageKind.CROSS_REF,
                f"Key paper for follow-up: {p.title}",
                references=[p.doi],
            )

    prompt = CROSS_ENRICHMENT_QUERIES.format(
        name=parameter.name,
        description=parameter.description,
        unit=parameter.unit,
        domain_context=parameter.domain_context,
        key_papers=key_papers_text,
        n_queries=settings.max_search_queries,
    )

    client = OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)
    new_queries: list[str] = []
    try:
        raw = _llm_json_call(
            client,
            settings.llm_model,
            [{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        if isinstance(raw, list):
            new_queries = [str(q) for q in raw[: settings.max_search_queries]]
    except Exception as e:
        logger.warning("[node:cross_enrich] LLM call failed: %s", e)

    # Run targeted searches with new queries
    from litopri.agent.search import search_all_queries

    if new_queries:
        new_papers = await search_all_queries(new_queries, settings)
        add_papers(state, new_papers)

    budget.cross_enrichment_used += 1
    budget.total_llm_calls_used += 1

    traces.append(
        _trace(
            "cross_enrich",
            t0,
            {
                "n_key_papers": len(key_papers),
                "n_new_queries": len(new_queries),
            },
        )
    )
    return {
        "all_papers": state.get("all_papers", []),
        "seen_dois": state.get("seen_dois", set()),
        "search_queries": new_queries,
        "budget": budget,
        "blackboard": blackboard,
        "trace_events": traces,
    }


async def refine_extraction_node(state: PipelineState) -> dict:
    t0 = time.time()
    settings = _settings_from_state(state)
    parameter = state["parameter"]
    papers = state.get("all_papers", [])
    papers_with_values = list(state.get("papers_with_values", []))
    enrichment = state.get("enrichment")
    budget = state.get("budget", IterationBudget())
    warnings = list(state.get("warnings", []))
    traces = list(state.get("trace_events", []))

    from litopri.agent.extract import extract_values_web_assisted

    # Select papers most likely to have values (high relevance, no values yet)
    existing_titles = {p.title for p in papers_with_values}
    candidates = [
        p
        for p in papers
        if p.title not in existing_titles and (p.full_text or p.relevance_score > 0.3)
    ]
    candidates.sort(key=lambda p: p.relevance_score, reverse=True)

    if candidates:
        web_papers = extract_values_web_assisted(
            candidates[:10],
            parameter,
            settings,
            enrichment=enrichment,
        )
        if web_papers:
            papers_with_values.extend(p for p in web_papers if p not in papers_with_values)
            warnings.append("Used refined extraction on high-relevance papers.")

    budget.extraction_refinement_used += 1
    budget.total_llm_calls_used += 1

    traces.append(
        _trace(
            "refine_extraction",
            t0,
            {
                "n_candidates": len(candidates),
                "n_new_values": len(papers_with_values) - len(state.get("papers_with_values", [])),
            },
        )
    )
    return {
        "papers_with_values": papers_with_values,
        "budget": budget,
        "warnings": warnings,
        "trace_events": traces,
    }


# ---------------------------------------------------------------------------
# Router functions for conditional edges
# ---------------------------------------------------------------------------


def route_after_deliberation(state: PipelineState) -> str:
    """Route after deliberation: check if cross-enrichment is needed."""
    budget = state.get("budget", IterationBudget())
    papers = state.get("all_papers", [])

    if budget.can_cross_enrich() and len(papers) >= 2:
        # Check if papers from multiple agents or high relevance
        high_rel = [p for p in papers if p.relevance_score > 0.6]
        if len(high_rel) >= 2:
            return "cross_enrich"

    return "fetch_fulltext"


def route_after_quality_gate(state: PipelineState) -> str:
    """Route after quality gate: synthesize, refine search, or refine extraction."""
    quality = state.get("quality", QualityMetrics())
    budget = state.get("budget", IterationBudget())

    # Loop A: 0 values, papers exist → refine search
    if quality.needs_search_refinement() and budget.can_refine_search():
        return "refine_search"

    # Loop C: low confidence values → refine extraction
    if quality.needs_extraction_refinement() and budget.can_refine_extraction():
        return "refine_extraction"

    return "synthesize"

"""LangGraph pipeline graph definition."""

from __future__ import annotations

import functools
import logging
from typing import Any, Callable, cast

from langgraph.graph import END, START, StateGraph

from distribird.agent import diagnostics
from distribird.agent.nodes import (
    cross_enrich_node,
    enrich_node,
    extract_node,
    fetch_fulltext_node,
    quality_gate_node,
    query_gen_node,
    refine_extraction_node,
    refine_search_node,
    relevance_judge_node,
    route_after_deliberation,
    route_after_enrich,
    route_after_quality_gate,
    search_node,
    synthesize_node,
    validity_check_node,
)
from distribird.agent.state import IterationBudget, PipelineState, QualityMetrics
from distribird.config import Settings, get_settings
from distribird.models import ParameterInput, ParameterValidity, PipelineResult

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, dict[str, Any]], None] | None

NODE_META: dict[str, tuple[str, float]] = {
    # (human_label, progress_weight) — weights sum to ~1.0 for happy path
    "enrich": ("Understanding parameter context", 0.05),
    "query_gen": ("Generating search queries", 0.05),
    "search": ("Searching scientific literature", 0.25),
    "relevance_judge": ("Judging paper relevance", 0.10),
    "cross_enrich": ("Cross-referencing citations", 0.05),
    "fetch_fulltext": ("Fetching full-text papers", 0.10),
    "extract": ("Extracting numerical values", 0.20),
    "quality_gate": ("Evaluating extraction quality", 0.05),
    "synthesize": ("Fitting prior distribution", 0.13),
    "validity_check": ("Validating parameter", 0.02),
    # Loop nodes — zero weight so progress never goes backwards
    "refine_search": ("Refining search strategy", 0.0),
    "refine_extraction": ("Refining value extraction", 0.0),
}


def _traced(name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a node so the debug tracer attributes its events to this node."""

    @functools.wraps(fn)
    async def wrapped(state: PipelineState) -> Any:
        diagnostics.set_node(name)
        return await fn(state)

    return wrapped


def build_pipeline_graph() -> StateGraph:  # type: ignore[type-arg]
    """Build and compile the LangGraph pipeline."""
    graph = StateGraph(PipelineState)

    # Add nodes (wrapped so the debug tracer tags each event with its node)
    graph.add_node("enrich", _traced("enrich", enrich_node))
    graph.add_node("query_gen", _traced("query_gen", query_gen_node))
    graph.add_node("search", _traced("search", search_node))
    graph.add_node("relevance_judge", _traced("relevance_judge", relevance_judge_node))
    graph.add_node("fetch_fulltext", _traced("fetch_fulltext", fetch_fulltext_node))
    graph.add_node("extract", _traced("extract", extract_node))
    graph.add_node("quality_gate", _traced("quality_gate", quality_gate_node))
    graph.add_node("synthesize", _traced("synthesize", synthesize_node))
    graph.add_node("validity_check", _traced("validity_check", validity_check_node))

    # Feedback loop nodes
    graph.add_node("refine_search", _traced("refine_search", refine_search_node))
    graph.add_node("cross_enrich", _traced("cross_enrich", cross_enrich_node))
    graph.add_node("refine_extraction", _traced("refine_extraction", refine_extraction_node))

    # Main flow edges
    graph.add_edge(START, "enrich")

    # After enrich: short-circuit to validity_check if the LLM did not
    # recognize the parameter — saves all downstream search/extract/synthesize
    # work for clearly-invalid requests.
    graph.add_conditional_edges(
        "enrich",
        route_after_enrich,
        {"query_gen": "query_gen", "validity_check": "validity_check"},
    )
    graph.add_edge("query_gen", "search")

    # After search → relevance judge → conditional: cross-enrich or fetch fulltext
    graph.add_edge("search", "relevance_judge")
    graph.add_conditional_edges(
        "relevance_judge",
        route_after_deliberation,
        {"cross_enrich": "cross_enrich", "fetch_fulltext": "fetch_fulltext"},
    )

    # Cross-enrich → fetch fulltext
    graph.add_edge("cross_enrich", "fetch_fulltext")

    # Fetch fulltext → extract
    graph.add_edge("fetch_fulltext", "extract")

    # Extract → quality gate
    graph.add_edge("extract", "quality_gate")

    # Quality gate → conditional: synthesize, refine_search, or refine_extraction
    graph.add_conditional_edges(
        "quality_gate",
        route_after_quality_gate,
        {
            "synthesize": "synthesize",
            "refine_search": "refine_search",
            "refine_extraction": "refine_extraction",
        },
    )

    # Loop A: refine_search → search
    graph.add_edge("refine_search", "search")

    # Loop C: refine_extraction → quality_gate
    graph.add_edge("refine_extraction", "quality_gate")

    # Synthesize → validity_check → END
    graph.add_edge("synthesize", "validity_check")
    graph.add_edge("validity_check", END)

    return graph


async def run_parameter_graph(
    parameter: ParameterInput,
    settings: Settings | None = None,
    on_node_complete: ProgressCallback = None,
) -> PipelineResult:
    """Run the LangGraph pipeline for a single parameter."""
    if settings is None:
        settings = get_settings()

    # Install a token accumulator scoped to this task; every `_llm_json_call`
    # invocation from this coroutine (and its asyncio subtasks) will add into it.
    from distribird.agent.extract import reset_token_accumulator

    token_usage = reset_token_accumulator()

    # Install a structured debug trace (no-op unless settings.debug_trace is on).
    trace = diagnostics.start_run(parameter, settings)

    compiled = build_pipeline_graph().compile()

    budget = IterationBudget(
        search_refinement_max=settings.search_refinement_max,
        cross_enrichment_max=settings.cross_enrichment_max,
        extraction_refinement_max=settings.extraction_refinement_max,
        total_llm_calls_max=settings.total_llm_calls_max,
    )

    initial_state: PipelineState = {
        "parameter": parameter,
        "settings_dict": settings.model_dump(),
        "enrichment": None,
        "search_queries": [],
        "all_queries_tried": [],
        "all_papers": [],
        "papers_with_values": [],
        "seen_dois": set(),
        "agent_findings": [],
        "deliberation": None,
        "prior": None,
        "blackboard": [],
        "quality": QualityMetrics(),
        "budget": budget,
        "warnings": [],
        "refinement_context": [],
        "trace_events": [],
    }

    final_state: dict[str, Any] = dict(initial_state)
    async for chunk in compiled.astream(
        cast(Any, initial_state),
        stream_mode="updates",
    ):
        for node_name, node_output in chunk.items():
            if node_name == "__start__" or not node_output:
                continue
            final_state.update(node_output)
            if on_node_complete is not None:
                on_node_complete(node_name, final_state)

    prior = final_state.get("prior")
    if prior is None:
        from distribird.distributions.uninformative import wide_normal_prior

        prior = wide_normal_prior(
            parameter.name,
            parameter.constraints.lower_bound,
            parameter.constraints.upper_bound,
        )

    # Finalize the debug trace: fold in the per-node TraceEvents, persist a JSON
    # artifact, and attach the dict to the result (None when tracing is disabled).
    debug_trace: dict[str, Any] | None = None
    if trace is not None:
        diagnostics.finish(node_events=final_state.get("trace_events", []))
        debug_trace = trace.to_dict()
        try:
            path = diagnostics.write_trace(trace, settings.trace_output_dir)
            logger.info("[trace] wrote debug trace to %s", path)
        except Exception as e:  # pragma: no cover - best-effort persistence
            logger.warning("[trace] failed to write debug trace: %s", e)

    return PipelineResult(
        parameter=parameter,
        prior=prior,
        search_queries=final_state.get("all_queries_tried", []),
        papers_found=len(final_state.get("all_papers", [])),
        values_extracted=sum(
            len(p.extracted_values) for p in final_state.get("papers_with_values", [])
        ),
        warnings=final_state.get("warnings", []),
        enrichment=final_state.get("enrichment"),
        deliberation=final_state.get("deliberation"),
        parameter_validity=final_state.get("parameter_validity", ParameterValidity.UNKNOWN),
        validity_reason=final_state.get("validity_reason", ""),
        validity_signals=final_state.get("validity_signals", {}),
        is_empirical=final_state.get("is_empirical"),
        token_usage=dict(token_usage),
        debug_trace=debug_trace,
    )

"""LangGraph pipeline graph definition."""

from __future__ import annotations

from typing import Any, Callable, cast

from langgraph.graph import END, START, StateGraph

from litopri.agent.nodes import (
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
    route_after_quality_gate,
    search_node,
    synthesize_node,
)
from litopri.agent.state import IterationBudget, PipelineState, QualityMetrics
from litopri.config import Settings, get_settings
from litopri.models import ParameterInput, PipelineResult

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
    "synthesize": ("Fitting prior distribution", 0.15),
    # Loop nodes — zero weight so progress never goes backwards
    "refine_search": ("Refining search strategy", 0.0),
    "refine_extraction": ("Refining value extraction", 0.0),
}


def build_pipeline_graph() -> StateGraph:
    """Build and compile the LangGraph pipeline."""
    graph = StateGraph(PipelineState)

    # Add nodes
    graph.add_node("enrich", enrich_node)
    graph.add_node("query_gen", query_gen_node)
    graph.add_node("search", search_node)
    graph.add_node("relevance_judge", relevance_judge_node)
    graph.add_node("fetch_fulltext", fetch_fulltext_node)
    graph.add_node("extract", extract_node)
    graph.add_node("quality_gate", quality_gate_node)
    graph.add_node("synthesize", synthesize_node)

    # Feedback loop nodes
    graph.add_node("refine_search", refine_search_node)
    graph.add_node("cross_enrich", cross_enrich_node)
    graph.add_node("refine_extraction", refine_extraction_node)

    # Main flow edges
    graph.add_edge(START, "enrich")
    graph.add_edge("enrich", "query_gen")
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

    # Synthesize → END
    graph.add_edge("synthesize", END)

    return graph


async def run_parameter_graph(
    parameter: ParameterInput,
    settings: Settings | None = None,
    on_node_complete: ProgressCallback = None,
) -> PipelineResult:
    """Run the LangGraph pipeline for a single parameter."""
    if settings is None:
        settings = get_settings()

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
        from litopri.distributions.uninformative import wide_normal_prior

        prior = wide_normal_prior(
            parameter.name,
            parameter.constraints.lower_bound,
            parameter.constraints.upper_bound,
        )

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
    )

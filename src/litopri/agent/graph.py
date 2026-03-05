"""LangGraph pipeline graph definition."""

from __future__ import annotations

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
    route_after_deliberation,
    route_after_quality_gate,
    search_node,
    synthesize_node,
)
from litopri.agent.state import IterationBudget, PipelineState, QualityMetrics
from litopri.config import Settings, get_settings
from litopri.models import ParameterInput, PipelineResult


def build_pipeline_graph() -> StateGraph:
    """Build and compile the LangGraph pipeline."""
    graph = StateGraph(PipelineState)

    # Add nodes
    graph.add_node("enrich", enrich_node)
    graph.add_node("query_gen", query_gen_node)
    graph.add_node("search", search_node)
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

    # After search → conditional: cross-enrich or fetch fulltext
    graph.add_conditional_edges(
        "search",
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

    final_state = await compiled.ainvoke(initial_state)

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
            len(p.extracted_values)
            for p in final_state.get("papers_with_values", [])
        ),
        warnings=final_state.get("warnings", []),
        enrichment=final_state.get("enrichment"),
        deliberation=final_state.get("deliberation"),
    )

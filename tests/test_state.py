"""Tests for LangGraph pipeline state and helpers."""

from litopri.agent.state import (
    IterationBudget,
    MessageKind,
    PipelineState,
    QualityMetrics,
    add_papers,
    get_messages,
    post_message,
    update_quality,
)
from litopri.models import ExtractedValue, LiteratureEvidence, ParameterInput


def _make_state(**overrides) -> PipelineState:
    base: PipelineState = {
        "parameter": ParameterInput(name="test", description="test"),
        "settings_dict": {},
        "all_papers": [],
        "papers_with_values": [],
        "seen_dois": set(),
        "blackboard": [],
        "warnings": [],
        "budget": IterationBudget(),
    }
    base.update(overrides)
    return base


def test_add_papers_deduplicates_by_doi():
    state = _make_state()
    p1 = LiteratureEvidence(title="Paper A", doi="10.1/a")
    p2 = LiteratureEvidence(title="Paper B", doi="10.1/b")
    p3 = LiteratureEvidence(title="Paper A dup", doi="10.1/a")

    added = add_papers(state, [p1, p2, p3])
    assert len(added) == 2
    assert len(state["all_papers"]) == 2
    assert "10.1/a" in state["seen_dois"]
    assert "10.1/b" in state["seen_dois"]


def test_add_papers_allows_no_doi():
    state = _make_state()
    p1 = LiteratureEvidence(title="No DOI 1", doi=None)
    p2 = LiteratureEvidence(title="No DOI 2", doi=None)
    added = add_papers(state, [p1, p2])
    assert len(added) == 2


def test_quality_metrics_is_sufficient():
    qm = QualityMetrics(n_total_values=3)
    assert qm.is_sufficient(min_values=2)
    assert not qm.is_sufficient(min_values=5)


def test_quality_metrics_needs_search_refinement():
    qm = QualityMetrics(n_papers_found=5, n_total_values=0)
    assert qm.needs_search_refinement()

    qm2 = QualityMetrics(n_papers_found=5, n_total_values=3)
    assert not qm2.needs_search_refinement()

    qm3 = QualityMetrics(n_papers_found=0, n_total_values=0)
    assert not qm3.needs_search_refinement()


def test_quality_metrics_needs_extraction_refinement():
    qm = QualityMetrics(
        n_total_values=3,
        n_high_confidence_values=0,
        value_cv=2.0,
    )
    assert qm.needs_extraction_refinement()

    qm2 = QualityMetrics(
        n_total_values=3,
        n_high_confidence_values=1,
        value_cv=2.0,
    )
    assert not qm2.needs_extraction_refinement()


def test_iteration_budget_limits():
    budget = IterationBudget(
        search_refinement_max=2,
        cross_enrichment_max=1,
        extraction_refinement_max=1,
        total_llm_calls_max=10,
    )
    assert budget.can_refine_search()
    assert budget.can_cross_enrich()
    assert budget.can_refine_extraction()
    assert budget.has_budget()

    budget.search_refinement_used = 2
    assert not budget.can_refine_search()

    budget.total_llm_calls_used = 10
    assert not budget.has_budget()
    assert not budget.can_cross_enrich()


def test_blackboard_post_and_get():
    state = _make_state()
    post_message(state, "agent_a", MessageKind.DISCOVERY, "Found something", ["10.1/x"])
    post_message(state, "agent_b", MessageKind.TERMINOLOGY, "alt name")

    msgs = get_messages(state)
    assert len(msgs) == 2

    msgs_disc = get_messages(state, kind=MessageKind.DISCOVERY)
    assert len(msgs_disc) == 1
    assert msgs_disc[0].sender == "agent_a"

    msgs_excl = get_messages(state, exclude_sender="agent_a")
    assert len(msgs_excl) == 1
    assert msgs_excl[0].sender == "agent_b"


def test_update_quality():
    papers = [
        LiteratureEvidence(
            title="P1",
            extracted_values=[ExtractedValue(reported_value=5.0)],
        ),
        LiteratureEvidence(
            title="P2",
            extracted_values=[ExtractedValue(reported_value=6.0)],
        ),
    ]
    state = _make_state(
        all_papers=papers,
        papers_with_values=papers,
    )
    qm = update_quality(state)
    assert qm.n_papers_found == 2
    assert qm.n_total_values == 2
    assert qm.extraction_coverage == 1.0
    assert qm.value_cv is not None

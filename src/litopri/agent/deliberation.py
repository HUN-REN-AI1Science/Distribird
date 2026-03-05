"""Multi-agent orchestration and moderator deliberation."""

from __future__ import annotations

import asyncio
import logging

from openai import OpenAI

from litopri.agent.agents import (
    DeepResearchAgent,
    OpenAlexAgent,
    SemanticScholarAgent,
    WebSearchAgent,
)
from litopri.agent.extract import _llm_json_call
from litopri.agent.prompts import DELIBERATION_MODERATOR
from litopri.config import Settings
from litopri.models import (
    AgentFinding,
    DeliberationResult,
    EnrichedContext,
    LiteratureEvidence,
    ParameterInput,
)

logger = logging.getLogger(__name__)


async def run_source_agents(
    parameter: ParameterInput,
    queries: list[str],
    settings: Settings,
    enrichment: EnrichedContext | None = None,
) -> list[AgentFinding]:
    """Run enabled source agents in parallel, collecting findings."""
    agents = []
    if settings.enable_semantic_scholar:
        agents.append(SemanticScholarAgent())
    if settings.enable_web_search_agent and settings.llm_web_search:
        agents.append(WebSearchAgent())
    if settings.enable_openalex:
        agents.append(OpenAlexAgent())
    if settings.enable_llm_deep_research:
        agents.append(DeepResearchAgent())

    if not agents:
        logger.warning("[deliberation] no agents enabled")
        return []

    logger.info(
        "[deliberation] running %d agents: %s",
        len(agents),
        [a.name for a in agents],
    )

    tasks = [
        a.search(parameter, queries, settings, enrichment) for a in agents
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    findings: list[AgentFinding] = []
    for agent, result in zip(agents, results):
        if isinstance(result, Exception):
            logger.warning(
                "[deliberation] agent %s failed: %s", agent.name, result,
            )
        else:
            findings.append(result)

    return findings


def _deduplicate_across_agents(
    findings: list[AgentFinding],
) -> tuple[list[LiteratureEvidence], dict[int, list[str]]]:
    """Merge papers across agents by DOI.

    Returns (all_papers, paper_sources) where paper_sources maps
    paper index → list of agent names that found it.
    Prefers verified metadata when both versions exist.
    """
    all_papers: list[LiteratureEvidence] = []
    paper_sources: dict[int, list[str]] = {}
    doi_to_index: dict[str, int] = {}

    for finding in findings:
        for paper in finding.papers:
            doi = paper.doi.strip().lower() if paper.doi else None
            if doi and doi in doi_to_index:
                idx = doi_to_index[doi]
                paper_sources[idx].append(finding.agent_name)
                # Prefer verified metadata
                if paper.verified and not all_papers[idx].verified:
                    all_papers[idx] = paper
            else:
                idx = len(all_papers)
                all_papers.append(paper)
                paper_sources[idx] = [finding.agent_name]
                if doi:
                    doi_to_index[doi] = idx

    return all_papers, paper_sources


def _build_context_block(enrichment: EnrichedContext | None) -> str:
    """Build a context summary block for the deliberation prompt."""
    if enrichment is None:
        return ""
    parts = []
    if enrichment.application_context:
        parts.append(f"Application context: {enrichment.application_context}")
    if enrichment.context_keywords:
        parts.append(f"Context keywords: {', '.join(enrichment.context_keywords)}")
    if enrichment.typical_range:
        parts.append(f"Expected range: {enrichment.typical_range}")
    return "\n".join(parts)


def _build_deliberation_prompt(
    papers: list[LiteratureEvidence],
    paper_sources: dict[int, list[str]],
    parameter: ParameterInput,
    enrichment: EnrichedContext | None = None,
) -> str:
    """Format papers into a numbered list for the moderator prompt."""
    lines = []
    for i, paper in enumerate(papers):
        sources = paper_sources.get(i, ["unknown"])
        sources_str = ", ".join(sources)
        abstract_snippet = (
            paper.abstract[:200] + "..." if len(paper.abstract) > 200 else paper.abstract
        )
        # Use 1-based numbering to match UI reference display
        lines.append(
            f"[{i + 1}] Title: {paper.title}\n"
            f"    Year: {paper.year}, DOI: {paper.doi}\n"
            f"    Verified: {paper.verified}, Source: {paper.source}\n"
            f"    Relevance: {paper.relevance_score:.2f}\n"
            f"    Found by: {sources_str}\n"
            f"    Abstract: {abstract_snippet}"
        )

    papers_block = "\n\n".join(lines)
    context_block = _build_context_block(enrichment)
    return DELIBERATION_MODERATOR.format(
        name=parameter.name,
        description=parameter.description,
        unit=parameter.unit,
        domain_context=parameter.domain_context,
        context_block=context_block,
        papers_block=papers_block,
    )


async def deliberate(
    findings: list[AgentFinding],
    parameter: ParameterInput,
    settings: Settings,
    enrichment: EnrichedContext | None = None,
) -> DeliberationResult:
    """Run moderator deliberation over agent findings.

    Deduplicates papers, then either skips LLM (single-agent all-verified)
    or calls the moderator LLM to select papers.
    """
    all_papers, paper_sources = _deduplicate_across_agents(findings)

    if not all_papers:
        return DeliberationResult(
            agent_findings=findings,
            warnings=["No papers found by any agent."],
        )

    # If only one agent contributed and all papers are verified, skip LLM
    contributing_agents = set()
    for sources in paper_sources.values():
        contributing_agents.update(sources)
    all_verified = all(p.verified for p in all_papers)

    if len(contributing_agents) <= 1 and all_verified:
        logger.info(
            "[deliberation] single agent, all verified — skipping moderator LLM",
        )
        return DeliberationResult(
            consensus_papers=all_papers,
            agent_findings=findings,
            moderator_rationale="Single agent with all verified papers; no deliberation needed.",
        )

    # Call moderator LLM
    prompt = _build_deliberation_prompt(all_papers, paper_sources, parameter, enrichment)
    model = settings.deliberation_model or settings.llm_model

    try:
        client = OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)
        result = _llm_json_call(
            client,
            model,
            [{"role": "user", "content": prompt}],
            temperature=0.1,
        )

        if not isinstance(result, dict):
            raise ValueError("Moderator LLM returned non-dict response")

        raw_selected = result.get("selected_papers", [])
        raw_excluded = result.get("excluded_papers", [])
        rationale = result.get("rationale", "")
        warnings = result.get("warnings", [])

        # The prompt uses 1-based numbering; convert to 0-based for indexing.
        # Accept both 0-based and 1-based by checking if max index > len.
        max_idx = max(raw_selected + raw_excluded, default=0)
        offset = 1 if max_idx > 0 and max_idx >= len(all_papers) else 0
        selected_indices = [i - offset for i in raw_selected]
        excluded_indices = [i - offset for i in raw_excluded]

        consensus = [all_papers[i] for i in selected_indices if 0 <= i < len(all_papers)]
        excluded = [all_papers[i] for i in excluded_indices if 0 <= i < len(all_papers)]

        return DeliberationResult(
            consensus_papers=consensus,
            excluded_papers=excluded,
            moderator_rationale=rationale,
            agent_findings=findings,
            warnings=warnings,
        )

    except Exception as e:
        logger.warning(
            "[deliberation] moderator LLM failed: %s, falling back to verified papers", e,
        )
        # Fallback: include all verified papers sorted by relevance
        verified_papers = sorted(
            [p for p in all_papers if p.verified],
            key=lambda p: p.relevance_score,
            reverse=True,
        )
        return DeliberationResult(
            consensus_papers=verified_papers,
            excluded_papers=[p for p in all_papers if not p.verified],
            moderator_rationale=f"Moderator LLM failed ({e}); using all verified papers.",
            agent_findings=findings,
            warnings=[f"Moderator deliberation failed: {e}"],
        )

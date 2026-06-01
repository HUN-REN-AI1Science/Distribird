"""Source agent implementations for multi-agent literature search."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol

from distribird.agent.extract import _llm_json_call
from distribird.agent.llm_client import get_client
from distribird.agent.prompts import WEB_SEARCH_AGENT
from distribird.agent.search import (
    _CONFIDENCE_TO_RELEVANCE,
    llm_deep_research,
    search_all_queries,
    verify_deep_research_papers,
)
from distribird.agent.search_openalex import search_openalex_all_queries
from distribird.config import Settings
from distribird.models import (
    AgentFinding,
    EnrichedContext,
    LiteratureEvidence,
    ParameterInput,
)

if TYPE_CHECKING:
    from distribird.agent.state import BlackboardMessage

logger = logging.getLogger(__name__)


class SourceAgent(Protocol):
    name: str
    source_type: str

    async def search(
        self,
        parameter: ParameterInput,
        queries: list[str],
        settings: Settings,
        enrichment: EnrichedContext | None = None,
        blackboard: list[BlackboardMessage] | None = None,
    ) -> AgentFinding: ...


class SemanticScholarAgent:
    name: str = "semantic_scholar"
    source_type: str = "semantic_scholar"

    async def search(
        self,
        parameter: ParameterInput,
        queries: list[str],
        settings: Settings,
        enrichment: EnrichedContext | None = None,
        blackboard: list[BlackboardMessage] | None = None,
    ) -> AgentFinding:
        # Incorporate cross-ref DOIs from blackboard as extra queries
        extra_queries: list[str] = []
        if blackboard:
            from distribird.agent.state import MessageKind

            for msg in blackboard:
                if msg.kind == MessageKind.CROSS_REF and msg.sender != self.name:
                    extra_queries.extend(msg.references[:3])

        papers = await search_all_queries(queries, settings)
        return AgentFinding(
            agent_name=self.name,
            source_type=self.source_type,
            papers=papers,
            rationale=f"Found {len(papers)} papers via Semantic Scholar API",
            search_metadata={"n_queries": len(queries)},
        )


class WebSearchAgent:
    name: str = "web_search"
    source_type: str = "web_search"

    async def search(
        self,
        parameter: ParameterInput,
        queries: list[str],
        settings: Settings,
        enrichment: EnrichedContext | None = None,
        blackboard: list[BlackboardMessage] | None = None,
    ) -> AgentFinding:
        prompt = WEB_SEARCH_AGENT.format(
            name=parameter.name,
            description=parameter.description,
            unit=parameter.unit,
            domain_context=parameter.domain_context,
        )
        client = get_client(settings)
        papers_data = _llm_json_call(
            client,
            settings.llm_model,
            [{"role": "user", "content": prompt}],
            temperature=settings.llm_temperature_creative,
            label="web_search_agent",
        )

        if not isinstance(papers_data, list):
            return AgentFinding(
                agent_name=self.name,
                source_type=self.source_type,
                rationale="LLM returned non-list response",
            )

        papers = []
        for p in papers_data:
            if not isinstance(p, dict):
                continue
            confidence = p.get("confidence", "low")
            relevance = _CONFIDENCE_TO_RELEVANCE.get(confidence, 0.1)
            papers.append(
                LiteratureEvidence(
                    title=p.get("title", ""),
                    authors=p.get("authors", []),
                    year=p.get("year"),
                    doi=p.get("doi"),
                    abstract=p.get("abstract", ""),
                    relevance_score=relevance,
                    verified=False,
                    source="web_search",
                )
            )

        # Verify against Semantic Scholar
        if papers:
            papers, n_discarded = await verify_deep_research_papers(papers, settings)
            logger.info(
                "[WebSearchAgent] verified=%d discarded=%d",
                len(papers),
                n_discarded,
            )

        return AgentFinding(
            agent_name=self.name,
            source_type=self.source_type,
            papers=papers,
            rationale=f"Found {len(papers)} papers via web search LLM",
        )


class OpenAlexAgent:
    name: str = "openalex"
    source_type: str = "openalex"

    async def search(
        self,
        parameter: ParameterInput,
        queries: list[str],
        settings: Settings,
        enrichment: EnrichedContext | None = None,
        blackboard: list[BlackboardMessage] | None = None,
    ) -> AgentFinding:
        papers = await search_openalex_all_queries(queries, settings)
        return AgentFinding(
            agent_name=self.name,
            source_type=self.source_type,
            papers=papers,
            rationale=f"Found {len(papers)} papers via OpenAlex API",
            search_metadata={"n_queries": len(queries)},
        )


class DeepResearchAgent:
    name: str = "deep_research"
    source_type: str = "llm_deep_research"

    async def search(
        self,
        parameter: ParameterInput,
        queries: list[str],
        settings: Settings,
        enrichment: EnrichedContext | None = None,
        blackboard: list[BlackboardMessage] | None = None,
    ) -> AgentFinding:
        papers = await llm_deep_research(parameter, settings)
        papers, n_discarded = await verify_deep_research_papers(papers, settings)
        return AgentFinding(
            agent_name=self.name,
            source_type=self.source_type,
            papers=papers,
            rationale=f"Found {len(papers)} papers via deep research (discarded {n_discarded})",
            search_metadata={"n_discarded": n_discarded},
        )

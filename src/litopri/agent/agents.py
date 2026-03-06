"""Source agent implementations for multi-agent literature search."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol

from openai import OpenAI

from litopri.agent.extract import _llm_json_call
from litopri.agent.prompts import WEB_SEARCH_AGENT
from litopri.agent.search import (
    llm_deep_research,
    search_all_queries,
    verify_deep_research_papers,
)
from litopri.agent.search_openalex import search_openalex_all_queries
from litopri.config import Settings
from litopri.models import (
    AgentFinding,
    EnrichedContext,
    LiteratureEvidence,
    ParameterInput,
)

if TYPE_CHECKING:
    from litopri.agent.state import BlackboardMessage

logger = logging.getLogger(__name__)

_CONFIDENCE_TO_RELEVANCE = {"high": 0.5, "medium": 0.3, "low": 0.1}


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
            from litopri.agent.state import MessageKind

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
        client = OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)
        papers_data = _llm_json_call(
            client,
            settings.llm_model,
            [{"role": "user", "content": prompt}],
            temperature=0.3,
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

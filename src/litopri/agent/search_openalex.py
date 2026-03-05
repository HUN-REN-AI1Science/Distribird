"""Literature search via the OpenAlex API (open-access only)."""

from __future__ import annotations

import asyncio
import logging

import httpx

from litopri.agent.search import _compute_relevance
from litopri.config import Settings
from litopri.models import LiteratureEvidence

logger = logging.getLogger(__name__)

OPENALEX_BASE = "https://api.openalex.org/works"


def reconstruct_abstract(inverted_index: dict[str, list[int]] | None) -> str:
    """Rebuild plain text from OpenAlex's inverted abstract index.

    OpenAlex stores abstracts as ``{word: [positions]}``.
    """
    if not inverted_index:
        return ""
    pairs: list[tuple[int, str]] = []
    for word, positions in inverted_index.items():
        for pos in positions:
            pairs.append((pos, word))
    pairs.sort()
    return " ".join(word for _, word in pairs)


def _normalize_openalex_doi(doi: str | None) -> str | None:
    """Strip ``https://doi.org/`` prefix from an OpenAlex DOI."""
    if not doi:
        return None
    prefix = "https://doi.org/"
    if doi.startswith(prefix):
        return doi[len(prefix):]
    return doi


async def search_openalex(
    query: str,
    settings: Settings,
    limit: int = 20,
) -> list[LiteratureEvidence]:
    """Search OpenAlex for open-access papers matching *query*."""
    params: dict[str, str | int] = {
        "search": query,
        "per_page": limit,
        "filter": "is_oa:true",
    }
    if settings.openalex_email:
        params["mailto"] = settings.openalex_email

    logger.info("[OpenAlex:search] query=%r limit=%d", query, limit)

    async with httpx.AsyncClient(timeout=settings.extraction_timeout) as client:
        resp = await client.get(OPENALEX_BASE, params=params)
        resp.raise_for_status()
        data = resp.json()

    results = data.get("results") or []
    logger.info("[OpenAlex:search] query=%r results=%d", query, len(results))

    papers: list[LiteratureEvidence] = []
    for item in results:
        doi = _normalize_openalex_doi(item.get("doi"))
        title = item.get("title") or ""
        year = item.get("publication_year")
        cited_by = item.get("cited_by_count") or 0

        # Authors
        authors: list[str] = []
        for authorship in item.get("authorships") or []:
            author = authorship.get("author") or {}
            name = author.get("display_name")
            if name:
                authors.append(name)

        # Abstract
        abstract = reconstruct_abstract(item.get("abstract_inverted_index"))

        # OA URL
        oa_url = None
        best_oa = item.get("best_oa_location") or {}
        oa_url = best_oa.get("pdf_url") or best_oa.get("landing_page_url")

        relevance = _compute_relevance(cited_by, year)

        papers.append(
            LiteratureEvidence(
                title=title,
                authors=authors,
                year=year,
                doi=doi,
                abstract=abstract,
                pdf_url=oa_url,
                relevance_score=relevance,
                verified=True,
                source="openalex",
            )
        )

    return papers


async def search_openalex_all_queries(
    queries: list[str],
    settings: Settings,
) -> list[LiteratureEvidence]:
    """Execute multiple OpenAlex queries and deduplicate by DOI."""
    all_papers: list[LiteratureEvidence] = []
    seen_dois: set[str] = set()

    logger.info("[OpenAlex:search_all] starting queries=%d", len(queries))

    for i, query in enumerate(queries):
        if i > 0:
            await asyncio.sleep(0.5)  # polite rate limiting
        try:
            results = await search_openalex(
                query, settings, limit=settings.max_papers_per_query
            )
            for paper in results:
                if paper.doi and paper.doi in seen_dois:
                    continue
                if paper.doi:
                    seen_dois.add(paper.doi)
                all_papers.append(paper)
        except Exception as e:
            logger.warning("[OpenAlex:search_all] query failed: %r error=%s", query, e)

    all_papers.sort(
        key=lambda p: (p.relevance_score, p.year or 0),
        reverse=True,
    )
    result = all_papers[: settings.max_papers_per_query]
    logger.info(
        "[OpenAlex:search_all] done total_unique=%d returned=%d",
        len(all_papers), len(result),
    )
    return result

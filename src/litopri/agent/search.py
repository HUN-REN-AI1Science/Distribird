"""Literature search: Semantic Scholar API + LLM deep research fallback."""

from __future__ import annotations

import asyncio
import json
import logging

import httpx
from openai import OpenAI

from litopri.agent.extract import _llm_json_call
from litopri.config import Settings
from litopri.models import EnrichedContext, LiteratureEvidence, ParameterInput

logger = logging.getLogger(__name__)

# Deep research confidence → relevance_score mapping
_CONFIDENCE_TO_RELEVANCE = {"high": 0.5, "medium": 0.3, "low": 0.1}


def _compute_relevance(citation_count: int, year: int | None) -> float:
    """Compute relevance score from citation count and publication year.

    Uses annual citation rate and a recency bonus for papers < 5 years old.
    """
    import datetime

    current_year = datetime.datetime.now(datetime.timezone.utc).year
    pub_year = year or current_year

    years_since = max(1, current_year - pub_year + 1)
    annual_citations = citation_count / years_since
    score = annual_citations / 10.0

    # Recency bonus: up to +0.2 for papers from the last 5 years
    age = current_year - pub_year
    if age < 5:
        score += 0.2 * (1 - age / 5)

    return min(1.0, score)


async def search_semantic_scholar(
    query: str,
    settings: Settings,
    limit: int = 20,
) -> list[LiteratureEvidence]:
    """Search Semantic Scholar API for papers matching the query."""
    headers = {}
    if settings.semantic_scholar_api_key:
        headers["x-api-key"] = settings.semantic_scholar_api_key

    # Request extra papers to compensate for OA filtering
    api_limit = min(limit * 2, 100)
    params = {
        "query": query,
        "limit": api_limit,
        "fields": "title,authors,year,externalIds,abstract,citationCount,openAccessPdf",
    }

    logger.info(
        "[S2:search] query=%r limit=%d url=%s",
        query, limit, settings.semantic_scholar_base_url,
    )

    async with httpx.AsyncClient(timeout=settings.extraction_timeout) as client:
        resp = await client.get(
            f"{settings.semantic_scholar_base_url}/paper/search",
            params=params,
            headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()

    total = len(data.get("data", []))
    logger.info("[S2:search] query=%r results=%d", query, total)

    papers = []
    for item in data.get("data", []):
        doi = None
        ext_ids = item.get("externalIds") or {}
        if isinstance(ext_ids, dict):
            doi = ext_ids.get("DOI")

        authors = []
        for a in item.get("authors") or []:
            if isinstance(a, dict) and "name" in a:
                authors.append(a["name"])

        citation_count = item.get("citationCount") or 0
        year = item.get("year")
        relevance = _compute_relevance(citation_count, year)

        oa_pdf = item.get("openAccessPdf") or {}
        pdf_url = oa_pdf.get("url") if isinstance(oa_pdf, dict) else None

        # Skip papers without open-access full text
        if pdf_url is None:
            continue

        papers.append(
            LiteratureEvidence(
                title=item.get("title", ""),
                authors=authors,
                year=year,
                doi=doi,
                abstract=item.get("abstract") or "",
                pdf_url=pdf_url,
                relevance_score=relevance,
                verified=True,
                source="semantic_scholar",
            )
        )

    logger.info(
        "[S2:search] query=%r OA filter: %d/%d papers have open-access PDF",
        query, len(papers), total,
    )
    return papers[:limit]


async def search_all_queries(
    queries: list[str],
    settings: Settings,
) -> list[LiteratureEvidence]:
    """Execute multiple search queries and deduplicate results by DOI."""
    all_papers: list[LiteratureEvidence] = []
    seen_dois: set[str] = set()

    logger.info("[S2:search_all] starting queries=%d", len(queries))

    for i, query in enumerate(queries):
        if i > 0:
            await asyncio.sleep(1.1)  # Semantic Scholar rate limit: ~1 req/sec
        try:
            results = await search_semantic_scholar(
                query, settings, limit=settings.max_papers_per_query
            )
            for paper in results:
                if paper.doi and paper.doi in seen_dois:
                    continue
                if paper.doi:
                    seen_dois.add(paper.doi)
                all_papers.append(paper)
        except Exception as e:
            logger.warning("[S2:search_all] query failed: %r error=%s", query, e)

    # Sort by relevance (citation-based) and recency
    all_papers.sort(
        key=lambda p: (p.relevance_score, p.year or 0),
        reverse=True,
    )
    result = all_papers[:settings.max_papers_per_query]
    logger.info(
        "[S2:search_all] done total_unique=%d returned=%d",
        len(all_papers), len(result),
    )
    return result


_RELEVANCE_LLM_SCORE = {"high": 0.9, "medium": 0.5, "low": 0.1}


def judge_paper_relevance(
    papers: list[LiteratureEvidence],
    parameter: ParameterInput,
    settings: Settings,
    enrichment: EnrichedContext | None = None,
    batch_size: int = 10,
) -> int:
    """Judge relevance of papers via LLM. Updates papers in-place.

    Returns the number of LLM calls made.
    """
    from litopri.agent.prompts import RELEVANCE_JUDGMENT

    unjudged = [p for p in papers if not p.relevance_snippet]
    if not unjudged:
        return 0

    enrichment_block = _build_enrichment_block(enrichment)
    client = OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)
    llm_calls = 0

    for start in range(0, len(unjudged), batch_size):
        batch = unjudged[start : start + batch_size]
        papers_block = "\n".join(
            f"[{i}] {p.title} ({p.year}): {(p.abstract or '')[:300]}"
            for i, p in enumerate(batch)
        )

        prompt = RELEVANCE_JUDGMENT.format(
            name=parameter.name,
            description=parameter.description,
            unit=parameter.unit,
            domain_context=parameter.domain_context,
            enrichment_block=enrichment_block,
            papers_block=papers_block,
        )

        try:
            raw = _llm_json_call(
                client, settings.llm_model,
                [{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            llm_calls += 1
        except Exception as e:
            logger.warning("[relevance] LLM call failed: %s", e)
            llm_calls += 1
            continue

        if not isinstance(raw, dict):
            continue

        for i, paper in enumerate(batch):
            entry = raw.get(str(i), {})
            if not isinstance(entry, dict):
                continue
            level = entry.get("relevance", "low")
            snippet = entry.get("snippet", "")
            llm_score = _RELEVANCE_LLM_SCORE.get(level, 0.1)
            citation_score = paper.relevance_score
            paper.relevance_score = 0.7 * llm_score + 0.3 * citation_score
            paper.relevance_snippet = snippet or ""

    return llm_calls


async def fetch_citations(
    paper_id: str,
    settings: Settings,
    direction: str = "citations",
    limit: int = 10,
) -> list[LiteratureEvidence]:
    """Fetch citing or referenced papers from S2 API.

    direction: "citations" (forward) or "references" (backward).
    """
    headers = {}
    if settings.semantic_scholar_api_key:
        headers["x-api-key"] = settings.semantic_scholar_api_key

    url = (
        f"{settings.semantic_scholar_base_url}/paper/{paper_id}/{direction}"
        f"?fields=title,authors,year,externalIds,abstract,citationCount,openAccessPdf"
        f"&limit={limit}"
    )

    logger.info("[S2:snowball] paper=%s direction=%s limit=%d", paper_id, direction, limit)

    try:
        async with httpx.AsyncClient(timeout=settings.extraction_timeout) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        logger.warning("[S2:snowball] failed paper=%s: %s", paper_id, e)
        return []

    source_tag = f"snowball_{direction.rstrip('s')}"  # snowball_citation or snowball_reference
    papers: list[LiteratureEvidence] = []
    for entry in data.get("data") or []:
        item = entry.get("citingPaper" if direction == "citations" else "citedPaper", entry)
        if not isinstance(item, dict) or not item.get("title"):
            continue

        oa_pdf = item.get("openAccessPdf") or {}
        pdf_url = oa_pdf.get("url") if isinstance(oa_pdf, dict) else None
        if pdf_url is None:
            continue

        ext_ids = item.get("externalIds") or {}
        doi = ext_ids.get("DOI") if isinstance(ext_ids, dict) else None

        authors = []
        for a in item.get("authors") or []:
            if isinstance(a, dict) and "name" in a:
                authors.append(a["name"])

        citation_count = item.get("citationCount") or 0
        year = item.get("year")

        papers.append(
            LiteratureEvidence(
                title=item["title"],
                authors=authors,
                year=year,
                doi=doi,
                abstract=item.get("abstract") or "",
                pdf_url=pdf_url,
                relevance_score=_compute_relevance(citation_count, year),
                verified=True,
                source=source_tag,
            )
        )

    logger.info("[S2:snowball] paper=%s direction=%s found=%d", paper_id, direction, len(papers))
    return papers


async def snowball_papers(
    seed_papers: list[LiteratureEvidence],
    settings: Settings,
    existing_dois: set[str],
    max_seeds: int = 3,
    limit_per_seed: int = 10,
) -> list[LiteratureEvidence]:
    """Snowball from top seed papers via forward + backward citations.

    Returns deduplicated new papers (excluding those in existing_dois).
    """
    seeds = sorted(seed_papers, key=lambda p: p.relevance_score, reverse=True)[:max_seeds]
    all_new: list[LiteratureEvidence] = []
    seen: set[str] = set(existing_dois)

    for seed in seeds:
        if not seed.doi:
            continue
        paper_id = f"DOI:{seed.doi}"
        for direction in ("citations", "references"):
            await asyncio.sleep(1.1)  # S2 rate limit
            found = await fetch_citations(paper_id, settings, direction, limit_per_seed)
            for p in found:
                doi = p.doi.strip().lower() if p.doi else None
                if doi and doi in seen:
                    continue
                if doi:
                    seen.add(doi)
                all_new.append(p)

    logger.info("[snowball] seeds=%d new_papers=%d", len(seeds), len(all_new))
    return all_new


def _build_enrichment_block(enrichment: EnrichedContext | None) -> str:
    """Build the enrichment block for the search query prompt."""
    if enrichment is None or (not enrichment.common_terminology and not enrichment.search_hints):
        return ""
    lines = [
        "\nContext enrichment (use these terms instead of model-internal jargon):"
    ]
    if enrichment.common_terminology:
        lines.append(f"  Common scientific terms: {', '.join(enrichment.common_terminology)}")
    if enrichment.search_hints:
        lines.append(f"  Search hints: {', '.join(enrichment.search_hints)}")
    if enrichment.enriched_description:
        lines.append(f"  Enriched description: {enrichment.enriched_description}")
    if enrichment.application_context:
        lines.append(f"  Application context: {enrichment.application_context}")
    if enrichment.context_keywords:
        lines.append(f"  Context-specific keywords: {', '.join(enrichment.context_keywords)}")
    return "\n".join(lines)


def generate_search_queries(
    parameter: ParameterInput,
    settings: Settings,
    enrichment: EnrichedContext | None = None,
) -> list[str]:
    """Use LLM to generate search queries for a parameter."""
    from litopri.agent.prompts import SEARCH_QUERY_GENERATION

    prompt = SEARCH_QUERY_GENERATION.format(
        n_queries=settings.max_search_queries,
        name=parameter.name,
        description=parameter.description,
        unit=parameter.unit,
        domain_context=parameter.domain_context,
        enrichment_block=_build_enrichment_block(enrichment),
    )

    logger.info(
        "[LLM:search_queries] param=%r model=%s n_queries=%d",
        parameter.name, settings.llm_model, settings.max_search_queries,
    )

    client = OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)
    try:
        queries = _llm_json_call(
            client,
            settings.llm_model,
            [{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        if isinstance(queries, list):
            result = [str(q) for q in queries[: settings.max_search_queries]]
            logger.info("[LLM:search_queries] generated queries=%r", result)
            return result
    except json.JSONDecodeError:
        logger.warning("[LLM:search_queries] JSON parse failed after retries")

    # Fallback: generate simple queries from the parameter info
    fallback = [
        f"{parameter.name} {parameter.domain_context}",
        f"{parameter.description} measurement",
        f"{parameter.name} calibration {parameter.unit}",
    ]
    logger.info("[LLM:search_queries] using fallback queries=%r", fallback)
    return fallback


async def llm_deep_research(
    parameter: ParameterInput,
    settings: Settings,
) -> list[LiteratureEvidence]:
    """Fallback: ask LLM to recall papers with reported values.

    Uses chat completions (compatible with any OpenAI-compatible proxy).
    When ``enable_deep_research_model`` is True, a dedicated deep research
    model with built-in web search is used instead of the default LLM.
    Note: these results may contain hallucinated references — DOIs should be
    cross-checked when possible.
    """
    from litopri.agent.prompts import DEEP_RESEARCH_WEB

    prompt = DEEP_RESEARCH_WEB.format(
        name=parameter.name,
        description=parameter.description,
        unit=parameter.unit,
        domain_context=parameter.domain_context,
    )
    client = OpenAI(
        base_url=settings.deep_research_base_url,
        api_key=settings.deep_research_api_key,
    )
    model = settings.deep_research_model
    extra_body = None
    logger.info(
        "[LLM:deep_research] param=%r model=%s",
        parameter.name, model,
    )

    try:
        papers_data = _llm_json_call(
            client,
            model,
            [{"role": "user", "content": prompt}],
            temperature=0.3,
            extra_body=extra_body,
        )

        if not isinstance(papers_data, list):
            logger.warning("[LLM:deep_research] response not a list, returning empty")
            return []

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
                    source="llm_deep_research",
                )
            )
        logger.info("[LLM:deep_research] param=%r papers_recalled=%d", parameter.name, len(papers))
        return papers

    except Exception as e:
        logger.warning("[LLM:deep_research] failed: %s", e)
        return []


def _normalize_doi(doi: str) -> str:
    """Strip common prefixes and whitespace from a DOI."""
    doi = doi.strip()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if doi.lower().startswith(prefix.lower()):
            doi = doi[len(prefix):]
    return doi.strip()


async def verify_paper_doi(
    paper: LiteratureEvidence,
    settings: Settings,
) -> LiteratureEvidence | None:
    """Verify a paper's DOI against Semantic Scholar.

    Returns the paper with S2 metadata if verified, or None if unverifiable.
    """
    if not paper.doi:
        logger.info("[S2:verify] no DOI, discarding paper=%r", paper.title)
        return None

    doi = _normalize_doi(paper.doi)
    headers = {}
    if settings.semantic_scholar_api_key:
        headers["x-api-key"] = settings.semantic_scholar_api_key

    url = (
        f"{settings.semantic_scholar_base_url}/paper/DOI:{doi}"
        "?fields=title,authors,year,externalIds,abstract,citationCount,openAccessPdf"
    )

    logger.info("[S2:verify] looking up DOI=%s", doi)

    try:
        async with httpx.AsyncClient(timeout=settings.extraction_timeout) as client:
            resp = await client.get(url, headers=headers)

        if resp.status_code == 404:
            logger.info("[S2:verify] DOI=%s not found (404)", doi)
            return None

        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning("[S2:verify] DOI=%s lookup failed: %s", doi, e)
        return None

    # Replace metadata with verified S2 data
    authors = []
    for a in data.get("authors") or []:
        if isinstance(a, dict) and "name" in a:
            authors.append(a["name"])

    ext_ids = data.get("externalIds") or {}
    verified_doi = ext_ids.get("DOI") if isinstance(ext_ids, dict) else doi

    citation_count = data.get("citationCount") or 0
    year = data.get("year")
    relevance = _compute_relevance(citation_count, year)

    oa_pdf = data.get("openAccessPdf") or {}
    pdf_url = oa_pdf.get("url") if isinstance(oa_pdf, dict) else None

    paper.title = data.get("title") or paper.title
    paper.authors = authors or paper.authors
    paper.year = year or paper.year
    paper.doi = verified_doi or paper.doi
    paper.abstract = data.get("abstract") or paper.abstract
    paper.pdf_url = pdf_url or paper.pdf_url
    paper.relevance_score = relevance
    paper.verified = True

    logger.info("[S2:verify] DOI=%s verified title=%r", doi, paper.title)
    return paper


async def verify_deep_research_papers(
    papers: list[LiteratureEvidence],
    settings: Settings,
) -> tuple[list[LiteratureEvidence], int]:
    """Verify deep-research papers against Semantic Scholar.

    Returns (verified_papers, n_discarded).
    """
    verified: list[LiteratureEvidence] = []
    discarded = 0

    for i, paper in enumerate(papers):
        if i > 0:
            await asyncio.sleep(1.1)  # S2 rate limit
        result = await verify_paper_doi(paper, settings)
        if result is not None:
            verified.append(result)
        else:
            discarded += 1
            logger.info(
                "[S2:verify_batch] discarded paper=%r doi=%s",
                paper.title, paper.doi,
            )

    logger.info(
        "[S2:verify_batch] verified=%d discarded=%d",
        len(verified), discarded,
    )
    return verified, discarded

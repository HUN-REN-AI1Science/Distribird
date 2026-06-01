"""Fetch and extract full text from open access papers."""

from __future__ import annotations

import asyncio
import logging
import re
from urllib.parse import urljoin, urlparse

import httpx
import pymupdf

from distribird.agent import diagnostics
from distribird.config import Settings
from distribird.models import LiteratureEvidence

logger = logging.getLogger(__name__)

# Max characters to keep from full text (LLM context limit)
MAX_TEXT_CHARS = 30_000

# Browser-like request headers. Many publishers (e.g. MDPI/Cloudflare) reject the
# default `python-httpx/x.y` User-Agent with 403; a realistic browser header set
# gets past naive bot filters. Note: this does NOT defeat full JS bot challenges.
_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/pdf,text/html;q=0.9,application/xhtml+xml;q=0.8,*/*;q=0.7",
    "Accept-Language": "en-US,en;q=0.9",
}

# Direct-PDF pointer embedded by most repositories/publishers (Google Scholar
# standard). Lets us turn an HTML landing page into the actual PDF. Both attribute
# orderings occur in the wild.
_CITATION_PDF_URL = re.compile(
    r"""<meta[^>]+name=["']citation_pdf_url["'][^>]+content=["']([^"']+)["']"""
    r"""|<meta[^>]+content=["']([^"']+)["'][^>]+name=["']citation_pdf_url["']""",
    re.IGNORECASE,
)

# Section headers that likely contain parameter values
_PRIORITY_SECTIONS = re.compile(
    r"(?i)^(?:2\.?|3\.?|4\.?)?\s*"
    r"(?:method|material|result|calibrat|parameteriz|model\s+setup"
    r"|data\s+and\s+method|experimental|discussion)",
)


def _extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes using PyMuPDF."""
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")  # type: ignore[no-untyped-call]
    pages = []
    for page in doc:  # type: ignore[attr-defined]
        pages.append(page.get_text())
    doc.close()  # type: ignore[no-untyped-call]
    return "\n".join(pages)


def _smart_truncate(text: str, max_chars: int = MAX_TEXT_CHARS) -> str:
    """Truncate text, prioritizing Methods/Results/Calibration sections.

    If the full text is under the limit, return as-is.
    Otherwise, try to find and prioritize sections that likely contain
    parameter values (Methods, Results, Calibration).
    """
    if len(text) <= max_chars:
        return text

    lines = text.split("\n")

    # Find priority section boundaries
    priority_start = None
    priority_end = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if _PRIORITY_SECTIONS.match(stripped):
            if priority_start is None:
                priority_start = i
            priority_end = i

    if priority_start is not None:
        # Include from first priority section to some lines after last
        section_end = min(len(lines), (priority_end or priority_start) + 500)
        priority_text = "\n".join(lines[priority_start:section_end])
        if len(priority_text) <= max_chars:
            # Prepend abstract/intro if room
            intro = "\n".join(lines[:priority_start])
            remaining = max_chars - len(priority_text)
            if remaining > 500:
                return intro[:remaining] + "\n...\n" + priority_text
            return priority_text
        return priority_text[:max_chars]

    # No sections found, just truncate from start
    return text[:max_chars]


def _normalize_doi(doi: str) -> str:
    """Strip URL/`doi:` prefixes so a bare DOI is left for API lookups."""
    d = doi.strip()
    d = re.sub(r"(?i)^https?://(dx\.)?doi\.org/", "", d)
    d = re.sub(r"(?i)^doi:\s*", "", d)
    return d


async def _resolve_oa_mirrors(
    client: httpx.AsyncClient, doi: str, email: str, exclude_url: str
) -> list[str]:
    """Ask Unpaywall for alternate open-access PDF URLs for ``doi``.

    Returns candidate PDF URLs (best location first), excluding any that share a
    host with ``exclude_url`` — the host we already failed on. Repository mirrors
    (PMC, institutional repos) usually have no bot wall, unlike publisher sites.
    """
    bare = _normalize_doi(doi)
    if not bare:
        return []
    api = f"https://api.unpaywall.org/v2/{bare}"
    try:
        resp = await client.get(api, params={"email": email or "distribird@example.org"})
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:  # network/JSON/404 — no mirrors available
        logger.info("[fulltext] unpaywall lookup failed for %s: %s", bare, e)
        return []

    excluded_host = urlparse(exclude_url).hostname or ""
    locations = []
    best = data.get("best_oa_location")
    if isinstance(best, dict):
        locations.append(best)
    locations.extend(loc for loc in (data.get("oa_locations") or []) if isinstance(loc, dict))

    urls: list[str] = []
    for loc in locations:
        for key in ("url_for_pdf", "url"):
            url = loc.get(key)
            if not url or url in urls:
                continue
            if (urlparse(url).hostname or "") == excluded_host:
                continue  # same wall we just hit
            urls.append(url)
    return urls


async def _attempt_fetch(
    client: httpx.AsyncClient, url: str, _depth: int = 0
) -> tuple[str, str, str, dict[str, object]]:
    """Fetch and extract text from one PDF URL.

    Returns ``(text, outcome, reason, extra)`` where ``outcome`` is one of
    ``downloaded`` / ``skipped`` / ``failed``. ``text`` is non-empty only on a
    ``downloaded`` outcome. If ``url`` serves an HTML landing page that advertises
    a ``citation_pdf_url``, follow it once to reach the real PDF.
    """
    try:
        resp = await client.get(url)

        # Simple 429 retry for diverse PDF hosts (no shared limiter needed)
        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After")
            try:
                delay = float(retry_after) if retry_after else 5.0
            except (ValueError, TypeError):
                delay = 5.0
            logger.info("[fulltext] 429 from %s — retrying in %.1fs", url, delay)
            await asyncio.sleep(delay)
            resp = await client.get(url)
            if resp.status_code == 429:
                logger.warning("[fulltext] still 429 after retry for %s", url)
                return "", "failed", "rate_limited_429", {}

        resp.raise_for_status()

        content_type = resp.headers.get("content-type", "")
        if "pdf" not in content_type and not url.endswith(".pdf"):
            # HTML landing page — follow its citation_pdf_url to the real PDF (once).
            if _depth == 0 and "html" in content_type:
                m = _CITATION_PDF_URL.search(resp.text)
                pdf_url = (m.group(1) or m.group(2)) if m else None
                if pdf_url and pdf_url != url:
                    logger.info("[fulltext] landing page → citation_pdf_url=%s", pdf_url)
                    return await _attempt_fetch(client, urljoin(url, pdf_url), _depth + 1)
            logger.info("[fulltext] skipping non-PDF content-type=%r", content_type)
            return "", "skipped", "non_pdf_content_type", {"content_type": content_type}

        raw_text = _extract_text_from_pdf(resp.content)
        if not raw_text:
            return "", "failed", "empty_pdf_text", {"n_bytes": len(resp.content)}

        text = _smart_truncate(raw_text)
        logger.info("[fulltext] extracted %d/%d chars from %s", len(text), len(raw_text), url)
        return (
            text,
            "downloaded",
            "ok",
            {
                "content_type": content_type,
                "n_bytes": len(resp.content),
                "n_chars_raw": len(raw_text),
                "n_chars_kept": len(text),
            },
        )
    except Exception as e:
        logger.warning("[fulltext] failed url=%s error=%s", url, e)
        return "", "failed", f"exception: {type(e).__name__}: {e}", {}


async def fetch_paper_fulltext(
    paper: LiteratureEvidence,
    settings: Settings,
) -> str:
    """Fetch full text for a single paper from its open access PDF URL.

    Tries the paper's primary ``pdf_url`` first; if that fails (e.g. a publisher
    bot wall returning 403) and the paper has a DOI, falls back to alternate
    open-access mirrors resolved via Unpaywall. Returns extracted text, or empty
    string if no source yields a PDF.
    """

    def _trace(url: str, outcome: str, reason: str, **extra: object) -> None:
        if diagnostics.enabled():
            diagnostics.record(
                "pdf_fetch",
                {
                    "doi": paper.doi,
                    "title": paper.title[:200],
                    "url": url,
                    "outcome": outcome,
                    "reason": reason,
                    **extra,
                },
            )

    if not paper.pdf_url:
        _trace("", "skipped", "no_pdf_url")
        return ""

    async with httpx.AsyncClient(
        timeout=settings.extraction_timeout,
        follow_redirects=True,
        headers=_BROWSER_HEADERS,
    ) as client:
        logger.info("[fulltext] fetching pdf=%r paper=%r", paper.pdf_url, paper.title[:60])
        text, outcome, reason, extra = await _attempt_fetch(client, paper.pdf_url)
        _trace(paper.pdf_url, outcome, reason, **extra)
        if text:
            return text

        # Primary source failed — try Unpaywall OA mirrors (PMC, repositories).
        if not paper.doi:
            return ""
        mirrors = await _resolve_oa_mirrors(
            client, paper.doi, settings.openalex_email, paper.pdf_url
        )
        if mirrors:
            logger.info("[fulltext] trying %d OA mirror(s) for %r", len(mirrors), paper.doi)
        for mirror in mirrors:
            text, outcome, reason, extra = await _attempt_fetch(client, mirror)
            _trace(mirror, outcome, reason, via="unpaywall", **extra)
            if text:
                return text
        return ""


async def fetch_all_fulltexts(
    papers: list[LiteratureEvidence],
    settings: Settings,
    max_concurrent: int = 5,
) -> int:
    """Fetch full text for all papers with PDF URLs.

    Updates paper.full_text in-place.
    Returns the number of papers with full text fetched.
    """
    import asyncio

    papers_with_urls = [p for p in papers if p.pdf_url]
    if not papers_with_urls:
        logger.info("[fulltext] no papers with PDF URLs")
        return 0

    logger.info(
        "[fulltext] fetching %d/%d papers with PDF URLs",
        len(papers_with_urls),
        len(papers),
    )

    semaphore = asyncio.Semaphore(max_concurrent)

    async def _fetch_one(paper: LiteratureEvidence) -> None:
        async with semaphore:
            text = await fetch_paper_fulltext(paper, settings)
            if text:
                paper.full_text = text

    await asyncio.gather(*[_fetch_one(p) for p in papers_with_urls])

    count = sum(1 for p in papers if p.full_text)
    logger.info("[fulltext] fetched full text for %d papers", count)
    return count

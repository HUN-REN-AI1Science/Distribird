"""Fetch and extract full text from open access papers."""

from __future__ import annotations

import logging
import re

import httpx
import pymupdf

from litopri.config import Settings
from litopri.models import LiteratureEvidence

logger = logging.getLogger(__name__)

# Max characters to keep from full text (LLM context limit)
MAX_TEXT_CHARS = 30_000

# Section headers that likely contain parameter values
_PRIORITY_SECTIONS = re.compile(
    r"(?i)^(?:2\.?|3\.?|4\.?)?\s*"
    r"(?:method|material|result|calibrat|parameteriz|model\s+setup"
    r"|data\s+and\s+method|experimental|discussion)",
)


def _extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes using PyMuPDF."""
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()
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


async def fetch_paper_fulltext(
    paper: LiteratureEvidence,
    settings: Settings,
) -> str:
    """Fetch full text for a single paper from its open access PDF URL.

    Returns extracted text, or empty string on failure.
    """
    if not paper.pdf_url:
        return ""

    logger.info(
        "[fulltext] fetching pdf=%r paper=%r",
        paper.pdf_url,
        paper.title[:60],
    )

    try:
        async with httpx.AsyncClient(
            timeout=settings.extraction_timeout,
            follow_redirects=True,
        ) as client:
            resp = await client.get(paper.pdf_url)
            resp.raise_for_status()

            content_type = resp.headers.get("content-type", "")
            if "pdf" not in content_type and not paper.pdf_url.endswith(".pdf"):
                logger.info(
                    "[fulltext] skipping non-PDF content-type=%r",
                    content_type,
                )
                return ""

            raw_text = _extract_text_from_pdf(resp.content)
            if raw_text:
                text = _smart_truncate(raw_text)
                logger.info(
                    "[fulltext] extracted %d/%d chars from paper=%r",
                    len(text),
                    len(raw_text),
                    paper.title[:60],
                )
                return text
            return ""

    except Exception as e:
        logger.warning(
            "[fulltext] failed paper=%r error=%s",
            paper.title[:60],
            e,
        )
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

"""Fetch and extract full text from open access papers."""

from __future__ import annotations

import asyncio
import logging
import re
from urllib.parse import urljoin, urlparse, urlsplit, urlunsplit

import httpx
import pymupdf

from distribird.agent import diagnostics
from distribird.config import Settings
from distribird.models import LiteratureEvidence

logger = logging.getLogger(__name__)

# Max characters to keep from full text (LLM context limit)
MAX_TEXT_CHARS = 30_000

# After navigating to a host's landing page, wait this long for its JS bot
# challenge (e.g. MDPI's bm-verify ~5s meta-refresh) to set clearance cookies
# before requesting PDFs through the cleared session.
_STEALTH_CHALLENGE_WAIT_MS = 7000

# Browser-like request headers. Many publishers reject the default
# python-httpx User-Agent with a 403, and a realistic browser header set gets
# past simple User-Agent filters. This does not defeat JS bot challenges such as
# MDPI's Akamai bm-verify; the stealth tier handles those. Keep the Chrome
# version reasonably current so it does not look like an old bot.
_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36"
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

# Markers of a JS bot-challenge or access-wall interstitial rather than the
# article itself. When HTML extraction hits one of these, we reject the text so
# the challenge boilerplate is not mistaken for paper content. Only the start of
# the text is scanned (challenges front-load the message) so a long article that
# merely mentions one of these words in its body is not falsely rejected.
_HTML_BOT_CHALLENGE = re.compile(
    r"(?i)(?:verifying you are human|just a moment|checking your browser"
    r"|enable javascript and cookies|please enable cookies|captcha"
    r"|are you a robot|access denied|ddos protection|cf-chl|cloudflare)"
)
_HTML_CHALLENGE_SCAN_CHARS = 1500

# Section headers that likely contain parameter values
_PRIORITY_SECTIONS = re.compile(
    r"(?i)^(?:2\.?|3\.?|4\.?)?\s*"
    r"(?:method|material|result|calibrat|parameteriz|model\s+setup"
    r"|data\s+and\s+method|experimental|discussion)",
)


def _extract_text(content: bytes, filetype: str) -> str:
    """Extract text from document bytes using PyMuPDF (``filetype`` pdf or html)."""
    doc = pymupdf.open(stream=content, filetype=filetype)  # type: ignore[no-untyped-call]
    try:
        return "\n".join(page.get_text() for page in doc)  # type: ignore[attr-defined]
    finally:
        doc.close()  # type: ignore[no-untyped-call]


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


def _pdf_bytes_to_text(content: bytes) -> tuple[str, dict[str, object]]:
    """Extract and smart-truncate text from raw PDF bytes.

    Returns ``(text, extra)``; ``text`` is empty when the bytes yield no text.
    ``extra`` carries byte/char counts for the trace.
    """
    raw_text = _extract_text(content, "pdf")
    if not raw_text:
        return "", {"n_bytes": len(content)}
    text = _smart_truncate(raw_text)
    return text, {
        "n_bytes": len(content),
        "n_chars_raw": len(raw_text),
        "n_chars_kept": len(text),
    }


def _html_bytes_to_text(content: bytes, min_chars: int) -> tuple[str, str, dict[str, object]]:
    """Extract and smart-truncate article text from raw HTML bytes.

    Some publishers serve the full article (or a landing page) as HTML rather
    than a PDF. PyMuPDF opens HTML and drops ``<script>``/``<style>``, so we
    reuse the same extraction path as PDFs. A quality gate guards against feeding
    bot-challenge interstitials or thin abstract-only pages to the extractor.

    Returns ``(text, reason, extra)``. On success ``text`` is non-empty and
    ``reason`` is empty; on rejection ``text`` is empty and ``reason`` names why
    (``html_bot_challenge`` / ``html_too_thin`` / ``html_parse_error``).
    """
    try:
        raw_text = _extract_text(content, "html")
    except Exception:
        return "", "html_parse_error", {"n_bytes": len(content)}

    extra: dict[str, object] = {"n_bytes": len(content), "n_chars_raw": len(raw_text)}
    if _HTML_BOT_CHALLENGE.search(raw_text[:_HTML_CHALLENGE_SCAN_CHARS]):
        return "", "html_bot_challenge", extra
    if len(raw_text) < min_chars:
        return "", "html_too_thin", extra
    text = _smart_truncate(raw_text)
    extra["n_chars_kept"] = len(text)
    return text, "", extra


def _bytes_to_text(
    body: bytes, content_type: str, settings: Settings
) -> tuple[str, str, dict[str, object]]:
    """Pick the extractor for ``body`` based on ``content_type`` and run it.

    Returns ``(text, reason, extra)``. HTML responses go through the gated
    :func:`_html_bytes_to_text` (so bot-challenge and thin pages are rejected);
    everything else uses the ungated PDF path. PyMuPDF auto-detects the format
    from the bytes regardless of the ``filetype`` hint, so routing HTML through
    the PDF path would silently bypass the quality gate; this keeps the two
    extractors apart by content-type.
    """
    if settings.enable_html_fulltext and "pdf" not in content_type and "html" in content_type:
        text, reason, extra = _html_bytes_to_text(body, settings.html_fulltext_min_chars)
        return text, ("html_fulltext" if text else reason), extra
    text, extra = _pdf_bytes_to_text(body)
    return text, ("ok" if text else "empty_pdf_text"), extra


def _trace_pdf(
    paper: LiteratureEvidence, url: str, outcome: str, reason: str, **extra: object
) -> None:
    """Record a ``pdf_fetch`` event for this paper (no-op when tracing is off)."""
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
    host with ``exclude_url`` (the host we already failed on). Repository mirrors
    (PMC, institutional repos) usually have no bot wall, unlike publisher sites.

    Unpaywall asks for a real contact email, so this is skipped when
    ``email`` (DISTRIBIRD_OPENALEX_EMAIL) is unset rather than sending a fake one.
    """
    if not email.strip():
        logger.info(
            "[fulltext] skipping Unpaywall OA-mirror lookup; set DISTRIBIRD_OPENALEX_EMAIL "
            "to a real address to enable it"
        )
        return []
    bare = _normalize_doi(doi)
    if not bare:
        return []
    api = f"https://api.unpaywall.org/v2/{bare}"
    try:
        resp = await client.get(api, params={"email": email})
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:  # network/JSON/404, no mirrors available
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
    client: httpx.AsyncClient, url: str, settings: Settings, _depth: int = 0
) -> tuple[str, str, str, dict[str, object]]:
    """Fetch and extract text from one PDF URL.

    Returns ``(text, outcome, reason, extra)`` where ``outcome`` is one of
    ``downloaded`` / ``skipped`` / ``failed``. ``text`` is non-empty only on a
    ``downloaded`` outcome. If ``url`` serves an HTML landing page that advertises
    a ``citation_pdf_url``, follow it once to reach the real PDF; otherwise, when
    ``enable_html_fulltext`` is on, fall back to extracting the article text from
    the HTML itself.
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
        is_html = "html" in content_type
        is_pdf = "pdf" in content_type
        # Route through the gated HTML path whenever the server says HTML — even
        # at a '.pdf' URL. Publishers serve bot-challenge interstitials as
        # text/html at .pdf URLs; trusting the suffix would send the HTML bytes
        # to PyMuPDF (which auto-detects HTML regardless of the filetype hint),
        # smuggling the challenge boilerplate past _HTML_BOT_CHALLENGE as "PDF
        # text". The PDF path is kept for genuine PDFs (pdf content-type, or a
        # .pdf URL with a non-HTML/ambiguous content-type).
        if is_html or (not is_pdf and not url.endswith(".pdf")):
            # HTML landing page — follow its citation_pdf_url to the real PDF (once).
            if _depth == 0 and is_html:
                pdf_url = _citation_pdf_url(resp.text, url)
                if pdf_url:
                    logger.info("[fulltext] landing page → citation_pdf_url=%s", pdf_url)
                    return await _attempt_fetch(client, pdf_url, settings, _depth + 1)
            # No PDF available — try the HTML itself (PMC, repositories, DOAJ).
            if is_html and settings.enable_html_fulltext:
                text, reason, extra = _html_bytes_to_text(
                    resp.content, settings.html_fulltext_min_chars
                )
                extra = {"content_type": content_type, **extra}
                if text:
                    logger.info(
                        "[fulltext] extracted %d chars of HTML full text from %s", len(text), url
                    )
                    return text, "downloaded", "html_fulltext", extra
                logger.info("[fulltext] HTML full text rejected (%s) from %s", reason, url)
                return "", "skipped", reason, extra
            logger.info("[fulltext] skipping non-PDF content-type=%r", content_type)
            return "", "skipped", "non_pdf_content_type", {"content_type": content_type}

        text, extra = _pdf_bytes_to_text(resp.content)
        if not text:
            return "", "failed", "empty_pdf_text", extra
        logger.info(
            "[fulltext] extracted %d/%d chars from %s", len(text), extra["n_chars_raw"], url
        )
        return text, "downloaded", "ok", {"content_type": content_type, **extra}
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
    open-access mirrors resolved via Unpaywall (tier B, gated by
    ``settings.enable_oa_mirror_fallback``). Returns extracted text, or empty
    string if no source yields a PDF. The heavier stealth-browser tier (C) runs
    separately in :func:`fetch_all_fulltexts`.
    """

    if not paper.pdf_url:
        _trace_pdf(paper, "", "skipped", "no_pdf_url")
        return ""

    async with httpx.AsyncClient(
        timeout=settings.extraction_timeout,
        follow_redirects=True,
        headers=_BROWSER_HEADERS,
    ) as client:
        logger.info("[fulltext] fetching pdf=%r paper=%r", paper.pdf_url, paper.title[:60])
        text, outcome, reason, extra = await _attempt_fetch(client, paper.pdf_url, settings)
        _trace_pdf(paper, paper.pdf_url, outcome, reason, **extra)
        if text:
            return text

        # Primary source failed — try Unpaywall OA mirrors (PMC, repositories).
        if not paper.doi or not settings.enable_oa_mirror_fallback:
            return ""
        mirrors = await _resolve_oa_mirrors(
            client, paper.doi, settings.openalex_email, paper.pdf_url
        )
        if mirrors:
            logger.info("[fulltext] trying %d OA mirror(s) for %r", len(mirrors), paper.doi)
        for mirror in mirrors:
            text, outcome, reason, extra = await _attempt_fetch(client, mirror, settings)
            _trace_pdf(paper, mirror, outcome, reason, via="unpaywall", **extra)
            if text:
                return text
        return ""


def _landing_page(pdf_url: str) -> str:
    """Derive a publisher HTML landing page from a PDF URL.

    Used to clear a host's JS bot challenge on a normal page before requesting
    the PDF (e.g. MDPI ``.../<article>/pdf?version=..`` → ``.../<article>``).
    """
    parts = urlsplit(pdf_url)
    path = re.sub(r"(?i)/pdf/?$", "", parts.path)
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def _citation_pdf_url(html: str, base_url: str) -> str | None:
    """Return the absolute ``citation_pdf_url`` advertised by an HTML page, if any."""
    m = _CITATION_PDF_URL.search(html)
    pdf_url = (m.group(1) or m.group(2)) if m else None
    if not pdf_url:
        return None
    absolute = urljoin(base_url, pdf_url)
    return absolute if absolute != base_url else None


async def _stealth_fetch_batch(
    papers: list[LiteratureEvidence],
    settings: Settings,
) -> None:
    """Recover PDFs blocked by JS bot walls via one headless stealth browser.

    Launches a single Camoufox instance, clears each host's bot challenge once on
    a landing page, then downloads every PDF for that host through the cleared
    browser session (cookies + TLS fingerprint of a real browser). Updates
    ``paper.full_text`` in place. Lazily imports ``camoufox`` so the heavy browser
    dependency is only required when ``enable_stealth_fetch`` is on.
    """
    try:
        from camoufox.async_api import AsyncCamoufox
    except Exception as e:
        # ImportError when the extra is missing, but a broken native install can
        # raise other errors; degrade either way instead of aborting the fetch.
        logger.warning(
            "[fulltext] enable_stealth_fetch is on but the stealth browser is "
            "unavailable (%s). Install it with: pip install 'distribird[stealth]' "
            "then python -m camoufox fetch",
            e,
        )
        return

    papers_with_urls = [(p, p.pdf_url) for p in papers if p.pdf_url]
    timeout_ms = int(settings.stealth_fetch_timeout * 1000)
    logger.info("[fulltext] stealth fetch: %d paper(s)", len(papers_with_urls))
    try:
        async with AsyncCamoufox(headless=True) as browser:  # type: ignore[no-untyped-call]
            page = await browser.new_page()
            cleared: set[str] = set()
            for paper, url in papers_with_urls:
                try:
                    # Navigate to the article page first. This follows DOI/handle
                    # redirects to the real publisher and triggers the bot challenge
                    # on THAT origin (the resolver, e.g. doi.org, has no challenge of
                    # its own, so clearing it never helped). The interstitial often
                    # never fires "load", so tolerate a timeout.
                    try:
                        await page.goto(
                            _landing_page(url), wait_until="domcontentloaded", timeout=timeout_ms
                        )
                    except Exception as e:
                        logger.info(
                            "[fulltext] stealth landing goto for %r: %s",
                            paper.doi,
                            type(e).__name__,
                        )
                    # Clear the challenge once per RESOLVED publisher host.
                    host = urlparse(page.url).netloc
                    if host and host not in cleared:
                        await page.wait_for_timeout(_STEALTH_CHALLENGE_WAIT_MS)
                        cleared.add(host)
                    # Prefer the publisher's own PDF link advertised on the cleared
                    # page; fall back to the original URL (now reachable with the
                    # publisher's clearance cookies in the browser session).
                    try:
                        pdf_url = _citation_pdf_url(await page.content(), page.url) or url
                    except Exception:
                        pdf_url = url

                    resp = await page.context.request.get(pdf_url, timeout=timeout_ms)
                    if not resp.ok:
                        _trace_pdf(
                            paper, pdf_url, "failed", f"stealth_http_{resp.status}", via="stealth"
                        )
                        continue
                    body = await resp.body()
                    ctype = resp.headers.get("content-type", "")
                    text, reason, extra = _bytes_to_text(body, ctype, settings)
                    if text:
                        paper.full_text = text
                        _trace_pdf(paper, pdf_url, "downloaded", reason, via="stealth", **extra)
                        logger.info(
                            "[fulltext] stealth recovered %r (%d chars)", paper.doi, len(text)
                        )
                    else:
                        _trace_pdf(paper, pdf_url, "failed", reason, via="stealth", **extra)
                except Exception as e:
                    _trace_pdf(
                        paper,
                        url,
                        "failed",
                        f"stealth_exception: {type(e).__name__}: {e}",
                        via="stealth",
                    )
    except Exception as e:
        logger.warning("[fulltext] stealth browser failed: %s", e)


async def fetch_all_fulltexts(
    papers: list[LiteratureEvidence],
    settings: Settings,
    max_concurrent: int = 5,
) -> int:
    """Fetch full text for all papers with PDF URLs.

    Updates paper.full_text in-place.
    Returns the number of papers with full text fetched.
    """
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

    # Last-resort stealth-browser pass for papers still blocked by bot walls.
    if settings.enable_stealth_fetch:
        still_missing = [p for p in papers_with_urls if not p.full_text]
        if still_missing:
            await _stealth_fetch_batch(still_missing, settings)

    count = sum(1 for p in papers if p.full_text)
    logger.info("[fulltext] fetched full text for %d papers", count)
    return count

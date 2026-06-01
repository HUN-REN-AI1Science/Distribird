"""Tests for full-text fetch helpers and the stealth-fetch fallback wiring."""

from unittest.mock import AsyncMock, patch

import httpx
import pymupdf
import respx

from distribird.agent import fulltext
from distribird.config import Settings
from distribird.models import LiteratureEvidence


def _article_html(body: str) -> bytes:
    """A minimal HTML article page wrapping ``body``."""
    return (
        f"<html><head><title>Maize LAI study</title></head>"
        f"<body><h1>Results</h1><p>{body}</p></body></html>"
    ).encode()


def _pdf_bytes(text: str) -> bytes:
    """Build a tiny one-page PDF carrying ``text`` (for real PyMuPDF extraction)."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    out = doc.tobytes()
    doc.close()
    return out


def test_landing_page_strips_pdf_segment_and_query():
    """MDPI-style PDF URLs collapse to their article landing page."""
    assert (
        fulltext._landing_page("https://www.mdpi.com/2073-4395/13/2/532/pdf?version=1676194747")
        == "https://www.mdpi.com/2073-4395/13/2/532"
    )
    # No /pdf suffix → only the query is dropped.
    assert (
        fulltext._landing_page("https://host.org/article/123?x=1")
        == "https://host.org/article/123"
    )


def test_citation_pdf_url_resolves_relative_against_base():
    """citation_pdf_url is returned as an absolute URL; self-references are dropped."""
    base = "https://publisher.example/doi/10.1/x"
    html = '<meta name="citation_pdf_url" content="/doi/pdf/10.1/x">'
    assert fulltext._citation_pdf_url(html, base) == "https://publisher.example/doi/pdf/10.1/x"
    # No meta tag -> None.
    assert fulltext._citation_pdf_url("<html></html>", base) is None
    # A meta that points back at the same page -> None (avoids a self-loop).
    assert (
        fulltext._citation_pdf_url(f'<meta name="citation_pdf_url" content="{base}">', base)
        is None
    )


async def test_oa_mirror_fallback_gated_by_config():
    """When enable_oa_mirror_fallback is off, Unpaywall is never queried."""
    paper = LiteratureEvidence(title="t", doi="10.1/x", pdf_url="https://walled.example/x.pdf")

    async def _fail(client, url, settings, _depth=0):
        return "", "failed", "exception: 403", {}

    with (
        patch.object(fulltext, "_attempt_fetch", _fail),
        patch.object(fulltext, "_resolve_oa_mirrors", AsyncMock(return_value=[])) as mirrors,
    ):
        out = await fulltext.fetch_paper_fulltext(paper, Settings(enable_oa_mirror_fallback=False))

    assert out == ""
    mirrors.assert_not_called()


async def test_stealth_fetch_skipped_when_disabled():
    """With enable_stealth_fetch off, the stealth browser is never invoked."""
    paper = LiteratureEvidence(title="t", doi="10.1/x", pdf_url="https://walled.example/x.pdf")
    settings = Settings(enable_stealth_fetch=False)

    with (
        patch.object(fulltext, "fetch_paper_fulltext", AsyncMock(return_value="")),
        patch.object(fulltext, "_stealth_fetch_batch", AsyncMock()) as stealth,
    ):
        count = await fulltext.fetch_all_fulltexts([paper], settings)

    assert count == 0
    stealth.assert_not_called()


async def test_stealth_fetch_invoked_for_missing_when_enabled():
    """When enabled, only papers still lacking full text are passed to stealth."""
    got = LiteratureEvidence(title="got", doi="10.1/a", pdf_url="https://ok.example/a.pdf")
    got.full_text = "already here"
    missing = LiteratureEvidence(
        title="miss", doi="10.1/b", pdf_url="https://walled.example/b.pdf"
    )
    settings = Settings(enable_stealth_fetch=True)

    with (
        patch.object(fulltext, "fetch_paper_fulltext", AsyncMock(return_value="")),
        patch.object(fulltext, "_stealth_fetch_batch", AsyncMock()) as stealth,
    ):
        await fulltext.fetch_all_fulltexts([got, missing], settings)

    stealth.assert_awaited_once()
    passed = stealth.await_args.args[0]
    assert [p.doi for p in passed] == ["10.1/b"]


async def test_stealth_batch_no_camoufox_is_graceful(monkeypatch):
    """Missing camoufox dependency degrades to a logged no-op, not a crash."""
    import builtins

    real_import = builtins.__import__

    def _blocked(name, *args, **kwargs):
        if name.startswith("camoufox"):
            raise ImportError("camoufox not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked)
    paper = LiteratureEvidence(title="t", doi="10.1/x", pdf_url="https://walled.example/x.pdf")
    # Should return without raising and leave the paper untouched.
    await fulltext._stealth_fetch_batch([paper], Settings(enable_stealth_fetch=True))
    assert paper.full_text == ""


def test_html_bytes_to_text_extracts_article():
    """A full HTML article clears the quality gate and yields its text."""
    body = "The maximum leaf area index of maize reached 5.2 m2/m2. " * 80
    text, reason, extra = fulltext._html_bytes_to_text(_article_html(body), min_chars=2000)
    assert reason == ""
    assert "5.2" in text
    assert extra["n_chars_raw"] >= 2000


def test_html_bytes_to_text_rejects_thin_page():
    """An abstract-only page falls below the min-chars floor and is rejected."""
    text, reason, _ = fulltext._html_bytes_to_text(
        _article_html("Short abstract."), min_chars=2000
    )
    assert text == ""
    assert reason == "html_too_thin"


def test_html_bytes_to_text_rejects_bot_challenge():
    """A bot-challenge interstitial is rejected even when it is long."""
    challenge = _article_html("Verifying you are human. " + ("please wait. " * 300))
    text, reason, _ = fulltext._html_bytes_to_text(challenge, min_chars=2000)
    assert text == ""
    assert reason == "html_bot_challenge"


def test_html_bytes_to_text_keeps_article_mentioning_trigger_word():
    """A long article that merely mentions a trigger word in its body is kept."""
    body = (
        "The maximum leaf area index of maize reached 5.2 m2/m2. " * 80
        + " We discuss Cloudflare CDN access denied errors in passing. "
    )
    text, reason, _ = fulltext._html_bytes_to_text(_article_html(body), min_chars=2000)
    assert reason == ""
    assert "5.2" in text


def test_bytes_to_text_gates_html_served_as_html():
    """An HTML challenge body is rejected, not accepted via the ungated PDF path."""
    challenge = _article_html("Just a moment. " + ("checking your browser. " * 50))
    text, reason, _ = fulltext._bytes_to_text(challenge, "text/html", Settings())
    assert text == ""
    assert reason == "html_bot_challenge"


def test_bytes_to_text_extracts_html_article():
    """A real HTML article body is extracted and labelled html_fulltext."""
    body = "Maize maximum LAI was 5.2 m2/m2 at silking. " * 80
    text, reason, _ = fulltext._bytes_to_text(_article_html(body), "text/html", Settings())
    assert reason == "html_fulltext"
    assert "5.2" in text


def test_bytes_to_text_uses_pdf_path_for_pdf_content_type():
    """A real PDF body uses the ungated PDF path (reason 'ok')."""
    text, reason, _ = fulltext._bytes_to_text(
        _pdf_bytes("Maize LAI 5.2 " * 50), "application/pdf", Settings()
    )
    assert reason == "ok"
    assert "5.2" in text


@respx.mock
async def test_attempt_fetch_extracts_html_fulltext():
    """When the URL serves a full HTML article, it is downloaded as html_fulltext."""
    url = "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1/"
    body = "Maize maximum LAI was 5.2 m2/m2 at silking under irrigation. " * 80
    respx.get(url).mock(
        return_value=httpx.Response(
            200, content=_article_html(body), headers={"content-type": "text/html; charset=utf-8"}
        )
    )
    async with httpx.AsyncClient() as client:
        text, outcome, reason, _ = await fulltext._attempt_fetch(client, url, Settings())
    assert outcome == "downloaded"
    assert reason == "html_fulltext"
    assert "5.2" in text


@respx.mock
async def test_attempt_fetch_html_skipped_when_flag_off():
    """With enable_html_fulltext off, HTML pages still skip as before."""
    url = "https://repo.example/article/1"
    body = "Maize maximum LAI was 5.2 m2/m2. " * 80
    respx.get(url).mock(
        return_value=httpx.Response(
            200, content=_article_html(body), headers={"content-type": "text/html"}
        )
    )
    async with httpx.AsyncClient() as client:
        text, outcome, reason, _ = await fulltext._attempt_fetch(
            client, url, Settings(enable_html_fulltext=False)
        )
    assert text == ""
    assert outcome == "skipped"
    assert reason == "non_pdf_content_type"


@respx.mock
async def test_attempt_fetch_prefers_citation_pdf_url_over_html():
    """A landing page advertising citation_pdf_url is followed to the real PDF."""
    landing = "https://publisher.example/article/1"
    pdf = "https://publisher.example/article/1.pdf"
    html = (
        '<html><head><meta name="citation_pdf_url" content="'
        + pdf
        + '"></head><body>'
        + ("Landing page text. " * 200)
        + "</body></html>"
    ).encode()
    respx.get(landing).mock(
        return_value=httpx.Response(200, content=html, headers={"content-type": "text/html"})
    )
    respx.get(pdf).mock(
        return_value=httpx.Response(
            200,
            content=_pdf_bytes("Real PDF: maize LAI 5.2"),
            headers={"content-type": "application/pdf"},
        )
    )
    async with httpx.AsyncClient() as client:
        text, outcome, reason, _ = await fulltext._attempt_fetch(client, landing, Settings())
    assert outcome == "downloaded"
    assert reason == "ok"  # PDF preferred over HTML extraction
    assert "5.2" in text


def test_smart_truncate_keeps_text_under_storage_cap():
    """Text between the old 30k cut and the storage cap passes through untouched.

    Regression: the historical hard 30k truncation discarded most of a paper.
    """
    text = "Sentence about maize LAI.\n\n" * 4000  # ~108k chars, well under 400k
    assert 30_000 < len(text) < 400_000
    assert fulltext._smart_truncate(text, 400_000) == text


def test_smart_truncate_still_prioritises_above_storage_cap():
    """Beyond the storage cap, Methods/Results prioritisation still applies."""
    body = "Intro paragraph.\n" * 50 + "Methods\n" + "We measured the parameter.\n" * 30000
    assert len(body) > 400_000
    out = fulltext._smart_truncate(body, 400_000)
    assert len(out) <= 400_000
    assert "Methods" in out


def test_pdf_bytes_to_text_respects_max_chars():
    """The PDF extractor truncates to the passed max_chars (threaded from settings)."""
    long_raw = "maize canopy measurement. " * 5000  # ~135k chars
    with patch.object(fulltext, "_extract_text", return_value=long_raw):
        text, extra = fulltext._pdf_bytes_to_text(b"ignored", max_chars=20_000)
    assert len(text) <= 20_000
    assert extra["n_chars_raw"] == len(long_raw)

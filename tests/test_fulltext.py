"""Tests for full-text fetch helpers and the stealth-fetch fallback wiring."""

from unittest.mock import AsyncMock, patch

from distribird.agent import fulltext
from distribird.config import Settings
from distribird.models import LiteratureEvidence


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


async def test_oa_mirror_fallback_gated_by_config():
    """When enable_oa_mirror_fallback is off, Unpaywall is never queried."""
    paper = LiteratureEvidence(title="t", doi="10.1/x", pdf_url="https://walled.example/x.pdf")

    async def _fail(client, url, _depth=0):
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

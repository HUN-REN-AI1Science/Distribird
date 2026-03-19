"""Tests for the shared rate limiter and 429 retry logic."""

from __future__ import annotations

import asyncio
import time

import httpx
import pytest
import respx

from distribird.agent.ratelimit import (
    AsyncRateLimiter,
    get_limiter,
    rate_limited_request,
    reset_limiters,
)


@pytest.fixture(autouse=True)
def _clean_limiters():
    """Reset the global limiter registry between tests."""
    reset_limiters()
    yield
    reset_limiters()


# ---------------------------------------------------------------------------
# AsyncRateLimiter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_limiter_first_acquire_is_immediate():
    limiter = AsyncRateLimiter(rate=1.0, burst=1)
    t0 = time.monotonic()
    await limiter.acquire()
    assert time.monotonic() - t0 < 0.1


@pytest.mark.asyncio
async def test_limiter_enforces_rate():
    """Two back-to-back acquires at 5 req/s should take ~0.2 s total."""
    limiter = AsyncRateLimiter(rate=5.0, burst=1)
    t0 = time.monotonic()
    await limiter.acquire()
    await limiter.acquire()
    elapsed = time.monotonic() - t0
    assert 0.15 <= elapsed <= 0.5


@pytest.mark.asyncio
async def test_limiter_fixed_interval():
    """Three acquires at 1 req/s should take ~2s (fixed interval)."""
    limiter = AsyncRateLimiter(rate=1.0, burst=1)
    t0 = time.monotonic()
    await limiter.acquire()
    await limiter.acquire()
    await limiter.acquire()
    elapsed = time.monotonic() - t0
    assert 1.8 <= elapsed <= 2.5


@pytest.mark.asyncio
async def test_limiter_concurrent_acquires():
    """Concurrent acquires share the same bucket."""
    limiter = AsyncRateLimiter(rate=5.0, burst=1)
    t0 = time.monotonic()
    await asyncio.gather(limiter.acquire(), limiter.acquire(), limiter.acquire())
    elapsed = time.monotonic() - t0
    # 3 acquires at 5/s: first instant, second ~0.2s, third ~0.4s
    assert elapsed >= 0.3


# ---------------------------------------------------------------------------
# get_limiter / reset_limiters
# ---------------------------------------------------------------------------


def test_get_limiter_singleton():
    a = get_limiter("test", rate=1.0)
    b = get_limiter("test", rate=99.0)  # rate ignored for existing
    assert a is b


def test_get_limiter_different_names():
    a = get_limiter("a", rate=1.0)
    b = get_limiter("b", rate=2.0)
    assert a is not b


def test_reset_limiters():
    a = get_limiter("x", rate=1.0)
    reset_limiters()
    b = get_limiter("x", rate=1.0)
    assert a is not b


# ---------------------------------------------------------------------------
# rate_limited_request — 429 retry
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_rate_limited_request_success():
    """Normal 200 response passes through."""
    limiter = AsyncRateLimiter(rate=100.0, burst=10)
    route = respx.get("https://example.com/api").respond(200, json={"ok": True})

    async with httpx.AsyncClient() as client:
        resp = await rate_limited_request(client, "GET", "https://example.com/api", limiter)

    assert resp.status_code == 200
    assert route.call_count == 1


@pytest.mark.asyncio
@respx.mock
async def test_rate_limited_request_retries_on_429():
    """429 then 200 → should retry and succeed."""
    limiter = AsyncRateLimiter(rate=100.0, burst=10)
    route = respx.get("https://example.com/api").mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "0.1"}),
            httpx.Response(200, json={"ok": True}),
        ]
    )

    async with httpx.AsyncClient() as client:
        resp = await rate_limited_request(
            client, "GET", "https://example.com/api", limiter, base_backoff=0.1
        )

    assert resp.status_code == 200
    assert route.call_count == 2


@pytest.mark.asyncio
@respx.mock
async def test_rate_limited_request_exhausts_retries():
    """All 429s → returns last 429 response."""
    limiter = AsyncRateLimiter(rate=100.0, burst=10)
    respx.get("https://example.com/api").respond(429)

    async with httpx.AsyncClient() as client:
        resp = await rate_limited_request(
            client,
            "GET",
            "https://example.com/api",
            limiter,
            max_retries=2,
            base_backoff=0.05,
        )

    assert resp.status_code == 429


@pytest.mark.asyncio
@respx.mock
async def test_rate_limited_request_passes_kwargs():
    """Extra kwargs (params, headers) are forwarded."""
    limiter = AsyncRateLimiter(rate=100.0, burst=10)
    route = respx.get("https://example.com/api").respond(200, json={})

    async with httpx.AsyncClient() as client:
        resp = await rate_limited_request(
            client,
            "GET",
            "https://example.com/api",
            limiter,
            params={"q": "test"},
            headers={"X-Custom": "val"},
        )

    assert resp.status_code == 200
    req = route.calls[0].request
    assert b"q=test" in req.url.raw_path
    assert req.headers["X-Custom"] == "val"

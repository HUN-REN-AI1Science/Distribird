"""Shared async rate limiting and 429 retry for external API calls."""

from __future__ import annotations

import asyncio
import logging
import time

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fixed-interval rate limiter
# ---------------------------------------------------------------------------


class AsyncRateLimiter:
    """Fixed-interval rate limiter shared across all async coroutines.

    Enforces a minimum time gap of ``1/rate`` seconds between consecutive
    requests, regardless of how many coroutines are waiting.  This avoids
    the token-bucket problem where tokens accumulate during HTTP round-trip
    time and allow burst requests that exceed the API's per-second window.

    Parameters
    ----------
    rate : float
        Allowed requests per second.
    burst : int
        Kept for API compatibility but not used by the fixed-interval
        strategy.
    """

    def __init__(self, rate: float, burst: int = 1) -> None:
        if rate <= 0:
            raise ValueError(f"rate must be positive, got {rate}")
        self.rate = rate
        self.burst = burst
        self._interval = 1.0 / rate
        self._next_allowed = time.monotonic()
        # The lock is bound lazily to whichever event loop is running when it is
        # first awaited, and rebuilt if the running loop changes. Callers such as
        # the Streamlit UI run a fresh ``asyncio.run()`` loop per parameter; a
        # single Lock created here would stay bound to the first loop and raise
        # "is bound to a different event loop" on every later parameter.
        self._lock: asyncio.Lock | None = None
        self._lock_loop: asyncio.AbstractEventLoop | None = None

    def _get_lock(self) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        if self._lock is None or self._lock_loop is not loop:
            self._lock = asyncio.Lock()
            self._lock_loop = loop
        return self._lock

    async def acquire(self) -> None:
        """Wait until the next request slot, then reserve it."""
        async with self._get_lock():
            now = time.monotonic()
            if now < self._next_allowed:
                await asyncio.sleep(self._next_allowed - now)
            self._next_allowed = max(time.monotonic(), self._next_allowed) + self._interval


# ---------------------------------------------------------------------------
# Singleton registry
# ---------------------------------------------------------------------------

_limiters: dict[str, AsyncRateLimiter] = {}


def get_limiter(name: str, rate: float, burst: int = 1) -> AsyncRateLimiter:
    """Return the singleton limiter for *name*, creating one if needed."""
    if name not in _limiters:
        _limiters[name] = AsyncRateLimiter(rate=rate, burst=burst)
    return _limiters[name]


def reset_limiters() -> None:
    """Clear all registered limiters (for testing)."""
    _limiters.clear()


# ---------------------------------------------------------------------------
# Rate-limited HTTP request with 429 retry
# ---------------------------------------------------------------------------


async def rate_limited_request(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    limiter: AsyncRateLimiter,
    *,
    max_retries: int = 3,
    base_backoff: float = 2.0,
    **kwargs: object,
) -> httpx.Response:
    """Issue an HTTP request through *limiter* with 429 retry.

    Calls ``limiter.acquire()`` before each attempt.  On a 429 response,
    reads the ``Retry-After`` header (falling back to exponential backoff)
    and retries up to *max_retries* times.

    Returns the :class:`httpx.Response` — the caller is still responsible
    for calling ``raise_for_status()`` on non-429 errors.
    """
    for attempt in range(max_retries + 1):
        await limiter.acquire()
        resp = await client.request(method, url, **kwargs)  # type: ignore[arg-type]

        if resp.status_code != 429:
            return resp

        # 429 — compute delay
        retry_after = resp.headers.get("Retry-After")
        if retry_after is not None:
            try:
                delay = float(retry_after)
            except ValueError:
                delay = base_backoff * (2**attempt)
        else:
            delay = base_backoff * (2**attempt)

        if attempt < max_retries:
            logger.warning(
                "[ratelimit] 429 on %s %s — retry %d/%d in %.1fs",
                method,
                url,
                attempt + 1,
                max_retries,
                delay,
            )
            await asyncio.sleep(delay)
        else:
            logger.warning(
                "[ratelimit] 429 on %s %s — retries exhausted",
                method,
                url,
            )

    return resp

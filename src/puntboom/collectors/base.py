"""Base collector with async httpx, retry, and rate limiting."""

import asyncio
import logging
import time
from typing import Any

import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

logger = logging.getLogger(__name__)


class RateLimiter:
    """Token-bucket rate limiter."""

    def __init__(self, max_requests: int, period: float = 60.0):
        self.max_requests = max_requests
        self.period = period
        self.tokens = max_requests
        self.last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_refill
            self.tokens = min(
                self.max_requests,
                self.tokens + (elapsed / self.period) * self.max_requests,
            )
            self.last_refill = now

            if self.tokens < 1:
                wait_time = (1 - self.tokens) / (self.max_requests / self.period)
                logger.debug(f"Rate limit: waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
                self.tokens = 0
            else:
                self.tokens -= 1


class BaseCollector:
    """Base class for all data collectors."""

    BASE_URL: str = ""
    RATE_LIMIT: int = 10  # requests per minute

    def __init__(self, api_key: str = ""):
        self.api_key = api_key
        self.rate_limiter = RateLimiter(self.RATE_LIMIT)
        self._client: httpx.AsyncClient | None = None

    @property
    def headers(self) -> dict[str, str]:
        return {"Accept": "application/json"}

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.BASE_URL,
                headers=self.headers,
                timeout=30.0,
                follow_redirects=True,
            )
        return self._client

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.ConnectError)),
    )
    async def _request(
        self, method: str, path: str, **kwargs: Any
    ) -> dict[str, Any] | list[Any]:
        await self.rate_limiter.acquire()
        client = await self._get_client()
        response = await client.request(method, path, **kwargs)
        response.raise_for_status()
        return response.json()

    async def get(self, path: str, **kwargs: Any) -> dict[str, Any] | list[Any]:
        return await self._request("GET", path, **kwargs)

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

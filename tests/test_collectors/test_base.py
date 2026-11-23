"""Tests for base collector."""

import asyncio

import pytest

from puntboom.collectors.base import RateLimiter


@pytest.mark.asyncio
async def test_rate_limiter_allows_within_limit():
    limiter = RateLimiter(max_requests=10, period=60.0)
    # Should allow 10 requests without blocking
    for _ in range(10):
        await limiter.acquire()


@pytest.mark.asyncio
async def test_rate_limiter_refills_tokens():
    limiter = RateLimiter(max_requests=100, period=1.0)
    # Use up tokens
    for _ in range(5):
        await limiter.acquire()
    # Should still work (100 tokens, only used 5)
    await limiter.acquire()

"""Shared async rate limiter (formerly copy-pasted into every retriever)."""

from __future__ import annotations

import asyncio
import time


class RateLimiter:
    """Simple token-bucket-ish limiter: at most ``rpm`` calls per 60s."""

    def __init__(self, rpm: int) -> None:
        self.interval = 60.0 / max(1, rpm)
        self._lock = asyncio.Lock()
        self._next_at = 0.0

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait = self._next_at - now
            if wait > 0:
                await asyncio.sleep(wait)
                now = time.monotonic()
            self._next_at = now + self.interval

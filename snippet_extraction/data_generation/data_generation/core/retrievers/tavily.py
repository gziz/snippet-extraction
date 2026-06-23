"""Async Tavily search client with a global rate limiter.

Single entry point: ``TavilySearch.search(query)`` -> list of ``TavilyResult``.
Uses ``search_depth="advanced"`` with ``chunks_per_source`` so ``content`` holds
multiple query-relevant chunks joined by ``[...]`` (same convention as
``exa_align.py``). Each result carries a document-level relevance ``score``.

Chunks are split on ``[...]`` and de-duplicated (Tavily occasionally repeats a
chunk within one document).
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass

import httpx

from ..paths import load_env
from ..ratelimit import RateLimiter

TAVILY_URL = "https://api.tavily.com/search"
TAVILY_EXTRACT_URL = "https://api.tavily.com/extract"


@dataclass
class TavilyResult:
    url: str
    title: str | None
    content: str
    chunks: list[str]
    score: float | None
    raw_content: str | None = None
    published_date: str | None = None


@dataclass
class TavilyExtractResult:
    url: str
    raw_content: str
    success: bool


def _split_chunks(content: str) -> list[str]:
    """Split ``content`` on ``[...]`` markers, strip, drop dups and empties."""
    seen: set[str] = set()
    out: list[str] = []
    for part in (content or "").split("[...]"):
        c = part.strip()
        if not c or c in seen:
            continue
        seen.add(c)
        out.append(c)
    return out


class TavilySearch:
    def __init__(
        self,
        api_key: str | None = None,
        rpm: int = 60,
        num_results: int = 5,
        chunks_per_source: int = 3,
        search_depth: str = "advanced",
        timeout_s: float = 60.0,
    ) -> None:
        load_env()
        self.api_key = api_key or os.environ["TAVILY_API"]
        self.num_results = num_results
        self.chunks_per_source = chunks_per_source
        self.search_depth = search_depth
        self.timeout_s = timeout_s
        self.limiter = RateLimiter(rpm)
        self.client = httpx.AsyncClient(timeout=timeout_s)

    async def aclose(self) -> None:
        await self.client.aclose()

    async def _post_with_retry(self, url: str, payload: dict, *, max_retries: int = 6) -> dict:
        """POST with backoff on Tavily rate limits (HTTP 432 / 429).

        The free Researcher plan trips a burst/RPM limit (432) under sustained
        traffic even when credits remain; we retry with exponential backoff so a
        long run rides through the throttle instead of erroring out.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        attempt = 0
        while True:
            await self.limiter.acquire()
            resp = await self.client.post(url, json=payload, headers=headers)
            if resp.status_code in (429, 432):
                attempt += 1
                if attempt > max_retries:
                    resp.raise_for_status()
                retry_after = None
                try:
                    retry_after = float(resp.headers.get("Retry-After", "") or 0) or None
                except (TypeError, ValueError):
                    retry_after = None
                await asyncio.sleep(retry_after if retry_after else min(2**attempt, 30))
                continue
            resp.raise_for_status()
            return resp.json()

    async def search(self, query: str, *, num_results: int | None = None) -> list[TavilyResult]:
        payload = {
            "query": query,
            "search_depth": self.search_depth,
            "chunks_per_source": self.chunks_per_source,
            "max_results": num_results or self.num_results,
        }
        data = await self._post_with_retry(TAVILY_URL, payload)
        out: list[TavilyResult] = []
        for r in data.get("results", []):
            content = r.get("content", "") or ""
            out.append(
                TavilyResult(
                    url=r.get("url", ""),
                    title=r.get("title"),
                    content=content,
                    chunks=_split_chunks(content),
                    score=r.get("score"),
                    raw_content=r.get("raw_content"),
                    published_date=r.get("published_date"),
                )
            )
        return out

    async def extract(
        self, urls: list[str], *, extract_depth: str = "advanced"
    ) -> list[TavilyExtractResult]:
        """Extract full-page markdown for ``urls`` via Tavily Extract.

        ``advanced`` depth is the default: it renders dynamic/JS-heavy pages
        (e.g. GitHub issues) that ``basic`` silently truncates, and still costs
        only ~0.4 credits/doc. Failed URLs come back with ``success=False`` and
        empty ``raw_content`` (Tavily does not charge for failures).
        """
        if not urls:
            return []
        payload = {
            "urls": urls,
            "extract_depth": extract_depth,
            "format": "markdown",
        }
        data = await self._post_with_retry(TAVILY_EXTRACT_URL, payload)
        out: list[TavilyExtractResult] = []
        for r in data.get("results", []):
            out.append(
                TavilyExtractResult(
                    url=r.get("url", ""),
                    raw_content=r.get("raw_content") or "",
                    success=True,
                )
            )
        for r in data.get("failed_results", []):
            out.append(
                TavilyExtractResult(
                    url=r.get("url", "") if isinstance(r, dict) else str(r),
                    raw_content="",
                    success=False,
                )
            )
        return out

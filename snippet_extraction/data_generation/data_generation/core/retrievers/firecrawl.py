"""Async Firecrawl /v2/search client with scrape.

One call per query: search returns top-N URLs, each with scraped markdown body.
Cost: 2 credits per 10 search results + 1 credit per scraped page.
Rate limit (Free tier): 10 RPM. We mirror Exa's global token-bucket limiter.

Single entry: ``FirecrawlSearch.search(query) -> list[FCResult]``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import httpx

from ..paths import load_env
from ..ratelimit import RateLimiter

FC_URL = "https://api.firecrawl.dev/v2/search"
FC_SCRAPE_URL = "https://api.firecrawl.dev/v2/scrape"


@dataclass
class FCResult:
    url: str
    title: str | None
    description: str | None
    markdown: str
    status_code: int | None
    source_url: str | None
    position: int | None


class FirecrawlSearch:
    def __init__(
        self,
        api_key: str | None = None,
        rpm: int = 6,
        limit: int = 5,
        timeout_s: float = 120.0,
    ) -> None:
        load_env()
        self.api_key = api_key or os.environ["FIRECRAWL_API"]
        self.limit = limit
        self.timeout_s = timeout_s
        self.limiter = RateLimiter(rpm)
        self.client = httpx.AsyncClient(timeout=timeout_s)

    async def aclose(self) -> None:
        await self.client.aclose()

    async def scrape(self, url: str) -> str:
        """Scrape a single URL to main-content markdown.

        Used as the canonical document-body fetcher for the snippet-eval
        harness: every provider's surfaced URL is fetched the same way so
        gold is built from one uniform body, not each provider's own scraper.

        Raises on HTTP / network errors so callers can record the real cause
        (e.g. 402 insufficient credits, 429 rate limit, 5xx) instead of
        silently treating them as "empty body".
        """
        payload = {
            "url": url,
            "formats": ["markdown"],
            "onlyMainContent": True,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        await self.limiter.acquire()
        resp = await self.client.post(FC_SCRAPE_URL, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        d = data.get("data") or {}
        return d.get("markdown") or ""

    async def search(
        self,
        query: str,
        *,
        limit: int | None = None,
    ) -> tuple[list[FCResult], dict]:
        """Returns (results, meta) where meta has credits_used etc. when available."""
        payload = {
            "query": query,
            "limit": limit or self.limit,
            "scrapeOptions": {
                "formats": ["markdown"],
                "onlyMainContent": True,
            },
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        await self.limiter.acquire()
        resp = await self.client.post(FC_URL, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        out: list[FCResult] = []
        # SDK structure: {"success": true, "data": {"web": [{...}, ...]}}  with v2,
        # or sometimes {"data": [...]}. Handle both.
        web = []
        d = data.get("data")
        if isinstance(d, dict):
            web = d.get("web") or []
        elif isinstance(d, list):
            web = d
        for i, r in enumerate(web):
            meta = r.get("metadata") or {}
            out.append(
                FCResult(
                    url=r.get("url") or meta.get("sourceURL", ""),
                    title=r.get("title") or meta.get("title"),
                    description=r.get("description") or meta.get("description"),
                    markdown=r.get("markdown") or "",
                    status_code=meta.get("statusCode"),
                    source_url=meta.get("sourceURL"),
                    position=r.get("position") if r.get("position") is not None else i,
                )
            )
        return out, data

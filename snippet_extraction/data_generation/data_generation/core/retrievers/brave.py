"""Async Brave Web Search client with a global rate limiter.

Single entry point: ``BraveSearch.search(query)`` -> list of ``BraveResult``.
Brave is a GET endpoint. With ``extra_snippets=true`` each web result carries a
main ``description`` plus up to 5 ``extra_snippets`` excerpts. Brave does NOT
return the full page body, so the snippet-eval harness must fetch each surfaced
URL separately to build gold (same as Perplexity).

The provider's extracted text for scoring is ``description`` + ``extra_snippets``
(see ``snippets`` property), de-duplicated.

Note: extra snippets require a plan that enables them; on plans without it the
field is simply absent and ``extra_snippets`` will be empty.
"""

from __future__ import annotations

import html
import os
import re
from dataclasses import dataclass

import httpx

from ..paths import load_env
from ..ratelimit import RateLimiter

BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"

_TAG_RE = re.compile(r"<[^>]+>")


def _clean(s: str | None) -> str:
    """Strip Brave's inline HTML highlight tags (e.g. <strong>) and unescape."""
    if not s:
        return ""
    return html.unescape(_TAG_RE.sub("", s)).strip()


@dataclass
class BraveResult:
    url: str
    title: str | None
    description: str | None
    extra_snippets: list[str]
    position: int | None

    @property
    def snippets(self) -> list[str]:
        """description + extra_snippets, HTML-stripped, de-duplicated, in order."""
        seen: set[str] = set()
        out: list[str] = []
        for s in [self.description, *self.extra_snippets]:
            t = _clean(s)
            if not t or t in seen:
                continue
            seen.add(t)
            out.append(t)
        return out


class BraveSearch:
    def __init__(
        self,
        api_key: str | None = None,
        rpm: int = 60,
        count: int = 5,
        extra_snippets: bool = True,
        timeout_s: float = 30.0,
    ) -> None:
        load_env()
        self.api_key = api_key or os.environ["BRAVE_API"]
        self.count = count
        self.extra_snippets = extra_snippets
        self.timeout_s = timeout_s
        self.limiter = RateLimiter(rpm)
        self.client = httpx.AsyncClient(timeout=timeout_s)

    async def aclose(self) -> None:
        await self.client.aclose()

    async def search(self, query: str, *, count: int | None = None) -> list[BraveResult]:
        params = {
            "q": query,
            "count": count or self.count,
        }
        if self.extra_snippets:
            params["extra_snippets"] = "true"
        headers = {
            "X-Subscription-Token": self.api_key,
            "Accept": "application/json",
        }
        await self.limiter.acquire()
        resp = await self.client.get(BRAVE_URL, params=params, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        results = (data.get("web") or {}).get("results") or []
        out: list[BraveResult] = []
        for i, r in enumerate(results):
            out.append(
                BraveResult(
                    url=r.get("url", ""),
                    title=r.get("title"),
                    description=r.get("description"),
                    extra_snippets=r.get("extra_snippets") or [],
                    position=i,
                )
            )
        return out

"""Async Exa search client with a global rate limiter (10 RPM).

Single entry point: ``ExaSearch.search(query)`` -> list of result dicts with
``{url, title, text, highlights, score, published_date}``. Uses the /search
endpoint with ``contents={text:True, highlights:{numSentences:1, highlightsPerUrl:5}}``
so one HTTP call returns the document body AND the query-aware highlights.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import httpx

from ..paths import load_env
from ..ratelimit import RateLimiter

EXA_URL = "https://api.exa.ai/search"


@dataclass
class ExaResult:
    url: str
    title: str | None
    text: str
    highlights: list[str]
    highlight_scores: list[float]
    score: float | None
    published_date: str | None


class ExaSearch:
    def __init__(
        self,
        api_key: str | None = None,
        rpm: int = 10,
        num_results: int = 5,
        highlights_per_url: int = 5,
        min_highlight_score: float | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        load_env()
        self.api_key = api_key or os.environ["EXA_API"]
        self.num_results = num_results
        self.highlights_per_url = highlights_per_url
        self.min_highlight_score = min_highlight_score
        self.timeout_s = timeout_s
        self.limiter = RateLimiter(rpm)
        self.client = httpx.AsyncClient(timeout=timeout_s)

    async def aclose(self) -> None:
        await self.client.aclose()

    async def search(self, query: str, *, num_results: int | None = None) -> list[ExaResult]:
        payload = {
            "query": query,
            "numResults": num_results or self.num_results,
            "contents": {
                "text": True,
                "highlights": {
                    "highlightsPerUrl": self.highlights_per_url,
                    "query": query,
                },
            },
        }
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
        }
        await self.limiter.acquire()
        resp = await self.client.post(EXA_URL, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        out: list[ExaResult] = []
        for r in data.get("results", []):
            highlights = r.get("highlights", []) or []
            scores = r.get("highlightScores", []) or []
            if self.min_highlight_score is not None and scores:
                kept = [(h, s) for h, s in zip(highlights, scores) if s >= self.min_highlight_score]
                highlights = [h for h, _ in kept]
                scores = [s for _, s in kept]
            out.append(
                ExaResult(
                    url=r.get("url", ""),
                    title=r.get("title"),
                    text=r.get("text", "") or "",
                    highlights=highlights,
                    highlight_scores=scores,
                    score=r.get("score"),
                    published_date=r.get("publishedDate"),
                )
            )
        return out

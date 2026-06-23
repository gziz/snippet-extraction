"""Uniform provider adapters for the snippet-extraction eval.

Each adapter wraps one retriever (Exa, Tavily, Brave) and exposes a single
coroutine ``search(query) -> list[Surfaced]``. A ``Surfaced`` is one document
the provider returned, reduced to the only two things the harness cares about:
its ``url`` and the list of ``snippets`` the provider chose to extract for the
query. The full document body is NOT taken from the provider here — under
Option B the harness fetches a canonical body for ``url`` itself (Firecrawl)
and builds gold from that, so providers are compared on snippet quality alone.

    Exa     snippets = result.highlights
    Tavily  snippets = result.chunks
    Brave   snippets = result.snippets   (description + extra_snippets, HTML-stripped)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..retrievers.brave import BraveSearch
from ..retrievers.exa import ExaSearch
from ..retrievers.tavily import TavilySearch


@dataclass
class Surfaced:
    url: str
    snippets: list[str]
    score: float | None = None
    title: str | None = None


class ProviderAdapter(Protocol):
    name: str

    async def search(self, query: str) -> list[Surfaced]: ...

    async def aclose(self) -> None: ...


class ExaAdapter:
    name = "exa"

    def __init__(self, **kwargs) -> None:
        self.client = ExaSearch(**kwargs)

    async def search(self, query: str) -> list[Surfaced]:
        results = await self.client.search(query)
        return [
            Surfaced(url=r.url, snippets=list(r.highlights), score=r.score, title=r.title)
            for r in results
        ]

    async def aclose(self) -> None:
        await self.client.aclose()


class TavilyAdapter:
    name = "tavily"

    def __init__(self, **kwargs) -> None:
        self.client = TavilySearch(**kwargs)

    async def search(self, query: str) -> list[Surfaced]:
        results = await self.client.search(query)
        return [
            Surfaced(url=r.url, snippets=list(r.chunks), score=r.score, title=r.title)
            for r in results
        ]

    async def aclose(self) -> None:
        await self.client.aclose()


class BraveAdapter:
    name = "brave"

    def __init__(self, **kwargs) -> None:
        self.client = BraveSearch(**kwargs)

    async def search(self, query: str) -> list[Surfaced]:
        results = await self.client.search(query)
        return [
            Surfaced(url=r.url, snippets=r.snippets, score=None, title=r.title) for r in results
        ]

    async def aclose(self) -> None:
        await self.client.aclose()


_ADAPTERS = {
    "exa": ExaAdapter,
    "tavily": TavilyAdapter,
    "brave": BraveAdapter,
}


def make_provider(name: str, **kwargs) -> ProviderAdapter:
    p = name.lower()
    if p not in _ADAPTERS:
        raise ValueError(f"unknown provider: {name!r} (expected one of {sorted(_ADAPTERS)})")
    return _ADAPTERS[p](**kwargs)

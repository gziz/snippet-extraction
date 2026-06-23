"""The scrape driver with fake retrievers: rows, statuses, skips, resume."""

import asyncio
import json
from argparse import Namespace

from data_generation.core.retrievers.firecrawl import FCResult
from data_generation.core.retrievers.tavily import TavilyExtractResult, TavilyResult
from data_generation.pipelines import scrape

GOOD_BODY = "Alpha is a database. It stores rows on disk."
LONG_BODY = " ".join(f"This is filler sentence number {i}." for i in range(350))


def _tavily_result(url, title="t"):
    return TavilyResult(url=url, title=title, content="", chunks=[], score=1.0)


class FakeTavily:
    """Three URLs per query: one good, one excluded host, one empty body."""

    def __init__(self, **kwargs):
        self.extract_calls = []

    async def search(self, query):
        return [
            _tavily_result("https://good.example/a"),
            _tavily_result("https://reddit.com/r/x"),
            _tavily_result("https://empty.example/b"),
        ]

    async def extract(self, urls, extract_depth="advanced"):
        self.extract_calls.append(list(urls))
        out = []
        for u in urls:
            body = GOOD_BODY if "good" in u else ""
            out.append(TavilyExtractResult(url=u, raw_content=body, success=bool(body)))
        return out

    async def aclose(self):
        pass


def _args(queries, out, retriever="tavily", limit=10):
    return Namespace(
        retriever=retriever,
        queries=queries,
        out=out,
        limit=limit,
        rpm=60,
        num_results=5,
        extract_depth="advanced",
    )


def _write_queries(path, qids):
    with path.open("w") as f:
        for q in qids:
            f.write(
                json.dumps(
                    {"qid": q, "query": f"query {q}", "topic": "db", "intent": "definitional"}
                )
                + "\n"
            )


def _rows(path):
    return [json.loads(l) for l in path.open()]


def test_tavily_rows_statuses_and_exclusions(tmp_path, monkeypatch):
    monkeypatch.setattr(scrape, "TavilySearch", FakeTavily)
    queries, out = tmp_path / "q.jsonl", tmp_path / "out.jsonl"
    _write_queries(queries, ["q1"])

    asyncio.run(scrape.main_async(_args(queries, out)))
    rows = _rows(out)

    # excluded host never reaches extraction, empty body writes no row (tavily)
    assert [r["doc_id"] for r in rows] == ["https://good.example/a"]
    row = rows[0]
    assert row["status"] == "ok"
    assert row["origin"] == row["labeler_provider"] == "tavily"
    assert row["labeler_model"] == "tavily-extract-v1"
    assert row["prompt_version"] == "tavily_v1"
    assert row["query_id"] == "q1"
    assert row["body"] == GOOD_BODY
    assert row["units"] and row["n_units"] == len(row["units"])
    assert row["relevant_unit_ids"] == []  # not labeled yet


def test_tavily_excluded_hosts_not_extracted(tmp_path, monkeypatch):
    fake = FakeTavily()
    monkeypatch.setattr(scrape, "TavilySearch", lambda **kw: fake)
    queries, out = tmp_path / "q.jsonl", tmp_path / "out.jsonl"
    _write_queries(queries, ["q1"])
    asyncio.run(scrape.main_async(_args(queries, out)))
    for urls in fake.extract_calls:
        assert not any("reddit.com" in u for u in urls), "paid extraction on excluded host"


def test_resume_skips_done_queries(tmp_path, monkeypatch):
    monkeypatch.setattr(scrape, "TavilySearch", FakeTavily)
    queries, out = tmp_path / "q.jsonl", tmp_path / "out.jsonl"
    _write_queries(queries, ["q1", "q2"])

    asyncio.run(scrape.main_async(_args(queries, out)))
    n_first = len(_rows(out))
    asyncio.run(scrape.main_async(_args(queries, out)))
    assert len(_rows(out)) == n_first, "re-run must not duplicate rows"


def test_long_documents_recorded_but_not_segmented(tmp_path, monkeypatch):
    class LongTavily(FakeTavily):
        async def search(self, query):
            return [_tavily_result("https://long.example/doc")]

        async def extract(self, urls, extract_depth="advanced"):
            return [TavilyExtractResult(url=urls[0], raw_content=LONG_BODY, success=True)]

    monkeypatch.setattr(scrape, "TavilySearch", LongTavily)
    queries, out = tmp_path / "q.jsonl", tmp_path / "out.jsonl"
    _write_queries(queries, ["q1"])
    asyncio.run(scrape.main_async(_args(queries, out)))

    (row,) = _rows(out)
    assert row["status"].startswith("skipped:long(")
    assert row["units"] == []  # body kept for later, units intentionally empty
    assert row["body"] == LONG_BODY


def test_firecrawl_records_empty_bodies(tmp_path, monkeypatch):
    class FakeFirecrawl:
        def __init__(self, **kwargs):
            pass

        async def search(self, query):
            return [
                FCResult(
                    url="https://good.example/a",
                    title="t",
                    description="d",
                    markdown=GOOD_BODY,
                    status_code=200,
                    source_url=None,
                    position=1,
                ),
                FCResult(
                    url="https://empty.example/b",
                    title="t",
                    description=None,
                    markdown="",
                    status_code=200,
                    source_url=None,
                    position=2,
                ),
            ], {}

        async def aclose(self):
            pass

    monkeypatch.setattr(scrape, "FirecrawlSearch", FakeFirecrawl)
    queries, out = tmp_path / "q.jsonl", tmp_path / "out.jsonl"
    _write_queries(queries, ["q1"])
    asyncio.run(scrape.main_async(_args(queries, out, retriever="firecrawl")))

    by_url = {r["doc_id"]: r for r in _rows(out)}
    assert by_url["https://good.example/a"]["status"] == "ok"
    assert by_url["https://good.example/a"]["description"] == "d"
    assert by_url["https://good.example/a"]["prompt_version"] == "fc_v1"
    # firecrawl historically records empty-body rows (tavily does not)
    assert by_url["https://empty.example/b"]["status"] == "skipped:empty_md"
    assert by_url["https://empty.example/b"]["units"] == []

"""Search-and-scrape driver: queries -> scrape rows, one per (query, document).

Stage 1 of the web labeling pipeline (see README). For each query, a retriever
returns top-N URLs with full-page markdown bodies; each body is segmented into
units and written as one scrape row. ``relevant_unit_ids`` stays empty here —
the LLM labeler runs later via ``label_units``.

Retrievers (``--retriever``):

- ``tavily`` (default) — Tavily Search (1 credit/query) + batched Tavily
  Extract at ``advanced`` depth (~0.4 credits/doc). Search ``content`` chunks
  are ignored (already compressed); only the extract ``raw_content`` is kept.
- ``firecrawl`` — Firecrawl /v2/search, which returns scraped markdown in the
  search response itself. Legacy: produced the original corpus before the
  Tavily pivot (2026-06).

Resume-safe: skips any query_id already present in --out. Chrome-heavy /
scrape-hostile hosts are skipped before paying for extraction.

Usage:
    python -m data_generation.pipelines.scrape \
        --retriever tavily \
        --queries data_generation/data/queries/synth_v1_part3_unused.jsonl \
        --out     data_generation/data/labels/tavily_v1.jsonl \
        --limit   200 --num-results 5 --rpm 60
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path

from ..core.retrievers.firecrawl import FirecrawlSearch
from ..core.retrievers.tavily import TavilySearch
from ..core.schema import load_done_qids, make_scrape_row
from ..core.segment import segment

MAX_UNITS_PER_DOC = 300

# Hosts that consistently return empty markdown or have legal/scrape barriers.
EXCLUDE_HOST_SUBSTR = (
    "reddit.com",
    "facebook.com",
    "twitter.com",
    "x.com/",
    "youtube.com",
    "tiktok.com",
    "instagram.com",
    "linkedin.com",
)


def _excluded(url: str) -> bool:
    u = (url or "").lower()
    return any(s in u for s in EXCLUDE_HOST_SUBSTR)


@dataclass
class ScrapedDoc:
    url: str
    title: str | None
    description: str | None
    body: str
    position: int | None


@dataclass(frozen=True)
class RetrieverSpec:
    origin: str  # row "origin" / "labeler_provider"
    fetcher_name: str  # row "labeler_model"
    fetcher_version: str  # row "prompt_version"
    default_rpm: int
    write_empty_rows: bool  # firecrawl historically recorded empty-body rows


SPECS = {
    "tavily": RetrieverSpec("tavily", "tavily-extract-v1", "tavily_v1", 60, False),
    "firecrawl": RetrieverSpec("firecrawl", "firecrawl-v2", "fc_v1", 6, True),
}


async def _fetch_tavily(
    tv: TavilySearch, query: str, extract_depth: str
) -> tuple[list[ScrapedDoc], int]:
    """Search, drop excluded hosts *before* paying for extraction, then extract."""
    results = await tv.search(query)
    kept = [r for r in results if not _excluded(r.url)]
    n_excluded = len(results) - len(kept)
    if not kept:
        return [], n_excluded
    extracts = await tv.extract([r.url for r in kept], extract_depth=extract_depth)
    body_by_url = {e.url: e.raw_content for e in extracts if e.success}
    docs = [
        ScrapedDoc(
            url=r.url,
            title=r.title,
            description=None,
            body=body_by_url.get(r.url, ""),
            position=pos,
        )
        for pos, r in enumerate(kept, 1)
    ]
    return docs, n_excluded


async def _fetch_firecrawl(fc: FirecrawlSearch, query: str) -> tuple[list[ScrapedDoc], int]:
    results, _ = await fc.search(query)
    kept = [r for r in results if not _excluded(r.url)]
    docs = [
        ScrapedDoc(
            url=r.url,
            title=r.title,
            description=r.description,
            body=r.markdown or "",
            position=r.position,
        )
        for r in kept
    ]
    return docs, len(results) - len(kept)


async def scrape_queries(
    *,
    retriever: str,
    queries_path: Path,
    out_path: Path,
    limit: int,
    rpm: int | None = None,
    num_results: int = 5,
    extract_depth: str = "advanced",
) -> dict:
    """Search + scrape full bodies for a query set into ``out_path``.

    Stage 1 of the web pipeline. Resume-safe: queries already present in
    ``out_path`` are skipped. Returns a counts dict.
    """
    spec = SPECS[retriever]
    if rpm is None:
        rpm = spec.default_rpm
    queries = [json.loads(l) for l in queries_path.open()]
    done_qids = load_done_qids(out_path)
    pending = [q for q in queries if str(q["qid"]) not in done_qids][:limit]
    print(
        f"[scrape:{spec.origin}] queries_total={len(queries)} done_qids={len(done_qids)} "
        f"pending={len(pending)} (limit={limit})"
    )

    if retriever == "tavily":
        client = TavilySearch(rpm=rpm, num_results=num_results)

        async def fetch(query: str):
            return await _fetch_tavily(client, query, extract_depth)
    else:
        client = FirecrawlSearch(rpm=rpm, limit=num_results)

        async def fetch(query: str):
            return await _fetch_firecrawl(client, query)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    n_calls = n_written = n_ok = n_empty = n_excluded = n_long = n_errors = 0

    def row(q: dict, doc: ScrapedDoc, body: str, units: list[str], skipped: str | None) -> dict:
        return make_scrape_row(
            qid=str(q["qid"]),
            query=q["query"],
            topic=q.get("topic"),
            intent=q.get("intent"),
            url=doc.url,
            title=doc.title,
            description=doc.description,
            body=body,
            units=units,
            position=doc.position,
            origin=spec.origin,
            fetcher_name=spec.fetcher_name,
            fetcher_version=spec.fetcher_version,
            skipped_reason=skipped,
        )

    try:
        with out_path.open("a") as fout:
            for i, q in enumerate(pending, 1):
                try:
                    docs, n_exc = await fetch(q["query"])
                except Exception as e:
                    n_errors += 1
                    print(f"[{i:>4}/{len(pending)}] ERROR {q['qid']}: {str(e)[:160]}")
                    continue
                n_calls += 1
                n_excluded += n_exc

                for doc in docs:
                    if not doc.body.strip():
                        n_empty += 1
                        if spec.write_empty_rows:
                            fout.write(
                                json.dumps(
                                    row(q, doc, "", [], "skipped:empty_md"), ensure_ascii=False
                                )
                                + "\n"
                            )
                            n_written += 1
                        continue
                    units = segment(doc.body)
                    if not units:
                        n_empty += 1
                        continue
                    if len(units) > MAX_UNITS_PER_DOC:
                        n_long += 1
                        fout.write(
                            json.dumps(
                                row(q, doc, doc.body, [], f"skipped:long({len(units)})"),
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        n_written += 1
                        continue
                    fout.write(
                        json.dumps(row(q, doc, doc.body, units, None), ensure_ascii=False) + "\n"
                    )
                    n_written += 1
                    n_ok += 1
                fout.flush()

                if i % 10 == 0:
                    rate = n_calls / max(1e-9, (time.perf_counter() - t0)) * 60
                    print(
                        f"[{i:>4}/{len(pending)}] calls={n_calls} ok={n_ok} "
                        f"skipped[empty={n_empty} long={n_long} excluded={n_excluded}] "
                        f"errors={n_errors}  rate={rate:.1f}/min",
                        flush=True,
                    )
    finally:
        await client.aclose()

    dt_s = time.perf_counter() - t0
    print(
        f"\n[scrape:{spec.origin}] DONE. queries={n_calls} rows_written={n_written} "
        f"ok={n_ok} skipped[empty={n_empty} long={n_long} excluded={n_excluded}] "
        f"errors={n_errors} wall={dt_s:.1f}s"
    )
    return {
        "queries": n_calls,
        "written": n_written,
        "ok": n_ok,
        "empty": n_empty,
        "long": n_long,
        "excluded": n_excluded,
        "errors": n_errors,
    }


async def main_async(args) -> None:
    await scrape_queries(
        retriever=args.retriever,
        queries_path=args.queries,
        out_path=args.out,
        limit=args.limit,
        rpm=args.rpm,
        num_results=args.num_results,
        extract_depth=args.extract_depth,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--retriever", choices=sorted(SPECS), default="tavily")
    ap.add_argument("--queries", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--rpm", type=int, default=None, help="requests/min (default: per-retriever)")
    ap.add_argument("--num-results", type=int, default=5)
    ap.add_argument(
        "--extract-depth",
        choices=["basic", "advanced"],
        default="advanced",
        help="Tavily Extract depth (tavily only)",
    )
    args = ap.parse_args()
    if args.rpm is None:
        args.rpm = SPECS[args.retriever].default_rpm
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()

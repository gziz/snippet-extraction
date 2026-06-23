"""LEGACY: Exa-highlight-aligned labels (superseded by the LLM-labeler pipeline).

Kept runnable for provenance of exa_v1*.jsonl; new data should come from
``scrape`` + ``label_units``.

Pipeline per query:
  1. Call Exa /search with text + highlights.
  2. For each result: segment(body), drop docs with > MAX_UNITS_PER_DOC units,
     split highlight on `[...]`, align fragments -> kept_unit_ids.
  3. Append one JSONL row per kept-non-empty (q, doc) pair to the labels file.

Resume-safe: skips any (qid, url) already present in the output file. Run with
--concurrency 1 (default) since Exa is rate-limited to 10 RPM globally and
ExaSearch enforces that internally; higher concurrency only adds contention.

Usage:
    python -m data_generation.pipelines.label_exa \
        --queries data_generation/data/queries/synth_v1.jsonl \
        --out     data_generation/data/labels/exa_v1.jsonl \
        --limit   2000 \
        --rpm     10
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import time
from pathlib import Path

from ..core.retrievers.exa import ExaResult, ExaSearch
from ..core.retrievers.exa_align import AlignStats, align, split_fragments
from ..core.segment import segment

MAX_UNITS_PER_DOC = 300
PROMPT_VERSION = "exa_v1"
LABELER_MODEL = "exa-highlights-v1"


def load_done(path: Path) -> set[tuple[str, str]]:
    done: set[tuple[str, str]] = set()
    if not path.exists():
        return done
    with path.open() as f:
        for line in f:
            try:
                row = json.loads(line)
                done.add((row["query_id"], row["doc_id"]))
            except Exception:
                pass
    return done


def make_row(
    *,
    qid: str,
    query: str,
    topic: str | None,
    intent: str | None,
    result: ExaResult,
    units: list[str],
    kept_unit_ids: list[int],
    stats: AlignStats,
    n_fragments: int,
    skipped_reason: str | None,
) -> dict:
    return {
        "query_id": qid,
        "doc_id": result.url,
        "origin": "exa",
        "labeler_provider": "exa",
        "labeler_model": LABELER_MODEL,
        "prompt_version": PROMPT_VERSION,
        "status": skipped_reason or "ok",
        "query": query,
        "topic": topic,
        "intent": intent,
        "url": result.url,
        "title": result.title,
        "published_date": result.published_date,
        "body": result.text,
        "units": units,
        "relevant_unit_ids": kept_unit_ids,
        "n_units": len(units),
        "n_kept": len(kept_unit_ids),
        "n_fragments": n_fragments,
        "n_fragments_matched": stats.n_matched,
        "n_fragments_dropped": stats.n_dropped,
        "exa_score": result.score,
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }


async def main_async(
    queries_path: Path,
    out_path: Path,
    limit: int,
    rpm: int,
    num_results: int,
) -> None:
    queries = [json.loads(l) for l in queries_path.open()]
    done = load_done(out_path)
    # done is keyed on (qid, url). For resume we skip the whole query if any
    # of its results were written (Exa returns the same top-N for a given query).
    done_qids = {q for q, _ in done}
    pending = [q for q in queries if q["qid"] not in done_qids][:limit]
    print(
        f"[exa_label] queries_total={len(queries)} done_qids={len(done_qids)} "
        f"pending={len(pending)} (limit={limit})"
    )

    exa = ExaSearch(rpm=rpm, num_results=num_results)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    n_calls = 0
    n_pairs_written = 0
    n_pairs_kept = 0
    n_skipped_long = 0
    n_skipped_empty = 0
    n_errors = 0

    try:
        with out_path.open("a") as fout:
            for i, q in enumerate(pending, 1):
                qid = q["qid"]
                query = q["query"]
                try:
                    results = await exa.search(query)
                except Exception as e:
                    n_errors += 1
                    print(f"[{i:>4}/{len(pending)}] ERROR {qid}: {e}")
                    continue
                n_calls += 1
                for res in results:
                    if not res.text:
                        continue
                    units = segment(res.text)
                    n_units = len(units)
                    if n_units == 0:
                        continue
                    if n_units > MAX_UNITS_PER_DOC:
                        n_skipped_long += 1
                        row = make_row(
                            qid=qid,
                            query=query,
                            topic=q.get("topic"),
                            intent=q.get("intent"),
                            result=res,
                            units=[],
                            kept_unit_ids=[],
                            stats=AlignStats(0, 0, 0, []),
                            n_fragments=0,
                            skipped_reason=f"skipped:long({n_units})",
                        )
                        fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                        n_pairs_written += 1
                        continue
                    hl = res.highlights[0] if res.highlights else ""
                    frags = split_fragments(hl)
                    stats = align(frags, units)
                    if not stats.matched_unit_ids:
                        n_skipped_empty += 1
                        row = make_row(
                            qid=qid,
                            query=query,
                            topic=q.get("topic"),
                            intent=q.get("intent"),
                            result=res,
                            units=units,
                            kept_unit_ids=[],
                            stats=stats,
                            n_fragments=len(frags),
                            skipped_reason="skipped:empty_align",
                        )
                        fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                        n_pairs_written += 1
                        continue
                    row = make_row(
                        qid=qid,
                        query=query,
                        topic=q.get("topic"),
                        intent=q.get("intent"),
                        result=res,
                        units=units,
                        kept_unit_ids=stats.matched_unit_ids,
                        stats=stats,
                        n_fragments=len(frags),
                        skipped_reason=None,
                    )
                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                    n_pairs_written += 1
                    n_pairs_kept += 1
                fout.flush()
                if i % 20 == 0:
                    dt_s = time.perf_counter() - t0
                    rate = i / dt_s * 60
                    eta = (len(pending) - i) / max(1, i) * dt_s / 60
                    print(
                        f"[{i:>4}/{len(pending)}] calls={n_calls} pairs_kept={n_pairs_kept} "
                        f"skipped_long={n_skipped_long} skipped_empty={n_skipped_empty} "
                        f"errors={n_errors}  rate={rate:.1f}/min  eta={eta:.1f}min"
                    )
    finally:
        await exa.aclose()

    dt_s = time.perf_counter() - t0
    print(
        f"\n[exa_label] DONE. queries={n_calls} pairs_written={n_pairs_written} "
        f"pairs_kept={n_pairs_kept} skipped_long={n_skipped_long} "
        f"skipped_empty={n_skipped_empty} errors={n_errors} wall={dt_s:.1f}s"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=10_000)
    ap.add_argument("--rpm", type=int, default=10)
    ap.add_argument("--num-results", type=int, default=5)
    args = ap.parse_args()
    asyncio.run(main_async(args.queries, args.out, args.limit, args.rpm, args.num_results))


if __name__ == "__main__":
    main()

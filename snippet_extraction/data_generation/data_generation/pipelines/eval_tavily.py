"""Eval Tavily: send N queries to Tavily and dump results for quality review.

Usage:
    python -m data_generation.pipelines.eval_tavily \
        --queries data_generation/data/queries/smoke_v1.jsonl \
        --n 10 \
        --out data_generation/data/queries/eval_v1_tavily.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict
from pathlib import Path

from ..core.retrievers.tavily import TavilySearch


async def main_async(queries_path: Path, n: int, out_path: Path, rpm: int) -> None:
    rows = [json.loads(l) for l in queries_path.open()][:n]
    tav = TavilySearch(rpm=rpm)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fout = out_path.open("w")
    try:
        for i, r in enumerate(rows, 1):
            q = r["query"]
            print(f"\n[{i:>2}/{len(rows)}] {q!r}")
            try:
                results = await tav.search(q)
            except Exception as e:
                print(f"    ERROR: {e}")
                continue
            for j, res in enumerate(results, 1):
                score = res.score if res.score is not None else float("nan")
                print(f"    {j}. ({score:.3f}) {res.url}")
                for k, c in enumerate(res.chunks):
                    print(f"       C{k + 1} ({len(c.split())}w): {c[:160]}")
            fout.write(
                json.dumps(
                    {
                        "qid": r["qid"],
                        "query": q,
                        "topic": r.get("topic"),
                        "intent": r.get("intent"),
                        "results": [asdict(x) for x in results],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            fout.flush()
    finally:
        fout.close()
        await tav.aclose()

    print(f"\n[eval_tavily] wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True, type=Path)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--rpm", type=int, default=60)
    args = ap.parse_args()
    asyncio.run(main_async(args.queries, args.n, args.out, args.rpm))


if __name__ == "__main__":
    main()

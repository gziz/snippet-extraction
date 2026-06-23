"""Run an LLM labeler over stored (query, units) rows -> label rows.

The labeling stage of the pipeline: reads rows from --in (any JSONL carrying
``query`` + ``units`` — scrape rows, joined pairs, or re-segmented rows) and
runs each through the selected labeler client. Writes label rows with
``relevant_unit_ids`` filled in and labeler_provider/labeler_model/prompt_version
reflecting the LLM. It labelers ``units`` as-is — it does not re-segment; pair it
with ``segment_documents`` when the segmenter has changed.

Resume-safe: skips any (query_id, doc_id) already labeled with the same
(provider, model, prompt_version) in the output file.

Usage:
    python -m data_generation.pipelines.label_units \
        --in  data_generation/data/labels/fc2_03_pairs.jsonl \
        --out data_generation/data/labels/fc2_04_labeled_sonnet46.jsonl \
        --provider bedrock --concurrency 10
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from ..core.labeling import PROMPTS, AsyncLabelerClient, make_async_labeler
from ..core.schema import append_label, load_done_keys


async def _worker(
    name: int,
    queue: asyncio.Queue,
    labeler: AsyncLabelerClient,
    write_lock: asyncio.Lock,
    counters: dict,
    out_path: Path,
    prompt_version: str,
    prompt_hash: str,
    total: int,
) -> None:
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            return
        qid, did, query, units, body, origin = item
        try:
            result = await labeler.label(query, units)
            async with write_lock:
                append_label(
                    out_path,
                    query_id=qid,
                    doc_id=did,
                    query=query,
                    body=body,
                    units=units,
                    relevant_unit_ids=result.relevant_unit_ids,
                    labeler_provider=labeler.provider,
                    labeler_model=labeler.deployment,
                    prompt_version=prompt_version,
                    prompt_hash=prompt_hash,
                    tokens_in=result.tokens_in,
                    tokens_out=result.tokens_out,
                    latency_s=result.latency_s,
                    origin=origin,
                    status=result.status,
                )
                counters["done"] += 1
                counters["tok_in"] += result.tokens_in
                counters["tok_out"] += result.tokens_out
                if result.status == "ok":
                    kept_str = f"kept={len(result.relevant_unit_ids):>2}"
                else:
                    kept_str = f"FAIL={result.status}"
                done = counters["done"]
                if done % 25 == 0 or result.status != "ok":
                    print(
                        f"[{done:>4}/{total}] w{name} {result.status:14s} "
                        f"qid={qid} units={len(units):>3} {kept_str} "
                        f"tok_in={result.tokens_in:>5} lat={result.latency_s:.1f}s",
                        flush=True,
                    )
        except Exception as e:
            print(f"[worker {name}] unhandled error qid={qid} did={did}: {e}", flush=True)
        finally:
            queue.task_done()


async def label_pairs(
    *,
    in_path: Path,
    out_path: Path,
    provider: str = "bedrock",
    prompt_version: str = "v2",
    concurrency: int = 10,
    limit: int = 0,
) -> dict:
    """Run an LLM labeler over (query, units) rows from ``in_path``.

    Resume-safe on (query_id, doc_id) for the same (provider, model,
    prompt_version). Returns a counts dict.
    """
    labeler = make_async_labeler(provider, version=prompt_version)
    prompt = PROMPTS[prompt_version]
    ph = prompt.prompt_hash()

    in_rows = [json.loads(l) for l in in_path.open()]
    # Only labeler rows that actually have units (skip skipped:long, empty bodies).
    in_rows = [r for r in in_rows if r.get("status") == "ok" and r.get("units") and r.get("query")]
    if limit:
        in_rows = in_rows[:limit]

    print(f"provider        : {labeler.provider}")
    print(f"deployment      : {labeler.deployment}")
    print(f"prompt_version  : {prompt.version}  hash={ph}")
    print(f"in              : {in_path}  rows={len(in_rows)}")
    print(f"out             : {out_path}")
    print(f"concurrency     : {concurrency}")

    done = load_done_keys(out_path, labeler.provider, labeler.deployment, prompt.version)
    print(f"already labeled with this (provider, model, prompt_version): {len(done)}")

    pending = [r for r in in_rows if (str(r["query_id"]), r["doc_id"]) not in done]
    print(f"pending: {len(pending)}")
    if not pending:
        print("nothing to do")
        return {"labeled": 0, "tok_in": 0, "tok_out": 0}

    counters = {"done": 0, "tok_in": 0, "tok_out": 0}
    write_lock = asyncio.Lock()
    queue: asyncio.Queue = asyncio.Queue(maxsize=concurrency * 2)
    workers = [
        asyncio.create_task(
            _worker(
                i, queue, labeler, write_lock, counters, out_path, prompt.version, ph, len(pending)
            )
        )
        for i in range(concurrency)
    ]

    t0 = time.perf_counter()
    for r in pending:
        await queue.put(
            (
                r["query_id"],
                r["doc_id"],
                r["query"],
                r["units"],
                r.get("body", ""),
                r.get("origin", "unknown"),
            )
        )
    for _ in workers:
        await queue.put(None)
    await asyncio.gather(*workers)
    dt = time.perf_counter() - t0

    n = counters["done"]
    print(
        f"\nDONE. labeled={n} wall={dt:.1f}s rate={n / max(1, dt) * 60:.1f}/min "
        f"tok_in={counters['tok_in']:,} tok_out={counters['tok_out']:,}"
    )
    return {"labeled": n, "tok_in": counters["tok_in"], "tok_out": counters["tok_out"]}


async def main_async(args) -> None:
    await label_pairs(
        in_path=args.in_path,
        out_path=args.out,
        provider=args.provider,
        prompt_version=args.prompt_version,
        concurrency=args.concurrency,
        limit=args.limit,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument(
        "--provider", choices=["azure", "bedrock", "gemini", "openrouter"], default="bedrock"
    )
    ap.add_argument("--prompt-version", choices=sorted(PROMPTS), default="v2")
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()

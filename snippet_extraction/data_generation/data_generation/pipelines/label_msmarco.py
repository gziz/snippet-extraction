"""Label MS MARCO qrel pairs: sample qrels, fetch doc bodies, labeler, append.

Async runner with a concurrency limit and asyncio.Lock-protected JSONL
appends. Requires the one-time doc index:
``python -m data_generation.core.msmarco.build_doc_index``.

Usage:
    python -m data_generation.pipelines.label_msmarco --n 200 --seed 99 --concurrency 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import pandas as pd

from ..core.labeling import PROMPTS, AsyncLabelerClient, make_async_labeler
from ..core.msmarco.doc_fetch import fetch_docs, load_index
from ..core.paths import INDEXES_DIR, LABELS_DIR, MSMARCO_DATA_DIR
from ..core.schema import append_label, load_done_keys
from ..core.segment import segment

DOCS_TSV = MSMARCO_DATA_DIR / "msmarco-docs.tsv"
QUERIES_TSV = MSMARCO_DATA_DIR / "msmarco-doctrain-queries.tsv"
QRELS_TSV = MSMARCO_DATA_DIR / "msmarco-doctrain-qrels.tsv"

DEFAULT_OUT = LABELS_DIR / "pilot_v2.jsonl"
DOC_INDEX = INDEXES_DIR / "_doc_index.pkl"

PROMPT = PROMPTS["v2"]


def load_qrels_sample(n: int, seed: int) -> pd.DataFrame:
    qrels = pd.read_csv(
        QRELS_TSV,
        sep=r"\s+",
        header=None,
        engine="python",
        names=["qid", "iter", "docid", "rel"],
        dtype={"qid": "int64", "iter": "int32", "docid": "string", "rel": "int32"},
    )
    queries = pd.read_csv(
        QUERIES_TSV,
        sep="\t",
        header=None,
        names=["qid", "query"],
        dtype={"qid": "int64", "query": "string"},
        quoting=3,
        on_bad_lines="skip",
    )
    return (
        qrels.sample(n=n, random_state=seed)
        .merge(queries, on="qid", how="left")
        .reset_index(drop=True)
    )


async def _worker(
    name: int,
    queue: asyncio.Queue,
    labeler: AsyncLabelerClient,
    write_lock: asyncio.Lock,
    counters: dict,
    out_path: Path,
    ph: str,
    origin: str,
    total: int,
) -> None:
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            return
        idx, qid, did, query, units, body = item
        try:
            result = await labeler.label(query, units)

            async with write_lock:
                append_label(
                    out_path,
                    query_id=int(qid),
                    doc_id=did,
                    query=query,
                    body=body,
                    units=units,
                    relevant_unit_ids=result.relevant_unit_ids,
                    labeler_provider=labeler.provider,
                    labeler_model=labeler.deployment,
                    prompt_version=PROMPT.version,
                    prompt_hash=ph,
                    tokens_in=result.tokens_in,
                    tokens_out=result.tokens_out,
                    latency_s=result.latency_s,
                    origin=origin,
                    status=result.status,
                )
                counters["done"] += 1
                counters["tok_in"] += result.tokens_in
                counters["tok_out"] += result.tokens_out
                counters["lat"] += result.latency_s
                done = counters["done"]
                if result.status == "ok":
                    kept_str = f"kept={len(result.relevant_unit_ids):>2}"
                else:
                    kept_str = f"FAIL={result.status}"
                print(
                    f"[{done:>4}/{total}] w{name} {result.status:14s} "
                    f"qid={qid} did={did} units={len(units):>3} {kept_str} "
                    f"tok_in={result.tokens_in:>5} tok_out={result.tokens_out:>3} "
                    f"lat={result.latency_s:.1f}s",
                    flush=True,
                )
        finally:
            queue.task_done()


async def main_async(args) -> None:
    labeler = make_async_labeler(args.provider)
    ph = PROMPT.prompt_hash()

    print(f"provider        : {labeler.provider}")
    print(f"deployment      : {labeler.deployment}")
    print(f"prompt_version  : {PROMPT.version}  hash={ph}")
    print(f"out             : {args.out}")
    print(f"concurrency     : {args.concurrency}")
    if args.exclude:
        print(f"excluding pairs from: {[str(p) for p in args.exclude]}")
    print(f"sampling {args.n} qrels rows (seed={args.seed}, oversample={args.oversample}x)")

    # Build the global "already-labeled anywhere" set, regardless of provider/model.
    exclude_pairs: set[tuple[str, str]] = set()
    for p in args.exclude or []:
        if not p.exists():
            print(f"  exclude file missing, skipping: {p}")
            continue
        with p.open() as f:
            for line in f:
                row = json.loads(line)
                exclude_pairs.add((str(row["query_id"]), row["doc_id"]))
        print(f"  loaded {len(exclude_pairs)} cumulative exclusion pairs after {p.name}")

    # Oversample qrels so that after excluding existing pairs we still have ~n.
    sample = load_qrels_sample(args.n * args.oversample, args.seed)

    origin = "qrel"

    # Drop any pair that has already been labeled (anywhere in --exclude inputs).
    if exclude_pairs:
        before = len(sample)
        sample = sample[
            ~sample.apply(lambda r: (str(int(r["qid"])), r["docid"]) in exclude_pairs, axis=1)
        ].reset_index(drop=True)
        print(f"  excluded {before - len(sample)} pairs already labeled elsewhere")

    # Drop within-sample duplicates (different qrel rows can share a docid).
    before = len(sample)
    sample = sample.drop_duplicates(subset=["qid", "docid"]).reset_index(drop=True)
    if before != len(sample):
        print(f"  deduped {before - len(sample)} within-sample duplicate pairs")

    # Cap to the requested n now that exclusions are applied.
    if len(sample) > args.n:
        sample = sample.iloc[: args.n].reset_index(drop=True)
        print(f"  capped sample to first {args.n} after exclusion")
    elif len(sample) < args.n:
        print(
            f"  warning: only {len(sample)} pairs available after exclusion (wanted {args.n}). Bump --oversample if you need more."
        )

    done = load_done_keys(args.out, labeler.provider, labeler.deployment, PROMPT.version)
    print(f"already labeled with this (provider, model, prompt_version): {len(done)}")

    pending = sample[
        ~sample.apply(lambda r: (str(int(r["qid"])), r["docid"]) in done, axis=1)
    ].reset_index(drop=True)
    if pending.empty:
        print("nothing to do")
        return

    if not DOC_INDEX.exists():
        raise SystemExit(
            f"Doc index not found: {DOC_INDEX}\n"
            "Run once: python -m data_generation.core.msmarco.build_doc_index"
        )
    print(f"loading doc index from {DOC_INDEX.name}...")
    index = load_index(DOC_INDEX)
    print(f"  {len(index):,} docids indexed")

    print(f"fetching {len(pending)} doc bodies (random access)...")
    docs = fetch_docs(DOCS_TSV, set(pending["docid"]), index)
    print(f"  got {len(docs)}/{len(pending)} doc bodies")

    # Pre-segment and pre-filter (skip empty / too-big) so workers do only API work.
    queue: asyncio.Queue = asyncio.Queue()
    n_queued = n_skip = 0
    for idx, row in pending.iterrows():
        did = row["docid"]
        if did not in docs:
            n_skip += 1
            continue
        _, _, body = docs[did]
        units = segment(body)
        if not units or len(units) > 500:
            n_skip += 1
            continue
        await queue.put((idx, int(row["qid"]), did, row["query"], units, body))
        n_queued += 1
    print(f"queued {n_queued}  pre-skipped {n_skip}")
    if n_queued == 0:
        return

    write_lock = asyncio.Lock()
    counters = {"done": 0, "tok_in": 0, "tok_out": 0, "lat": 0.0}
    t0 = time.perf_counter()

    workers = [
        asyncio.create_task(
            _worker(i, queue, labeler, write_lock, counters, args.out, ph, origin, n_queued)
        )
        for i in range(args.concurrency)
    ]
    # Sentinels — one per worker — so each worker exits cleanly when queue is drained.
    for _ in range(args.concurrency):
        await queue.put(None)

    await queue.join()
    for w in workers:
        await w

    wall = time.perf_counter() - t0
    n = counters["done"]
    print()
    print(f"labeled: {n}   skipped: {n_skip}")
    if n:
        print(
            f"tokens : in={counters['tok_in']:,}  out={counters['tok_out']:,}  "
            f"avg_lat={counters['lat'] / n:.1f}s   wall={wall:.1f}s  "
            f"throughput={n / (wall / 60):.1f}/min  tpm_in={counters['tok_in'] / (wall / 60):,.0f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--exclude",
        type=Path,
        nargs="*",
        default=None,
        help="JSONL files whose (query_id, doc_id) pairs to exclude (any provider/model).",
    )
    ap.add_argument(
        "--oversample",
        type=int,
        default=1,
        help="Pre-sample n*oversample qrels rows to absorb --exclude losses (default 1).",
    )
    ap.add_argument(
        "--concurrency",
        type=int,
        default=3,
        help="concurrent in-flight API calls; ~3 for 500k TPM budget",
    )
    ap.add_argument(
        "--provider",
        choices=["azure", "bedrock", "gemini", "openrouter"],
        default="azure",
        help="labeler backend: azure (gpt-5.4), bedrock (claude sonnet 4.6), gemini (3.5 flash), or openrouter (deepseek v4 pro)",
    )
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()

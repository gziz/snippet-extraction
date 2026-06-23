"""Re-label the curated regression set and diff against the previous best run.

Usage:
    # default: Azure / current AZURE_OPENAI_DEPLOYMENT
    python -m data_generation.pipelines.regression

    # Sonnet 4.6 via Bedrock
    python -m data_generation.pipelines.regression --provider bedrock \\
        --baseline data_generation/data/labels/pilot_v2_sonnet46.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from ..core.labeling import PROMPTS, make_async_labeler
from ..core.msmarco.doc_fetch import fetch_docs, load_index
from ..core.paths import INDEXES_DIR, LABELS_DIR, MSMARCO_DATA_DIR, REGRESSION_FILE
from ..core.segment import segment

DOCS_TSV = MSMARCO_DATA_DIR / "msmarco-docs.tsv"
QUERIES_TSV = MSMARCO_DATA_DIR / "msmarco-doctrain-queries.tsv"
DOC_INDEX = INDEXES_DIR / "_doc_index.pkl"

PROMPT = PROMPTS["v2"]


def load_baseline(path: Path) -> dict[tuple[int, str], dict]:
    out: dict[tuple[int, str], dict] = {}
    if not path.exists():
        return out
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            out[(r["query_id"], r["doc_id"])] = r
    return out


async def main_async(args) -> None:
    labeler = make_async_labeler(args.provider)
    ph = PROMPT.prompt_hash()

    print(f"provider        : {labeler.provider}")
    print(f"deployment      : {labeler.deployment}")
    print(f"prompt_version  : {PROMPT.version}  hash={ph}")
    print(f"baseline        : {args.baseline.name}")
    print()

    baseline = load_baseline(args.baseline)

    items: list[dict] = []
    seen: set[tuple[int, str]] = set()
    with REGRESSION_FILE.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            key = (r["query_id"], r["doc_id"])
            if key in seen:
                continue
            seen.add(key)
            items.append(r)

    import pandas as pd

    qdf = pd.read_csv(
        QUERIES_TSV,
        sep="\t",
        header=None,
        names=["qid", "query"],
        dtype={"qid": "int64", "query": "string"},
        quoting=3,
        on_bad_lines="skip",
    )
    qmap = dict(zip(qdf["qid"].astype(int), qdf["query"].astype(str)))

    print("loading doc index...")
    index = load_index(DOC_INDEX)
    needed = {it["doc_id"] for it in items}
    print(f"fetching {len(needed)} doc bodies...")
    docs = fetch_docs(DOCS_TSV, needed, index)
    print(f"  got {len(docs)}/{len(needed)}")
    print()

    n_same = n_diff = 0
    for it in items:
        qid, did = it["query_id"], it["doc_id"]
        query = qmap.get(qid, "<unknown>")
        prior = baseline.get((qid, did))
        prior_kept = prior["relevant_unit_ids"] if prior else None
        if did not in docs:
            print(f"qid={qid} did={did}  SKIP (doc missing)")
            continue
        _, _, body = docs[did]
        units = segment(body)
        res = await labeler.label(query, units)
        match = (prior_kept is not None) and (
            set(res.relevant_unit_ids or []) == set(prior_kept or [])
        )
        if match:
            n_same += 1
        else:
            n_diff += 1
        prev_str = f"prior={prior_kept}" if prior else "prior=<not in baseline>"
        print(
            f"qid={qid:>7} did={did:<10}  status={res.status:14s}  current={res.relevant_unit_ids}  {prev_str}  {'OK' if match else 'DIFF'}"
        )
        print(f"   Q: {query}")
        if not match:
            cur = set(res.relevant_unit_ids or [])
            prv = set(prior_kept or [])
            for idx in sorted(cur - prv):
                snippet = units[idx][:180].replace("\n", " ")
                print(f"   + NEW [{idx}]: {snippet}")
            for idx in sorted(prv - cur):
                if prior and idx < len(prior["units"]):
                    snippet = prior["units"][idx][:180].replace("\n", " ")
                    print(f"   - DROPPED [{idx}]: {snippet}")
        if it.get("note"):
            print(f"   note: {it['note']}")
        print()

    print(f"summary: same={n_same}  different={n_diff}  total={n_same + n_diff}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--baseline",
        type=Path,
        default=LABELS_DIR / "pilot_v2_gpt54.jsonl",
        help="JSONL with previously-known kept_unit_ids per (query_id, doc_id)",
    )
    ap.add_argument(
        "--provider",
        choices=["azure", "bedrock"],
        default="azure",
        help="labeler backend",
    )
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()

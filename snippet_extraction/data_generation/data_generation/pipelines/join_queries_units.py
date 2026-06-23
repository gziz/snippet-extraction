"""Join per-doc synthetic queries to their document units -> labeler input rows.

Step 3 of the labeling pipeline. ``synth_queries_per_doc`` emits query rows
keyed by ``source_doc_id``; this pairs each generated query with the units of
its source document (from the segmented corpus) and writes rows ready for
``label_units`` (status="ok", carrying query + units + body).

Usage:
    python -m data_generation.pipelines.join_queries_units \
        --queries data_generation/data/queries/corpus_synth_q3.jsonl \
        --corpus  data_generation/data/labels/fc2_02_segmented.jsonl \
        --out     data_generation/data/labels/fc2_03_pairs.jsonl
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path


def join_queries_to_units(*, queries_path: Path, corpus_path: Path, out_path: Path) -> dict:
    """Pair each synthetic query with its source document's units.

    Emits rows ready for ``label_units`` (status="ok", query + units + body).
    Returns a counts dict.
    """
    # Index the segmented corpus by doc_id.
    by_doc: dict[str, dict] = {}
    for line in corpus_path.open():
        r = json.loads(line)
        if r.get("units"):
            by_doc[r["doc_id"]] = r

    n_q = n_written = n_no_doc = n_no_units = 0
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    with out_path.open("w") as fout:
        for line in queries_path.open():
            q = json.loads(line)
            n_q += 1
            did = q.get("source_doc_id")
            doc = by_doc.get(did)
            if doc is None:
                n_no_doc += 1
                continue
            units = doc.get("units") or []
            if not units:
                n_no_units += 1
                continue
            row = {
                "query_id": q["qid"],
                "doc_id": did,
                "origin": doc.get("origin", "unknown"),
                "status": "ok",
                "query": q["query"],
                "topic": q.get("topic"),
                "intent": q.get("intent"),
                "url": doc.get("url"),
                "title": doc.get("title"),
                "body": doc.get("body"),
                "units": units,
                "n_units": len(units),
                "relevant_unit_ids": [],
                "source_doc_id": did,
                "source_query_id": q.get("source_query_id"),
                "timestamp": now,
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"queries read     : {n_q}")
    print(f"docs indexed     : {len(by_doc)}")
    print(f"pairs written    : {n_written}")
    print(f"dropped no-doc   : {n_no_doc}")
    print(f"dropped no-units : {n_no_units}")
    print(f"out -> {out_path}")
    return {
        "queries": n_q,
        "docs": len(by_doc),
        "written": n_written,
        "no_doc": n_no_doc,
        "no_units": n_no_units,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--queries",
        required=True,
        type=Path,
        help="synth_queries_per_doc output (qid, query, source_doc_id, intent, ...)",
    )
    ap.add_argument(
        "--corpus", required=True, type=Path, help="segmented corpus (doc_id -> units, body)"
    )
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    join_queries_to_units(queries_path=args.queries, corpus_path=args.corpus, out_path=args.out)


if __name__ == "__main__":
    main()

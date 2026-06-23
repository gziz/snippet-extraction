"""Pretty-print labeled rows for human inspection.

Usage:
    python -m data_generation.pipelines.inspect_labels --in data_generation/data/labels/pilot.jsonl
    python -m data_generation.pipelines.inspect_labels --in <path> --limit 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--unit-trunc", type=int, default=220)
    args = ap.parse_args()

    with args.inp.open() as f:
        rows = [json.loads(line) for line in f]
    if args.limit:
        rows = rows[: args.limit]

    for i, row in enumerate(rows):
        status = row.get("status", "ok")
        kept = set(row.get("relevant_unit_ids") or [])
        print("=" * 100)
        print(
            f"#{i + 1}  qid={row.get('query_id')}  did={row.get('doc_id')}  "
            f"model={row.get('labeler_model')}  prompt={row.get('prompt_version')}/{row.get('prompt_hash')}  "
            f"status={status}"
        )
        print(f"Q: {row.get('query')}")
        print(
            f"units={row.get('n_units')}  kept={len(kept)}  "
            f"tok_in={row.get('tokens_in')}  tok_out={row.get('tokens_out')}  "
            f"lat={row.get('latency_s')}s"
        )
        print("-" * 100)
        if status != "ok":
            print(f"  (skipping unit dump — status={status})")
            print()
            continue
        for j, unit in enumerate(row["units"]):
            mark = "KEEP" if j in kept else "    "
            display = unit if len(unit) <= args.unit_trunc else unit[: args.unit_trunc] + " ..."
            print(f"  {mark}  [{j:>3}] {display}")
        print()


if __name__ == "__main__":
    main()

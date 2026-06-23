"""Merge two A/B arm files into one comparison JSONL for compare_visualizer.html.

Each output row carries the shared (query, doc, units) plus three id sets:
  v2_ids          strict answer from arm A (v2)
  v3_ids          strict answer from arm B (v3)
  v3_context_ids  answer+context from arm B (v3)

Per-unit classification (for coloring) is left to the viewer.

Usage:
    python -m data_generation.pipelines.merge_ab \
        --arm-a data_generation/data/labels/ab_test_v2v3_t0/ab_arm_a_v2.jsonl \
        --arm-b data_generation/data/labels/ab_test_v2v3_t0/ab_arm_b_v3.jsonl \
        --out   data_generation/data/labels/ab_test_v2v3_t0/compare.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> dict[tuple[str, str], dict]:
    out: dict[tuple[str, str], dict] = {}
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            out[(str(r["query_id"]), r["doc_id"])] = r
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-a", type=Path, required=True, help="v2 (strict) arm file")
    ap.add_argument("--arm-b", type=Path, required=True, help="v3 (strict+context) arm file")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    a = _load(args.arm_a)
    b = _load(args.arm_b)
    shared = [k for k in a if k in b]

    n = 0
    with args.out.open("w") as fout:
        for k in shared:
            ra, rb = a[k], b[k]
            row = {
                "query_id": ra["query_id"],
                "doc_id": ra["doc_id"],
                "query": ra["query"],
                "units": ra["units"],
                "n_units": ra.get("n_units", len(ra["units"])),
                "v2_ids": sorted(ra.get("relevant_unit_ids") or []),
                "v3_ids": sorted(rb.get("relevant_unit_ids") or []),
                "v3_context_ids": sorted(rb.get("context_unit_ids") or []),
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1

    print(f"arm_a={len(a)}  arm_b={len(b)}  shared={len(shared)}  written={n}")
    print(f"out -> {args.out}")


if __name__ == "__main__":
    main()

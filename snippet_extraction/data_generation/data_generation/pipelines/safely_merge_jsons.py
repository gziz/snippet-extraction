"""Safely merge JSONL row files into one output file.

One resume-safe, append-style merge that backs two assembly stages:

  * **corpus consolidation** — one row per unique doc, body-bearing only,
    projected to the document fields:
    ``--dedupe-key doc_id --require body --keep <DOC_FIELDS>``
  * **label aggregation** — concatenate label files row-for-row, dropping the
    heavy doc ``body``: ``--drop body``

Both are the same merge with different knobs:

    --dedupe-key FIELD   keep only the first row per value of FIELD
    --require FIELD       drop rows whose FIELD is missing/empty
    --keep FIELDS...      project each kept row to this whitelist of fields
    --drop FIELDS...      remove these fields from each kept row
    --inputs A B ...      source JSONL files (missing files are skipped)
    --out PATH

Usage (corpus):
    python -m data_generation.pipelines.safely_merge_jsons \
        --inputs firecrawl_v1.jsonl tavily_v1.jsonl \
        --out    fc2_01_corpus.jsonl \
        --dedupe-key doc_id --require body \
        --keep doc_id origin query query_id topic intent url title \
               description body position

Usage (aggregate):
    python -m data_generation.pipelines.safely_merge_jsons \
        --inputs msmarco_labels.jsonl fc2_04_labeled_sonnet46.jsonl \
        --out    all_labels_v2.jsonl --drop body
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

# Document fields kept by corpus consolidation (everything that describes the
# document, none of the labeled-pass fields like units / relevant_unit_ids).
DOC_FIELDS = (
    "doc_id",
    "origin",
    "query",
    "query_id",
    "topic",
    "intent",
    "url",
    "title",
    "description",
    "body",
    "position",
)


def merge(
    inputs: list[Path],
    out: Path,
    *,
    dedupe_key: str | None = None,
    require_field: str | None = None,
    keep_fields: tuple[str, ...] | list[str] | None = None,
    drop_fields: tuple[str, ...] | list[str] | None = None,
) -> dict:
    """Concatenate ``inputs`` into ``out`` with optional dedup/projection.

    For duplicate ``dedupe_key`` values the first-seen row wins, so list base
    files before incremental additions. Missing source files are skipped.
    Returns a counts dict.
    """
    seen: set = set()
    origins: Counter = Counter()
    n_in = n_kept = n_no_field = n_dup = 0
    with out.open("w") as fout:
        for src in inputs:
            if not src.exists():
                continue
            for line in src.open():
                line = line.strip()
                if not line:
                    continue
                n_in += 1
                r = json.loads(line)
                if require_field and not r.get(require_field):
                    n_no_field += 1
                    continue
                if dedupe_key is not None:
                    key = r.get(dedupe_key)
                    if key in seen:
                        n_dup += 1
                        continue
                    seen.add(key)
                if keep_fields is not None:
                    r = {k: r.get(k) for k in keep_fields}
                elif drop_fields:
                    for f in drop_fields:
                        r.pop(f, None)
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                origins[r.get("origin", "unknown")] += 1
                n_kept += 1

    print(f"sources      : {', '.join(p.name for p in inputs)}")
    print(f"rows scanned : {n_in}")
    if require_field:
        print(f"dropped no-{require_field}: {n_no_field}")
    if dedupe_key is not None:
        print(f"dropped dup {dedupe_key}: {n_dup}")
    print(f"rows written : {n_kept}")
    for o, c in origins.most_common():
        print(f"  {c:>6}  {o}")
    print(f"out -> {out}")
    return {
        "scanned": n_in,
        "written": n_kept,
        "no_field": n_no_field,
        "dup": n_dup,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        required=True,
        help="source JSONL files, base first (first-seen dedupe wins)",
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--dedupe-key", default=None, help="keep only the first row per value of this field"
    )
    ap.add_argument(
        "--require",
        dest="require_field",
        default=None,
        help="drop rows whose field is missing/empty",
    )
    ap.add_argument(
        "--keep",
        dest="keep_fields",
        nargs="+",
        default=None,
        help="project each kept row to this whitelist of fields",
    )
    ap.add_argument(
        "--drop",
        dest="drop_fields",
        nargs="+",
        default=None,
        help="remove these fields from each kept row",
    )
    args = ap.parse_args()
    merge(
        args.inputs,
        args.out,
        dedupe_key=args.dedupe_key,
        require_field=args.require_field,
        keep_fields=args.keep_fields,
        drop_fields=args.drop_fields,
    )


if __name__ == "__main__":
    main()

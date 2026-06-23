"""Stage 1 of relabeling: re-segment stored document bodies into fresh units.

Reads a label JSONL, runs each row's stored ``body`` through the current
block-aware ``segment()``, and writes a new JSONL whose ``units`` / ``n_units``
reflect the current segmenter. ``relevant_unit_ids`` is cleared because the new
units invalidate any prior judgments.

The output is consumed by ``label_units`` (stage 2), which runs an LLM labeler
over the refreshed units. It is also directly loadable by ``visualizer.html``
to eyeball how the new segmentation splits real documents.

Usage:
    python -m data_generation.pipelines.segment_documents \
        --in  data_generation/data/labels/firecrawl_v1_part2.jsonl \
        --out data_generation/data/labels/firecrawl_v1_part2_resegmented.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..core.paths import LABELS_DIR
from ..core.segment import segment


def segment_corpus(*, in_path: Path, out_path: Path) -> dict:
    """Re-segment stored document bodies into fresh units.

    Stage between corpus consolidation and labeling. Returns a counts dict.
    """
    n_in = n_out = n_no_body = 0
    with in_path.open() as fin, out_path.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            row = json.loads(line)
            body = row.get("body")
            if not body:
                n_no_body += 1
                continue
            units = segment(body)
            row["units"] = units
            row["n_units"] = len(units)
            row["relevant_unit_ids"] = []
            row["n_kept"] = 0
            # "ok" = body segmented cleanly and is ready for stage 2 (label_units).
            # The visualizer conveys "un-labeled" via 0 relevant units, so this
            # status doesn't mislead there either.
            row["status"] = "ok"
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_out += 1

    print(f"in={n_in}  written={n_out}  skipped_no_body={n_no_body}")
    print(f"out -> {out_path}")
    return {"in": n_in, "written": n_out, "skipped_no_body": n_no_body}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in", dest="in_path", type=Path, default=LABELS_DIR / "firecrawl_v1_part2.jsonl"
    )
    ap.add_argument(
        "--out",
        dest="out_path",
        type=Path,
        default=LABELS_DIR / "firecrawl_v1_part2_resegmented.jsonl",
    )
    args = ap.parse_args()
    segment_corpus(in_path=args.in_path, out_path=args.out_path)


if __name__ == "__main__":
    main()

"""One-time build of docid -> byte_offset index over msmarco-docs.tsv.

Run once:
    python -m data_generation.core.msmarco.build_doc_index

The output (`data/indexes/_doc_index.pkl`, ~150MB) lets subsequent label runs
fetch any doc body in microseconds instead of streaming 22GB.
"""

from __future__ import annotations

import pickle
import time

from ..paths import INDEXES_DIR, MSMARCO_DATA_DIR

DOCS_TSV = MSMARCO_DATA_DIR / "msmarco-docs.tsv"
INDEX_PATH = INDEXES_DIR / "_doc_index.pkl"


def main() -> None:
    if INDEX_PATH.exists():
        print(f"index already exists: {INDEX_PATH}  ({INDEX_PATH.stat().st_size / 1e6:.1f} MB)")
        print("delete the file to rebuild")
        return

    print(f"building index over {DOCS_TSV}")
    print(f"  file size : {DOCS_TSV.stat().st_size / 1e9:.2f} GB")
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

    index: dict[str, int] = {}
    t0 = time.perf_counter()
    with DOCS_TSV.open("rb") as f:
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            # docid is the first tab-separated field; decode just the prefix
            tab_pos = line.find(b"\t")
            if tab_pos <= 0:
                continue
            docid = line[:tab_pos].decode("ascii")
            index[docid] = offset
            if len(index) % 500_000 == 0:
                elapsed = time.perf_counter() - t0
                print(f"  indexed {len(index):,} docs  ({elapsed:.1f}s)")

    elapsed = time.perf_counter() - t0
    print(f"  indexed {len(index):,} docs total  ({elapsed:.1f}s)")

    print(f"writing {INDEX_PATH}")
    with INDEX_PATH.open("wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  wrote {INDEX_PATH.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()

"""One-time build of qid -> [top100 docids] index from the BM25 run file.

The TREC run file (msmarco-doctrain-top100.gz) is ~400MB compressed.
We stream it once and store a compact pickle (~150MB) mapping
qid -> list of 100 docids in BM25 rank order.

Run once after `download_bm25`:
    python -m data_generation.core.msmarco.build_bm25_index
"""

from __future__ import annotations

import gzip
import pickle
import time
from collections import defaultdict

from ..paths import INDEXES_DIR, MSMARCO_DATA_DIR

RUN_GZ = MSMARCO_DATA_DIR / "msmarco-doctrain-top100.gz"
INDEX_PATH = INDEXES_DIR / "_bm25_top100.pkl"


def main() -> None:
    if INDEX_PATH.exists():
        print(f"index already exists: {INDEX_PATH}  ({INDEX_PATH.stat().st_size / 1e6:.1f} MB)")
        print("delete the file to rebuild")
        return
    if not RUN_GZ.exists():
        raise SystemExit(
            f"Missing {RUN_GZ}. Run: python -m data_generation.core.msmarco.download_bm25"
        )

    print(f"streaming {RUN_GZ.name}")
    top: dict[int, list[str]] = defaultdict(list)
    t0 = time.perf_counter()
    n_lines = 0
    with gzip.open(RUN_GZ, "rt", encoding="ascii") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 6:
                continue
            qid = int(parts[0])
            docid = parts[2]
            top[qid].append(docid)
            n_lines += 1
            if n_lines % 5_000_000 == 0:
                print(f"  parsed {n_lines:,} lines  ({time.perf_counter() - t0:.1f}s)")

    print(f"  parsed {n_lines:,} lines total  ({time.perf_counter() - t0:.1f}s)")
    print(f"  unique queries: {len(top):,}")

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"writing {INDEX_PATH}")
    with INDEX_PATH.open("wb") as f:
        pickle.dump(dict(top), f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  wrote {INDEX_PATH.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()

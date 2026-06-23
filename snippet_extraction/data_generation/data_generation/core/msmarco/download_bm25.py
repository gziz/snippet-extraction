"""Download MS MARCO's pre-computed BM25 top-100 for the train queries.

The file (~403 MB gz, ~1.5 GB decompressed) is the canonical Anserini
BM25 run published alongside the dataset. We use it as our hard-negative
source: for each (q, d_pos) pair, sample a d_neg from the top-100 that
is not in the qrels positives for q.

Run once:
    python -m data_generation.core.msmarco.download_bm25
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

from ..paths import MSMARCO_DATA_DIR as DATA_DIR

URL = "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-doctrain-top100.gz"
DEST = DATA_DIR / "msmarco-doctrain-top100.gz"


def download(url: str, dest: Path, chunk: int = 1 << 20) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] {dest.name} already exists ({dest.stat().st_size / 1e6:.1f} MB)")
        return
    print(f"[get ] {dest.name} <- {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r, dest.open("wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        n = 0
        while True:
            buf = r.read(chunk)
            if not buf:
                break
            f.write(buf)
            n += len(buf)
            if total:
                pct = n * 100 / total
                print(
                    f"\r       {n / 1e6:8.1f} / {total / 1e6:.1f} MB ({pct:5.1f}%)",
                    end="",
                    flush=True,
                )
        print()


if __name__ == "__main__":
    download(URL, DEST)

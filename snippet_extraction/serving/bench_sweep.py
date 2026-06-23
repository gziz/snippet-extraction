"""Drive the throughput-vs-batch-size sweep against the bench endpoint.

Fires a fixed burst of large (multi-window) docs at the bench endpoint for each
``max_batch_tokens`` value, and reports, per value:
  - end-to-end wall time for the whole burst (throughput proxy)
  - the server-reported GPU batch time and peak CUDA memory (MiB)

Run AFTER ``modal deploy modal_bench.py``. Does not touch production.
"""

from __future__ import annotations

import asyncio
import os
import statistics
import sys
import time

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

# Reuse the eval's fetcher so the bench docs match real eval inputs.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "search_evals"))
from tools.snippets_compressor import compress_doc_remote, fetch_parallel_extract  # noqa: E402

BENCH_EP = "https://rev-71693--query-aware-compressor-bench-compress.modal.run"

URLS = [
    "https://en.wikipedia.org/wiki/Amal_Clooney",
    "https://en.wikipedia.org/wiki/Eiffel_Tower",
    "https://en.wikipedia.org/wiki/Photosynthesis",
    "https://en.wikipedia.org/wiki/Mona_Lisa",
    "https://en.wikipedia.org/wiki/Water",
    "https://en.wikipedia.org/wiki/Great_Wall_of_China",
    "https://en.wikipedia.org/wiki/Albert_Einstein",
    "https://en.wikipedia.org/wiki/Python_(programming_language)",
]

SWEEP = [16384]  # batch size barely moves the needle (sweep showed flat); isolate the GPU
BURST = 15  # mimic 3 parallel search_web x 5 docs


async def _call(md: str, mbt: int, report_gpu: bool) -> dict:
    import httpx

    payload = {
        "query": "general question about the topic",
        "document": md,
        "budget_tokens": 400,
        "min_score": 0.1,
        "max_batch_tokens": mbt,
        "report_gpu": report_gpu,
    }
    async with httpx.AsyncClient(timeout=300.0, follow_redirects=True) as client:
        r = await client.post(BENCH_EP, json=payload)
        r.raise_for_status()
        return r.json()


async def main() -> None:
    print("fetching bench docs...")
    docs = await fetch_parallel_extract(
        URLS, api_key=os.environ.get("PARALLEL_API_KEY"), cache=None
    )
    mds = [docs[u] for u in URLS if docs.get(u)]
    print(f"  {len(mds)} docs, sizes(chars): {sorted(len(m) for m in mds)}")

    # Warm the pinned container once.
    await _call(mds[0], 16384, False)

    print(
        f"\n{'max_batch_tokens':>16} | {'burst_wall_s':>12} | {'gpu_batch_s(med)':>16} | {'peak_MiB':>9}"
    )
    print("-" * 64)
    for mbt in SWEEP:
        burst = [mds[i % len(mds)] for i in range(BURST)]
        t0 = time.perf_counter()
        results = await asyncio.gather(*(_call(md, mbt, True) for md in burst))
        wall = time.perf_counter() - t0
        gpu_times = [r.get("gpu_batch_s", 0.0) for r in results]
        peak = max(r.get("gpu_peak_mib", 0.0) for r in results)
        print(f"{mbt:>16} | {wall:>12.2f} | {statistics.median(gpu_times):>16.2f} | {peak:>9.0f}")


if __name__ == "__main__":
    asyncio.run(main())

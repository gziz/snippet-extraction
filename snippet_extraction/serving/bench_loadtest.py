"""Drive a deployed compress endpoint and report latency under realistic load.

Measures the three things that matter for the burst-tail-under-scale-to-zero
goal:

  1. cold start   : the very first request after the app has 0 warm containers
                    (deploy fresh, or wait out scaledown_window first).
  2. warm single  : steady-state per-request latency once a container is hot.
  3. burst        : N concurrent requests (the search fan-out: several agents x
                    5 docs/query, each a long multi-window doc) -> p50/p95/p99.
  4. sustained    : a stream of requests at fixed concurrency.

Usage:
    python bench_loadtest.py <ENDPOINT_URL> [--burst 15] [--sustained 40 --conc 10]

Docs are synthetic but sized by window count (a window ~= 8k encoder tokens, the
real cost unit) so the GPU sees the same shape as production without shipping
megabytes over HTTP each call.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time

import httpx

# A pool of distinct sentences so the segmenter yields many real units and
# nothing gets deduped. ~ 18-22 tokens each.
_LEX = [
    "The regulatory framework for traditional medicine varies considerably across member states and regions.",
    "Clinical evidence supporting complementary therapies remains uneven and is frequently debated by researchers.",
    "Public health authorities recommend integrating safety monitoring into national pharmacovigilance systems.",
    "Economic analyses suggest that preventive interventions reduce long term hospital admission costs substantially.",
    "The committee reviewed annual statistics covering mortality, morbidity, and access to essential services.",
    "Implementation guidelines emphasize training, accreditation, and continuous professional development for practitioners.",
    "Surveillance data indicate seasonal variation in reported cases across both urban and rural districts.",
    "Funding allocations were adjusted to prioritize underserved populations and remote community clinics.",
    "Quality assurance protocols mandate periodic auditing of supply chains and storage conditions.",
    "Stakeholders called for harmonized reporting standards to enable cross border comparison of outcomes.",
]

# The relevant sentence (answers the query) placed into each doc so the model
# keeps something — exercises selection + render, not just scoring.
_RELEVANT = (
    "The capital of France is Paris, and the World Health Organization "
    "coordinates international public health from Geneva."
)
QUERY = "what does the WHO recommend for regulating traditional medicine"


def make_doc(n_sentences: int) -> str:
    """A markdown doc of ~n_sentences distinct sentences (~18-22 tokens each).

    ~400 sentences ≈ one 8k-token encoder window.
    """
    out = []
    for i in range(n_sentences):
        s = _LEX[i % len(_LEX)]
        # vary so units aren't identical (segmenter + dedup safe)
        out.append(f"[{i}] {s}")
        if i % 37 == 0:
            out.append(_RELEVANT)
    return "\n\n".join(out)


# Doc size mix for a burst: roughly models "5 docs/query" where some extracts
# are short snippets and some are long multi-window pages.
#   ~120 sent  ≈ 0.3 window   (short extract)
#   ~420 sent  ≈ 1   window
#   ~1500 sent ≈ ~4 windows   (long page — the stated pain point)
SIZES = {"short": 120, "medium": 420, "long": 1500}


async def _one(client: httpx.AsyncClient, url: str, doc: str) -> tuple[float, int]:
    t0 = time.perf_counter()
    r = await client.post(url, json={"query": QUERY, "document": doc})
    dt = time.perf_counter() - t0
    r.raise_for_status()
    j = r.json()
    return dt, len(j.get("kept_indices", []))


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1)))))
    return xs[k]


def _report(name: str, lats: list[float]) -> None:
    print(
        f"  {name:<26} n={len(lats):>3}  "
        f"p50={_pct(lats, 50):6.2f}s  p95={_pct(lats, 95):6.2f}s  "
        f"p99={_pct(lats, 99):6.2f}s  max={max(lats):6.2f}s  "
        f"mean={statistics.mean(lats):6.2f}s",
        flush=True,
    )


async def main(url: str, burst: int, sustained: int, conc: int) -> None:
    limits = httpx.Limits(max_connections=200, max_keepalive_connections=200)
    timeout = httpx.Timeout(180.0)
    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        short = make_doc(SIZES["short"])
        medium = make_doc(SIZES["medium"])
        long = make_doc(SIZES["long"])

        print(f"\n=== {url} ===", flush=True)

        # 1. COLD START — single request, app at 0 containers.
        dt, kept = await _one(client, url, medium)
        print(f"  cold start (1 medium doc): {dt:6.2f}s  (kept {kept})", flush=True)

        # 2. WARM SINGLE — 5 sequential mediums.
        warm = []
        for _ in range(5):
            dt, _ = await _one(client, url, medium)
            warm.append(dt)
        _report("warm single (medium)", warm)

        # 3. BURST — `burst` concurrent requests, mixed sizes (fan-out shape).
        docs = []
        for i in range(burst):
            docs.append([short, medium, long][i % 3])
        t0 = time.perf_counter()
        res = await asyncio.gather(*(_one(client, url, d) for d in docs))
        wall = time.perf_counter() - t0
        lats = [d for d, _ in res]
        print(f"\n  BURST of {burst} (mixed short/medium/long), wall={wall:.2f}s", flush=True)
        _report("burst all", lats)

        # 4. SUSTAINED — `sustained` requests at concurrency `conc`.
        if sustained > 0:
            sem = asyncio.Semaphore(conc)

            async def _gated(d):
                async with sem:
                    return await _one(client, url, d)

            sdocs = [[short, medium, long][i % 3] for i in range(sustained)]
            t0 = time.perf_counter()
            sres = await asyncio.gather(*(_gated(d) for d in sdocs))
            swall = time.perf_counter() - t0
            slats = [d for d, _ in sres]
            print(
                f"\n  SUSTAINED {sustained} reqs @ conc {conc}, wall={swall:.2f}s "
                f"({sustained / swall:.2f} req/s)",
                flush=True,
            )
            _report("sustained all", slats)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--burst", type=int, default=15)
    ap.add_argument("--sustained", type=int, default=40)
    ap.add_argument("--conc", type=int, default=10)
    a = ap.parse_args()
    asyncio.run(main(a.url, a.burst, a.sustained, a.conc))

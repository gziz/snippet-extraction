"""One-off hardware A/B: T4 vs CPU for the compressor inference path.

Runs the *exact* production pipeline (snippets_common.segment -> compress_long)
on documents of several lengths, on both a T4 GPU container and a CPU-only
container, and reports cold-start (weight load) time plus warm per-document
latency. The local entrypoint turns those latencies into per-request cost using
Modal's published per-second rates so T4 and CPU can be compared on $/request.

    modal run bench_hw.py

Nothing is deployed; both functions are ephemeral and scale straight back to 0.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import modal

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "runtime"))
sys.path.insert(0, str(_REPO / "common"))

CHECKPOINT_NAME = "run9"
VOLUME_MOUNT = "/checkpoints"
BASE_MODEL = "answerdotai/ModernBERT-base"

# CPU container shape under test (also used for the cost math below).
CPU_CORES = 8.0
CPU_MEM_GIB = 8

# A representative paragraph (real prose) repeated to hit target lengths.
_PARA = (
    "The IEEE Medal of Honor is the highest IEEE award, established in 1917 and "
    "awarded for an exceptional contribution or an extraordinary career in the "
    "IEEE fields of interest. The candidate need not be a member of the IEEE. "
    "The medal is sponsored by the IEEE Foundation and recognizes pioneering "
    "work across electrical engineering, communications, and computing. Past "
    "recipients include researchers whose discoveries shaped radio, information "
    "theory, semiconductors, and modern networking. Each citation summarizes the "
    "contribution that distinguished the recipient from their contemporaries. "
)

# Build documents at increasing lengths (~paragraphs); long ones span several
# encoder windows, which is the regime where GPU vs CPU diverges most.
DOCS = {
    "short_1x": _PARA * 1,
    "medium_8x": _PARA * 8,
    "long_32x": _PARA * 32,
    "xlong_96x": _PARA * 96,
}
QUERY = "Who received the IEEE Medal of Honor and for what contribution?"


def _prefetch_base_assets() -> None:
    from transformers import AutoModel, AutoTokenizer

    AutoModel.from_pretrained(BASE_MODEL)
    AutoTokenizer.from_pretrained(BASE_MODEL)


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4",
        "transformers>=4.45",
        "syntok>=1.4",
        "markdown-it-py>=3.0",
        "tiktoken>=0.7",
    )
    .run_function(_prefetch_base_assets)
    .add_local_python_source("snippets_runtime")
    .add_local_python_source("snippets_common")
)

volume = modal.Volume.from_name("compressor-checkpoints", create_if_missing=True)
app = modal.App("compressor-hw-bench")


def _bench(device: str, *, warmup: int = 2, iters: int = 6) -> dict:
    """Load weights (timed) then time warm runs of the full pipeline per doc."""
    import torch
    from snippets_common.segment import segment
    from snippets_runtime.inference import compress_long, load_for_inference

    if device == "cpu":
        torch.set_num_threads(int(CPU_CORES))

    t0 = time.perf_counter()
    model, tok = load_for_inference(
        f"{VOLUME_MOUNT}/{CHECKPOINT_NAME}", base=BASE_MODEL, device=device
    )
    cold_load_s = time.perf_counter() - t0

    # Pre-segment so segmentation cost (CPU, identical on both) is measured
    # inside the loop exactly as production does it.
    out: dict = {"device": device, "cold_load_s": round(cold_load_s, 3), "docs": {}}
    for name, text in DOCS.items():
        units = segment(text)
        for _ in range(warmup):
            compress_long(model, tok, QUERY, units)
        ts = []
        for _ in range(iters):
            s = time.perf_counter()
            compress_long(model, tok, QUERY, units)
            ts.append(time.perf_counter() - s)
        ts.sort()
        out["docs"][name] = {
            "n_units": len(units),
            "median_s": round(ts[len(ts) // 2], 4),
            "min_s": round(ts[0], 4),
        }
    return out


@app.function(image=image, gpu="T4", volumes={VOLUME_MOUNT: volume}, timeout=600)
def bench_t4() -> dict:
    return _bench("cuda")


@app.function(
    image=image,
    cpu=CPU_CORES,
    memory=CPU_MEM_GIB * 1024,
    volumes={VOLUME_MOUNT: volume},
    timeout=900,
)
def bench_cpu() -> dict:
    return _bench("cpu")


# Modal per-second rates (USD), fetched from modal.com/pricing.
RATE_T4 = 0.000164
RATE_CPU_CORE = 0.0000131
RATE_MEM_GIB = 0.00000222
# Small fixed CPU/mem allotment that rides along with a GPU container.
GPU_SIDE_CORES = 0.125
GPU_SIDE_MEM_GIB = 1.5


def _cost_per_req(seconds: float, *, kind: str) -> float:
    if kind == "t4":
        return seconds * (
            RATE_T4 + GPU_SIDE_CORES * RATE_CPU_CORE + GPU_SIDE_MEM_GIB * RATE_MEM_GIB
        )
    return seconds * (CPU_CORES * RATE_CPU_CORE + CPU_MEM_GIB * RATE_MEM_GIB)


@app.local_entrypoint()
def main():
    import json

    t4 = bench_t4.remote()
    cpu = bench_cpu.remote()

    print("\n=== raw ===")
    print(json.dumps({"t4": t4, "cpu": cpu}, indent=2))

    print(f"\ncold weight-load:  T4 {t4['cold_load_s']}s   CPU {cpu['cold_load_s']}s")
    print(
        f"\n{'doc':<12}{'units':>6}{'T4 med(s)':>12}{'CPU med(s)':>12}"
        f"{'CPU/T4':>9}{'$/req T4':>12}{'$/req CPU':>12}{'cheaper':>9}"
    )
    for name in DOCS:
        td = t4["docs"][name]
        cd = cpu["docs"][name]
        t4s, cps = td["median_s"], cd["median_s"]
        c_t4 = _cost_per_req(t4s, kind="t4")
        c_cpu = _cost_per_req(cps, kind="cpu")
        cheaper = "CPU" if c_cpu < c_t4 else "T4"
        print(
            f"{name:<12}{td['n_units']:>6}{t4s:>12.4f}{cps:>12.4f}"
            f"{cps / t4s:>9.1f}{c_t4:>12.2e}{c_cpu:>12.2e}{cheaper:>9}"
        )

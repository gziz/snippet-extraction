"""Benchmark: how does ``max_batch_tokens`` affect compress latency on a real
long-tail document?

Runs entirely on a single *warm* L4 container (model loaded once in
``@modal.enter``), so the sweep isolates GPU batching — no cold-start, no
network. For each cap value we score the same real PDF (pulled from the eval's
scrape cache and mounted into the image) and report wall time + peak GPU memory.

    cd snippet_extraction/serving
    modal run bench_batch_tokens.py

The "before" point is 16384 (the current default in compress_long_batch); the
rest show the headroom the 7.4 GiB peak / 24 GiB L4 implies.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import modal

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "runtime"))
sys.path.insert(0, str(_REPO / "common"))

APP_NAME = "query-aware-compressor-bench"
CHECKPOINT_NAME = "run9"
VOLUME_MOUNT = "/checkpoints"
BASE_MODEL = "answerdotai/ModernBERT-base"
# Local path to the benchmark docs (only used at image-build time, which runs
# locally). Inside the container the module is re-imported from /root, where
# parents[2] doesn't exist — fall back to the mounted remote path there.
try:
    BENCH_DOCS = Path(__file__).resolve().parents[2] / "search_evals" / "sandbox" / "bench_docs"
except IndexError:
    BENCH_DOCS = Path("/bench_docs")

# Cap values to sweep. 16384 is today's default (~2 full windows/pass).
# Capped well below the point where a single padded forward pass OOMs the L4
# (ModernBERT runs fp32 + SDPA here, so attention memory grows ~rows*seq_len^2,
# not linearly in the token budget). The sweep finds the real safe ceiling.
SWEEP = [16384, 24576, 32768, 40960, 49152, 65536]

# (name, query) per benchmark document mounted at /bench_docs.
DOCS = [
    ("who_142k", "traditional and complementary medicine WHO member states regulation"),
    ("un_289k", "world health report summary statistics"),
]


def _prefetch_base_assets() -> None:
    from transformers import AutoModel, AutoTokenizer

    AutoModel.from_pretrained(BASE_MODEL)
    AutoTokenizer.from_pretrained(BASE_MODEL)


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4",
        "transformers>=4.45",
        "fastapi[standard]",
        "syntok>=1.4",
        "markdown-it-py>=3.0",
        "tiktoken>=0.7",
    )
    .run_function(_prefetch_base_assets)
    .add_local_python_source("snippets_runtime")
    .add_local_python_source("snippets_common")
    .add_local_dir(str(BENCH_DOCS), remote_path="/bench_docs")
)

volume = modal.Volume.from_name("compressor-checkpoints", create_if_missing=True)
app = modal.App(APP_NAME)


@app.cls(
    image=image,
    gpu="L4",
    volumes={VOLUME_MOUNT: volume},
    scaledown_window=120,
    min_containers=0,
    timeout=900,
)
class Bench:
    @modal.enter()
    def load(self):
        import tiktoken
        from snippets_runtime.inference import load_for_inference

        ckpt_dir = f"{VOLUME_MOUNT}/{CHECKPOINT_NAME}"
        self.model, self.tokenizer = load_for_inference(ckpt_dir, base=BASE_MODEL, device="cuda")
        self._enc = tiktoken.get_encoding("cl100k_base")

    @modal.method()
    def run(self, sweep: list[int], docs: list[tuple[str, str]]) -> list[dict]:
        import torch
        from snippets_common.segment import segment
        from snippets_runtime.inference import DocRequest, compress_long_batch

        # Pre-segment each doc once (segmentation cost is constant across the
        # sweep; we're benchmarking the GPU scoring, not the CPU segmenter).
        prepared = []
        for name, query in docs:
            md = Path(f"/bench_docs/{name}.md").read_text()
            t0 = time.perf_counter()
            units = segment(md)
            seg_s = time.perf_counter() - t0
            prepared.append((name, query, units, seg_s, len(md)))

        results: list[dict] = []
        for name, query, units, seg_s, n_chars in prepared:
            print(f"[bench] {name}: {len(units)} units, warming up...", flush=True)
            # Warm the GPU once for this doc so the first timed run isn't paying
            # cuDNN autotune / allocator warmup.
            req = DocRequest(query=query, document=units)
            compress_long_batch(self.model, self.tokenizer, [req], device="cuda")
            torch.cuda.synchronize()

            for cap in sweep:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                try:
                    out = compress_long_batch(
                        self.model,
                        self.tokenizer,
                        [req],
                        device="cuda",
                        max_batch_tokens=cap,
                    )
                    torch.cuda.synchronize()
                    wall = time.perf_counter() - t0
                    peak_gib = torch.cuda.max_memory_allocated() / 2**30
                    n_kept = len(out[0].kept_indices)
                    oom = False
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    wall = float("nan")
                    peak_gib = float("nan")
                    n_kept = -1
                    oom = True
                print(
                    f"[bench] {name} cap={cap}: "
                    f"{'OOM' if oom else f'{wall:.2f}s peak={peak_gib:.2f}GiB'}",
                    flush=True,
                )
                results.append(
                    {
                        "doc": name,
                        "cap": cap,
                        "n_units": len(units),
                        "n_chars": n_chars,
                        "seg_s": seg_s,
                        "score_s": wall,
                        "peak_gib": peak_gib,
                        "n_kept": n_kept,
                        "oom": oom,
                    }
                )
        return results


@app.local_entrypoint()
def main():
    rows = Bench().run.remote(SWEEP, DOCS)

    by_doc: dict[str, list[dict]] = {}
    for r in rows:
        by_doc.setdefault(r["doc"], []).append(r)

    for doc, rs in by_doc.items():
        rs.sort(key=lambda r: r["cap"])
        base = next(r for r in rs if r["cap"] == 16384)["score_s"]
        n_units = rs[0]["n_units"]
        print(
            f"\n=== {doc}  ({n_units:,} units, {rs[0]['n_chars']:,} chars, "
            f"segment={rs[0]['seg_s']:.1f}s) ==="
        )
        print(f"  {'cap':>8}  {'score_s':>8}  {'speedup':>8}  {'peak_GiB':>9}  {'n_kept':>6}")
        for r in rs:
            if r.get("oom"):
                print(f"  {r['cap']:>8}  {'OOM':>8}  {'--':>8}  {'OOM':>9}  {'--':>6}")
                continue
            speedup = base / r["score_s"] if r["score_s"] == r["score_s"] else float("nan")
            print(
                f"  {r['cap']:>8}  {r['score_s']:>8.2f}  "
                f"{speedup:>7.2f}x  {r['peak_gib']:>9.2f}  {r['n_kept']:>6}"
            )

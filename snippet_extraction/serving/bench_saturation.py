"""Is one 8192-token window saturating the L40S? Sweep batch size, watch tok/s.

Runs the real checkpoint's forward on B identical full 8192-token windows for
B in 1,2,4,8,16,... measuring ms, tokens/sec, and peak GPU memory. If tok/s is
flat across B -> the GPU is already saturated at B=1 (batching can't help). If
tok/s climbs with B -> a single window leaves the GPU underutilized and a bigger
max_batch_tokens reclaims real throughput.

    modal run bench_saturation.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import modal

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "runtime"))
sys.path.insert(0, str(_REPO / "common"))

APP_NAME = "query-aware-compressor-bench-sat"
CHECKPOINT_NAME = "run9"
VOLUME_MOUNT = "/checkpoints"
BASE_MODEL = "answerdotai/ModernBERT-base"
GPU = "L40S"


def _prefetch():
    from transformers import AutoModel, AutoTokenizer

    AutoModel.from_pretrained(BASE_MODEL)
    AutoTokenizer.from_pretrained(BASE_MODEL)


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.4", "transformers>=4.45", "numpy>=1.26")
    .run_function(_prefetch)
    .add_local_python_source("snippets_runtime")
    .add_local_python_source("snippets_common")
)
volume = modal.Volume.from_name("compressor-checkpoints", create_if_missing=True)
app = modal.App(APP_NAME)


@app.cls(image=image, gpu=GPU, volumes={VOLUME_MOUNT: volume}, scaledown_window=60, timeout=1200)
class Sat:
    @modal.enter()
    def load(self):
        from snippets_runtime.inference import load_for_inference

        self.model, self.tokenizer = load_for_inference(
            f"{VOLUME_MOUNT}/{CHECKPOINT_NAME}", base=BASE_MODEL, device="cuda"
        )

    @modal.method()
    def run(self, batches: list[int]) -> list[dict]:
        import torch

        tok, model = self.tokenizer, self.model
        # one full 8192-token window
        enc = tok(
            "what does the WHO recommend",
            "regulatory framework text " * 1400,
            truncation="only_second",
            max_length=8192,
            padding="max_length",
            return_tensors="pt",
        )
        ids = enc["input_ids"].to("cuda")
        mask = enc["attention_mask"].to("cuda")
        T = ids.shape[1]

        def timed(B, n=8):
            iid = ids.repeat(B, 1)
            am = mask.repeat(B, 1)
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for _ in range(3):
                    model(input_ids=iid, attention_mask=am)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(n):
                    model(input_ids=iid, attention_mask=am)
                torch.cuda.synchronize()
                dt = (time.perf_counter() - t0) / n
            peak = torch.cuda.max_memory_allocated() / (1024**3)
            return dt, peak

        out = []
        for B in batches:
            try:
                dt, peak = timed(B)
                toks = B * T
                row = {
                    "B": B,
                    "ms": round(dt * 1000, 1),
                    "tok_s": round(toks / dt),
                    "ms_per_window": round(dt * 1000 / B, 1),
                    "peak_gib": round(peak, 2),
                }
                print(
                    f"[sat] B={B:>2}  {row['ms']:8.1f} ms  {row['tok_s']:>9,} tok/s  "
                    f"{row['ms_per_window']:6.1f} ms/window  peak {row['peak_gib']:.1f} GiB",
                    flush=True,
                )
                out.append(row)
            except torch.cuda.OutOfMemoryError:
                print(f"[sat] B={B:>2}  OOM", flush=True)
                torch.cuda.empty_cache()
                out.append({"B": B, "oom": True})
        return out


@app.local_entrypoint()
def main():
    rows = Sat().run.remote([1, 2, 4, 8, 16])
    print("\n==== L40S batch-saturation (full 8192-token windows, bf16) ====")
    base = next((r for r in rows if r.get("B") == 1 and "tok_s" in r), None)
    for r in rows:
        if r.get("oom"):
            print(f"  B={r['B']:>2}  OOM")
            continue
        rel = f"{r['tok_s'] / base['tok_s']:.2f}x vs B=1" if base else ""
        print(
            f"  B={r['B']:>2}  {r['tok_s']:>9,} tok/s  {r['ms_per_window']:6.1f} ms/window  "
            f"peak {r['peak_gib']:>5.1f} GiB   {rel}"
        )
    if base:
        best = max((r for r in rows if "tok_s" in r), key=lambda r: r["tok_s"])
        gain = best["tok_s"] / base["tok_s"]
        print(f"\n  verdict: best throughput at B={best['B']} = {gain:.2f}x of single-window.")
        print("  ~1.0x  -> saturated at B=1 (batching won't help latency)")
        print("  >1.3x  -> real headroom; a bigger max_batch_tokens reclaims it")

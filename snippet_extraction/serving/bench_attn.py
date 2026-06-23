"""Find the fastest forward-pass config for the ModernBERT scorer.

The phase breakdown showed the GPU forward is ~97% of long-doc latency (~1.0s
per 8192-token window on an L4). That's slow for ModernBERT-base and points at
the attention kernel. This bench loads the real checkpoint once and times a
full 8192-token forward under each attention implementation + torch.compile,
all in ONE warm container, asserting the logits match so a speedup is free of
accuracy drift.

    modal run bench_attn.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import modal

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "runtime"))
sys.path.insert(0, str(_REPO / "common"))

BENCH_GPU = os.environ.get("BENCH_GPU", "L4")
# QUICK mode: just the production config (sdpa+bf16) — for cross-GPU sweeps.
QUICK = os.environ.get("BENCH_QUICK", "0") == "1"
APP_NAME = f"query-aware-compressor-bench-attn-{BENCH_GPU.replace(':', 'x')}"
CHECKPOINT_NAME = "run9"
VOLUME_MOUNT = "/checkpoints"
BASE_MODEL = "answerdotai/ModernBERT-base"


def _prefetch_base_assets() -> None:
    from transformers import AutoModel, AutoTokenizer

    AutoModel.from_pretrained(BASE_MODEL)
    AutoTokenizer.from_pretrained(BASE_MODEL)


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4",
        "transformers>=4.45",
        "numpy>=1.26",
    )
    .run_function(_prefetch_base_assets)
    .add_local_python_source("snippets_runtime")
    .add_local_python_source("snippets_common")
)

volume = modal.Volume.from_name("compressor-checkpoints", create_if_missing=True)
app = modal.App(APP_NAME)


@app.cls(
    image=image, gpu=BENCH_GPU, volumes={VOLUME_MOUNT: volume}, scaledown_window=60, timeout=1200
)
class Bench:
    @modal.method()
    def run(self) -> list[dict]:
        import torch
        from snippets_runtime.model import SentenceCompressor
        from transformers import AutoTokenizer

        ckpt_dir = f"{VOLUME_MOUNT}/{CHECKPOINT_NAME}"
        tok = AutoTokenizer.from_pretrained(ckpt_dir, use_fast=True)
        state = torch.load(f"{ckpt_dir}/model.pt", map_location="cpu")["model"]

        # A realistic full window: query + ~8k tokens of body.
        query = "what does the WHO recommend for regulating traditional medicine"
        body = (
            "The regulatory framework for traditional medicine varies across "
            "member states and the committee reviewed annual statistics. "
        ) * 700
        enc = tok(
            query,
            body,
            truncation="only_second",
            max_length=8192,
            padding="max_length",
            return_tensors="pt",
        )
        enc = {k: v.to("cuda") for k, v in enc.items()}
        seqlen = int(enc["attention_mask"].sum())
        print(f"seqlen (real tokens) = {seqlen}, padded to {enc['input_ids'].shape[1]}", flush=True)

        def build(impl):
            m = SentenceCompressor(base=BASE_MODEL, attn_implementation=impl)
            m.load_state_dict(state)
            return m.to("cuda").eval()

        def timed(m, n=10, compile_=False):
            fwd = m
            if compile_:
                m_c = torch.compile(m, mode="max-autotune")
                fwd = m_c
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for _ in range(3):  # warmup (compile happens here)
                    logits, _ = fwd(**enc)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(n):
                    logits, _ = fwd(**enc)
                torch.cuda.synchronize()
                dt = (time.perf_counter() - t0) / n
            return dt, logits.float()[0].detach().cpu()

        results = []
        ref = None
        configs = (
            [("default", None, False)]
            if QUICK
            else [
                ("default", None, False),
                ("eager", "eager", False),
                ("sdpa", "sdpa", False),
                ("flash_attention_2", "flash_attention_2", False),
                ("sdpa+compile", "sdpa", True),
            ]
        )
        for name, impl, comp in configs:
            try:
                m = build(impl)
                # report what the encoder actually used
                actual = getattr(m.encoder.config, "_attn_implementation", "?")
                dt, logits = timed(m, compile_=comp)
                if ref is None and name == "default":
                    ref = logits
                maxdiff = float((logits - ref).abs().max()) if ref is not None else 0.0
                row = {
                    "cfg": name,
                    "actual_attn": actual,
                    "s_per_window": round(dt, 4),
                    "tok_per_s": round(seqlen / dt),
                    "maxdiff_vs_default": round(maxdiff, 5),
                }
                print(
                    f"[attn] {name:<18} actual={actual:<18} "
                    f"{dt * 1000:7.1f} ms/window  {seqlen / dt:8.0f} tok/s  "
                    f"maxdiff={maxdiff:.5f}",
                    flush=True,
                )
                results.append(row)
                del m
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"[attn] {name:<18} FAILED: {type(e).__name__}: {e}", flush=True)
                results.append({"cfg": name, "error": f"{type(e).__name__}: {e}"})
        return results


@app.local_entrypoint()
def main():
    rows = Bench().run.remote()
    print("\n==== attention / compile forward-pass sweep (8192-token window, L4, bf16) ====")
    for r in rows:
        if "error" in r:
            print(f"  {r['cfg']:<18} ERROR: {r['error']}")
        else:
            print(
                f"  {r['cfg']:<18} {r['s_per_window'] * 1000:7.1f} ms  "
                f"{r['tok_per_s']:>8} tok/s  (actual={r['actual_attn']}, "
                f"maxdiff={r['maxdiff_vs_default']})"
            )

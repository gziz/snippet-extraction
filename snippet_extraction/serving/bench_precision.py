"""Test inference precision (fp32 vs bf16 vs fp16) on the forward pass — the
phase that dominates compress latency (~85%). Measures speed and, critically,
whether the kept-snippet set still matches fp32 (methodology safety).

    cd snippet_extraction/serving
    modal run bench_precision.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import modal

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "runtime"))
sys.path.insert(0, str(_REPO / "common"))

APP_NAME = "query-aware-compressor-bench-prec"
CHECKPOINT_NAME = "run9"
VOLUME_MOUNT = "/checkpoints"
BASE_MODEL = "answerdotai/ModernBERT-base"
try:
    BENCH_DOCS = Path(__file__).resolve().parents[2] / "search_evals" / "sandbox" / "bench_docs"
except IndexError:
    BENCH_DOCS = Path("/bench_docs")

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
        "numpy>=1.26",
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
    timeout=1200,
)
class Bench:
    @modal.enter()
    def load(self):
        from snippets_runtime.inference import load_for_inference

        ckpt_dir = f"{VOLUME_MOUNT}/{CHECKPOINT_NAME}"
        self.model, self.tokenizer = load_for_inference(ckpt_dir, base=BASE_MODEL, device="cuda")

    @modal.method()
    def run(self, docs: list[tuple[str, str]]) -> list[dict]:
        import numpy as np
        import torch
        from snippets_common.segment import segment
        from snippets_common.spans import build_body_and_spans
        from snippets_common.windowing import pack_windows

        tok = self.tokenizer
        model = self.model
        device = "cuda"
        max_length = 8192
        window_margin = 8

        def plan_fast(units, query):
            q_tokens = len(tok(query, add_special_tokens=False)["input_ids"])
            specials = tok.num_special_tokens_to_add(pair=True)
            capacity = max(64, max_length - q_tokens - specials - window_margin)
            enc = tok(units, add_special_tokens=False)["input_ids"]
            unit_lens = [len(x) for x in enc]
            return list(pack_windows(unit_lens, capacity))

        def pool_fast(offsets, seq_ids, probs, spans, n_units, tt, st):
            offs = np.asarray(offsets, dtype=np.int64)
            sids = np.asarray([s if s is not None else -1 for s in seq_ids], dtype=np.int64)
            p = np.asarray(probs, dtype=np.float64)
            s, e = offs[:, 0], offs[:, 1]
            body = (sids == 1) & (e > s)
            if not body.any() or n_units == 0:
                return []
            mids = (s[body] + e[body]) // 2
            pb = p[body]
            starts = np.asarray([sp[0] for sp in spans], dtype=np.int64)
            ends = np.asarray([sp[1] for sp in spans], dtype=np.int64)
            cand = np.searchsorted(starts, mids, side="right") - 1
            valid = (cand >= 0) & (mids < ends[np.clip(cand, 0, n_units - 1)])
            u = cand[valid]
            pv = pb[valid]
            counts = np.bincount(u, minlength=n_units).astype(np.float64)
            sum_above = np.bincount(u, weights=(pv >= tt).astype(np.float64), minlength=n_units)
            has = counts > 0
            frac_above = np.zeros(n_units)
            frac_above[has] = sum_above[has] / counts[has]
            return [int(j) for j in np.nonzero(has & (frac_above >= st))[0]]

        def score_doc(units, query, dtype):
            """Return (kept_set, fwd_seconds) for a given autocast dtype (or None=fp32)."""
            windows = plan_fast(units, query)
            kept_all = []
            fwd_s = 0.0
            for start, end in windows:
                w_units = units[start:end]
                body, spans = build_body_and_spans(w_units)
                enc = tok(
                    query,
                    body,
                    truncation="only_second",
                    max_length=max_length,
                    return_offsets_mapping=True,
                    return_tensors="pt",
                )
                offsets = enc.pop("offset_mapping")[0].tolist()
                seq_ids = enc.sequence_ids()
                enc = {k: v.to(device) for k, v in enc.items()}
                torch.cuda.synchronize()
                t = time.perf_counter()
                with torch.no_grad():
                    if dtype is None:
                        logits, _ = model(**enc)
                    else:
                        with torch.autocast("cuda", dtype=dtype):
                            logits, _ = model(**enc)
                    probs = torch.sigmoid(logits)[0].float().cpu().tolist()
                torch.cuda.synchronize()
                fwd_s += time.perf_counter() - t
                kept_all.extend(
                    start + j
                    for j in pool_fast(offsets, seq_ids, probs, spans, len(w_units), 0.5, 0.5)
                )
            return set(kept_all), fwd_s

        modes = [("fp32", None), ("bf16", torch.bfloat16), ("fp16", torch.float16)]
        results = []
        for name, query in docs:
            md = Path(f"/bench_docs/{name}.md").read_text()
            units = segment(md)
            # warm
            score_doc(units, query, torch.bfloat16)
            ref = None
            row = {"doc": name, "n_units": len(units)}
            for label, dt in modes:
                kept, fwd_s = score_doc(units, query, dt)
                if label == "fp32":
                    ref = kept
                inter = len(kept & ref) if ref is not None else len(kept)
                union = len(kept | ref) if ref is not None else len(kept)
                jacc = inter / union if union else 1.0
                row[f"{label}_fwd_s"] = round(fwd_s, 2)
                row[f"{label}_nkept"] = len(kept)
                row[f"{label}_jaccard_vs_fp32"] = round(jacc, 4)
                print(
                    f"[prec] {name} {label}: fwd={fwd_s:.2f}s nkept={len(kept)} "
                    f"jaccard_vs_fp32={jacc:.4f}",
                    flush=True,
                )
            results.append(row)
        return results


@app.local_entrypoint()
def main():
    rows = Bench().run.remote(DOCS)
    print("\n==== precision: forward-pass speed & agreement vs fp32 ====")
    for r in rows:
        print(f"\n{r['doc']} ({r['n_units']:,} units)")
        f32 = r["fp32_fwd_s"]
        for label in ("fp32", "bf16", "fp16"):
            fwd = r[f"{label}_fwd_s"]
            print(
                f"  {label:>5}: fwd={fwd:>6.2f}s  speedup={f32 / fwd:>4.2f}x  "
                f"nkept={r[f'{label}_nkept']:>4}  "
                f"jaccard_vs_fp32={r[f'{label}_jaccard_vs_fp32']}"
            )

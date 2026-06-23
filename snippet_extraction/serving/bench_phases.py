"""Profile + optimize the compress pipeline phases in ONE warm container.

The batch-token sweep proved wall time is flat vs GPU batch size, so the cost
is host-side. This benchmark breaks a long doc's compress into phases and times
each, then tests optimized variants inline (asserting identical output) so we
find the efficient approach without paying cold-start per iteration.

    cd snippet_extraction/serving
    modal run bench_phases.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import modal

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "runtime"))
sys.path.insert(0, str(_REPO / "common"))

APP_NAME = "query-aware-compressor-bench-phases"
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
        from snippets_runtime.inference import _pool_window

        tok = self.tokenizer
        model = self.model
        device = "cuda"
        max_length = 8192
        window_margin = 8

        # ---- phase: window planning (length tokenization) -------------- #
        def plan_slow(units, query):
            q_tokens = len(tok(query, add_special_tokens=False)["input_ids"])
            specials = tok.num_special_tokens_to_add(pair=True)
            capacity = max(64, max_length - q_tokens - specials - window_margin)
            unit_lens = [len(tok(u, add_special_tokens=False)["input_ids"]) for u in units]
            return list(pack_windows(unit_lens, capacity)), capacity

        def plan_fast(units, query):
            q_tokens = len(tok(query, add_special_tokens=False)["input_ids"])
            specials = tok.num_special_tokens_to_add(pair=True)
            capacity = max(64, max_length - q_tokens - specials - window_margin)
            # One batched tokenizer call instead of one per unit.
            enc = tok(units, add_special_tokens=False)["input_ids"]
            unit_lens = [len(x) for x in enc]
            return list(pack_windows(unit_lens, capacity)), capacity

        # ---- phase: pooling -------------------------------------------- #
        def pool_fast(offsets, seq_ids, probs, spans, n_units, token_threshold, sentence_threshold):
            """Vectorized equivalent of _pool_window using numpy."""
            offs = np.asarray(offsets, dtype=np.int64)  # (T, 2)
            sids = np.asarray([s if s is not None else -1 for s in seq_ids], dtype=np.int64)  # (T,)
            p = np.asarray(probs, dtype=np.float64)  # (T,)
            s, e = offs[:, 0], offs[:, 1]
            body = (sids == 1) & (e > s)
            if not body.any() or n_units == 0:
                return ([0.0] * n_units, [0.0] * n_units, [])
            mids = (s[body] + e[body]) // 2
            pb = p[body]
            starts = np.asarray([sp[0] for sp in spans], dtype=np.int64)
            ends = np.asarray([sp[1] for sp in spans], dtype=np.int64)
            cand = np.searchsorted(starts, mids, side="right") - 1
            valid = (cand >= 0) & (mids < ends[np.clip(cand, 0, n_units - 1)])
            u = cand[valid]
            pv = pb[valid]
            counts = np.bincount(u, minlength=n_units).astype(np.float64)
            sum_prob = np.bincount(u, weights=pv, minlength=n_units)
            sum_above = np.bincount(
                u, weights=(pv >= token_threshold).astype(np.float64), minlength=n_units
            )
            has = counts > 0
            frac_above = np.zeros(n_units)
            mean_prob = np.zeros(n_units)
            frac_above[has] = sum_above[has] / counts[has]
            mean_prob[has] = sum_prob[has] / counts[has]
            kept = [int(j) for j in np.nonzero(has & (frac_above >= sentence_threshold))[0]]
            return (frac_above.tolist(), mean_prob.tolist(), kept)

        results = []
        for name, query in docs:
            md = Path(f"/bench_docs/{name}.md").read_text()
            units = segment(md)
            n_units = len(units)

            # plan: slow vs fast (assert identical windows)
            t = time.perf_counter()
            w_slow, cap = plan_slow(units, query)
            plan_slow_s = time.perf_counter() - t
            t = time.perf_counter()
            w_fast, _ = plan_fast(units, query)
            plan_fast_s = time.perf_counter() - t
            assert w_slow == w_fast, f"window mismatch on {name}"

            # build windows
            jobs = []
            for start, end in w_slow:
                w_units = units[start:end]
                body, spans = build_body_and_spans(w_units)
                jobs.append((start, end, w_units, body, spans))

            # forward + pool, timing each phase; compare slow vs fast pooling
            fwd_s = 0.0
            pool_slow_s = 0.0
            pool_fast_s = 0.0
            kept_slow_all, kept_fast_all = [], []
            for start, end, w_units, body, spans in jobs:
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
                    logits, _ = model(**enc)
                probs = torch.sigmoid(logits)[0].float().cpu().tolist()
                torch.cuda.synchronize()
                fwd_s += time.perf_counter() - t

                t = time.perf_counter()
                ss, mm, kk = _pool_window(offsets, seq_ids, probs, spans, len(w_units), 0.5, 0.5)
                pool_slow_s += time.perf_counter() - t
                kept_slow_all.extend(start + j for j in kk)

                t = time.perf_counter()
                ss2, mm2, kk2 = pool_fast(offsets, seq_ids, probs, spans, len(w_units), 0.5, 0.5)
                pool_fast_s += time.perf_counter() - t
                kept_fast_all.extend(start + j for j in kk2)

            kept_match = sorted(kept_slow_all) == sorted(kept_fast_all)
            results.append(
                {
                    "doc": name,
                    "n_units": n_units,
                    "n_windows": len(jobs),
                    "plan_slow_s": plan_slow_s,
                    "plan_fast_s": plan_fast_s,
                    "fwd_s": fwd_s,
                    "pool_slow_s": pool_slow_s,
                    "pool_fast_s": pool_fast_s,
                    "kept_match": kept_match,
                    "n_kept": len(kept_slow_all),
                }
            )
            print(
                f"[phase] {name}: units={n_units} windows={len(jobs)} "
                f"plan {plan_slow_s:.2f}->{plan_fast_s:.2f}s  fwd {fwd_s:.2f}s  "
                f"pool {pool_slow_s:.2f}->{pool_fast_s:.2f}s  kept_match={kept_match}",
                flush=True,
            )
        return results


@app.local_entrypoint()
def main():
    rows = Bench().run.remote(DOCS)
    print("\n==== phase breakdown (seconds) ====")
    hdr = (
        "doc",
        "units",
        "win",
        "plan_slow",
        "plan_fast",
        "fwd",
        "pool_slow",
        "pool_fast",
        "match",
    )
    print("  " + "  ".join(f"{h:>10}" for h in hdr))
    for r in rows:
        print(
            "  "
            + "  ".join(
                f"{v:>10}"
                for v in (
                    r["doc"],
                    r["n_units"],
                    r["n_windows"],
                    f"{r['plan_slow_s']:.2f}",
                    f"{r['plan_fast_s']:.2f}",
                    f"{r['fwd_s']:.2f}",
                    f"{r['pool_slow_s']:.2f}",
                    f"{r['pool_fast_s']:.2f}",
                    str(r["kept_match"]),
                )
            )
        )
    for r in rows:
        old = r["plan_slow_s"] + r["fwd_s"] + r["pool_slow_s"]
        new = r["plan_fast_s"] + r["fwd_s"] + r["pool_fast_s"]
        print(
            f"  {r['doc']}: scored-total {old:.2f}s -> {new:.2f}s "
            f"({old / new:.2f}x), kept_match={r['kept_match']}"
        )

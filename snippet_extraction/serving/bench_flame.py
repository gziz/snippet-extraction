"""Span-instrumented profile of one full compress request -> flame graph data.

Reproduces the production pipeline (segment -> plan -> tokenize -> H2D -> forward
-> D2H -> pool -> select -> render) inside one warm L40S container, wrapping each
phase in a nested span with ``cuda.synchronize`` bracketing so GPU time is
attributed to the forward span and not smeared across the host phases. Emits a
flat span list (name, depth, start_ms, dur_ms) that render_flame.py draws.

    modal run bench_flame.py            # profiles a long (~8-window) doc
    modal run bench_flame.py --sizes medium,long

The two-container hop (CPU endpoint -> GPU class) is profiled here as one
container; the per-phase durations are identical regardless of which container
runs them. The endpoint->GPU dispatch + HTTP is reported separately by the
caller (warm round-trip minus measured server compute).
"""

from __future__ import annotations

import contextlib
import sys
import time
from pathlib import Path

import modal

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "runtime"))
sys.path.insert(0, str(_REPO / "common"))

APP_NAME = "query-aware-compressor-bench-flame"
CHECKPOINT_NAME = "run9"
VOLUME_MOUNT = "/checkpoints"
BASE_MODEL = "answerdotai/ModernBERT-base"
GPU = "L40S"


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
        "numpy>=1.26",
    )
    .run_function(_prefetch_base_assets)
    .add_local_python_source("snippets_runtime")
    .add_local_python_source("snippets_common")
)

volume = modal.Volume.from_name("compressor-checkpoints", create_if_missing=True)
app = modal.App(APP_NAME)

_LEX = [
    "The regulatory framework for traditional medicine varies considerably across member states and regions.",
    "Clinical evidence supporting complementary therapies remains uneven and is frequently debated by researchers.",
    "Public health authorities recommend integrating safety monitoring into national pharmacovigilance systems.",
    "Economic analyses suggest that preventive interventions reduce long term hospital admission costs substantially.",
    "The committee reviewed annual statistics covering mortality, morbidity, and access to essential services.",
    "Implementation guidelines emphasize training, accreditation, and continuous professional development.",
    "Surveillance data indicate seasonal variation in reported cases across both urban and rural districts.",
    "Funding allocations were adjusted to prioritize underserved populations and remote community clinics.",
]


def _make_doc(n_sentences: int) -> str:
    return "\n\n".join(f"[{i}] {_LEX[i % len(_LEX)]}" for i in range(n_sentences))


# ~450 sentences ≈ one 8192-token window.
SIZE_SENTENCES = {"short": 120, "medium": 450, "long": 3600}


class Spans:
    """Nested wall-clock span recorder -> flat list for a flame/icicle chart."""

    def __init__(self):
        self.spans: list[dict] = []
        self.depth = 0
        self._t0 = time.perf_counter()

    @contextlib.contextmanager
    def span(self, name: str):
        d = self.depth
        s = (time.perf_counter() - self._t0) * 1000.0
        self.depth += 1
        try:
            yield
        finally:
            self.depth = d
            e = (time.perf_counter() - self._t0) * 1000.0
            self.spans.append(
                {"name": name, "depth": d, "start": round(s, 3), "dur": round(e - s, 3)}
            )


@app.cls(image=image, gpu=GPU, volumes={VOLUME_MOUNT: volume}, scaledown_window=120, timeout=1200)
class Flame:
    @modal.enter()
    def load(self):
        from snippets_runtime.inference import load_for_inference

        self.model, self.tokenizer = load_for_inference(
            f"{VOLUME_MOUNT}/{CHECKPOINT_NAME}", base=BASE_MODEL, device="cuda"
        )

    @modal.method()
    def profile(self, document: str, query: str, max_batch_tokens: int = 16384):
        import numpy as np
        import torch
        from snippets_common.segment import segment
        from snippets_common.spans import build_body_and_spans
        from snippets_common.windowing import pack_windows, render_snippet
        from snippets_runtime.inference import INFERENCE_DTYPE, _pool_window

        tok, model, device = self.tokenizer, self.model, "cuda"
        max_length, window_margin = 8192, 8

        def sync():
            torch.cuda.synchronize()

        # Warm the GPU (cuDNN autotune / allocator) so the profiled run is steady.
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            warm = tok(
                "q",
                "warm up the kernels " * 200,
                truncation="only_second",
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            model(**{k: v for k, v in warm.items() if k != "offset_mapping"})
        sync()

        sp = Spans()
        with sp.span(f"request total ({len(document):,} chars)"):
            with sp.span("segment (sentence splitter)"):
                units = segment(document)

            with sp.span("score (GPU container)"):
                # ---- plan windows (length tokenization) ----
                with sp.span("plan windows (tokenize lengths)"):
                    q_tokens = len(tok(query, add_special_tokens=False)["input_ids"])
                    specials = tok.num_special_tokens_to_add(pair=True)
                    capacity = max(64, max_length - q_tokens - specials - window_margin)
                    unit_lens = [len(x) for x in tok(units, add_special_tokens=False)["input_ids"]]
                    windows = list(pack_windows(unit_lens, capacity))

                with sp.span(f"build bodies+spans ({len(windows)} windows)"):
                    jobs = []
                    for start, end in windows:
                        w_units = units[start:end]
                        body, spans = build_body_and_spans(w_units)
                        est = min(max_length, q_tokens + specials + sum(unit_lens[start:end]))
                        jobs.append((start, end, w_units, body, spans, est))

                # length-bucket + greedy fill (mirrors compress_long_batch)
                order = sorted(range(len(jobs)), key=lambda k: jobs[k][5])
                minibatches, i = [], 0
                while i < len(order):
                    batch, mx = [], 0
                    while i < len(order):
                        cand = max(mx, jobs[order[i]][5])
                        if batch and cand * (len(batch) + 1) > max_batch_tokens:
                            break
                        batch.append(jobs[order[i]])
                        mx = cand
                        i += 1
                    minibatches.append(batch)

                kept_all: list[int] = []
                sent_acc = [0.0] * len(units)
                mean_acc = [0.0] * len(units)
                with sp.span(f"minibatch loop ({len(minibatches)} passes)"):
                    for bi, batch in enumerate(minibatches):
                        with sp.span(f"minibatch {bi} ({len(batch)} win)"):
                            with sp.span("tokenize (query,body)"):
                                enc = tok(
                                    [query for _ in batch],
                                    [j[3] for j in batch],
                                    truncation="only_second",
                                    max_length=max_length,
                                    padding=True,
                                    return_offsets_mapping=True,
                                    return_tensors="pt",
                                )
                                offsets_b = enc.pop("offset_mapping").tolist()
                                seq_ids_b = [enc.sequence_ids(k) for k in range(len(batch))]
                            with sp.span("H2D copy"):
                                enc = {k: v.to(device) for k, v in enc.items()}
                                sync()
                            with sp.span("forward (GPU)"):
                                with (
                                    torch.no_grad(),
                                    torch.autocast(
                                        "cuda",
                                        dtype=torch.bfloat16 if INFERENCE_DTYPE else torch.float32,
                                    ),
                                ):
                                    logits, _ = model(**enc)
                                sync()
                            with sp.span("sigmoid + D2H copy"):
                                probs_b = torch.sigmoid(logits).float().cpu().tolist()
                            with sp.span("pool (numpy)"):
                                for (
                                    start,
                                    end,
                                    w_units,
                                    body,
                                    spans,
                                    est,
                                ), offs, sids, probs in zip(batch, offsets_b, seq_ids_b, probs_b):
                                    s, m, kept = _pool_window(
                                        offs, sids, probs, spans, len(w_units), 0.5, 0.5
                                    )
                                    for li, gi in enumerate(range(start, end)):
                                        sent_acc[gi] = s[li]
                                        mean_acc[gi] = m[li]
                                    kept_all.extend(start + j for j in kept)

                with sp.span("select + render"):
                    kept_sorted = sorted(kept_all)
                    _ = render_snippet(units, kept_sorted)

        compute_ms = sp.spans[-1]["dur"]
        return {
            "spans": sp.spans,
            "n_units": len(units),
            "n_windows": len(windows),
            "n_minibatches": len(minibatches),
            "n_kept": len(kept_all),
            "compute_ms": compute_ms,
        }


@app.local_entrypoint()
def main(sizes: str = "long"):
    import json

    query = "what does the WHO recommend for regulating traditional medicine"
    out = {}
    for size in sizes.split(","):
        size = size.strip()
        doc = _make_doc(SIZE_SENTENCES[size])
        res = Flame().profile.remote(doc, query)
        out[size] = res
        print(
            f"\n=== {size}: {res['n_units']} units, {res['n_windows']} windows, "
            f"{res['n_minibatches']} minibatches, compute={res['compute_ms']:.1f} ms ==="
        )
        # flat phase totals (depth>=1 leaves under score)
        for s in res["spans"]:
            print(f"  {'  ' * s['depth']}{s['name']:<34} {s['dur']:8.2f} ms")
    Path("/tmp/flame_spans.json").write_text(json.dumps(out, indent=2))
    print("\nwrote /tmp/flame_spans.json")

"""Throughput-vs-batch-size sweep harness — a SEPARATE Modal app.

This is a disposable benchmark deployment. It is intentionally a different
``modal.App`` (``APP_NAME`` below) with its own endpoint URL so running it does
NOT touch the production ``query-aware-compressor`` deployment or any in-flight
eval. Tear it down with ``modal app stop`` when finished.

It mirrors the production scorer but adds two knobs the production app hardcodes,
both read per-request so a single deploy can sweep every value without redeploys:

  - ``max_batch_tokens`` : the real GPU forward-pass budget (rows * padded_len),
    the lever that actually controls GPU utilization / memory.
  - ``report_gpu``       : when true, attach peak CUDA memory (MiB) to the
    response so the sweep can read the memory ceiling alongside latency.

Deploy:   modal deploy modal_bench.py
Sweep:    python bench_sweep.py            (drives this endpoint)
Teardown: modal app stop query-aware-compressor-bench
"""

from __future__ import annotations

import sys
from pathlib import Path

import modal

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "runtime"))
sys.path.insert(0, str(_REPO / "common"))

APP_NAME = "query-aware-compressor-bench"  # distinct app -> distinct URL
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
        "fastapi[standard]",
        "syntok>=1.4",
        "markdown-it-py>=3.0",
        "tiktoken>=0.7",
    )
    .run_function(_prefetch_base_assets)
    .add_local_python_source("snippets_runtime")
    .add_local_python_source("snippets_common")
)

volume = modal.Volume.from_name("compressor-checkpoints", create_if_missing=True)

app = modal.App(APP_NAME)


@app.cls(
    image=image,
    gpu="L4",
    volumes={VOLUME_MOUNT: volume},
    scaledown_window=120,
    # Pin to ONE container so the sweep measures pure per-batch GPU cost without
    # autoscaling muddying the latency signal.
    min_containers=1,
    max_containers=1,
)
class BenchCompressor:
    @modal.enter()
    def load(self):
        import tiktoken
        from snippets_runtime.inference import load_for_inference

        ckpt_dir = f"{VOLUME_MOUNT}/{CHECKPOINT_NAME}"
        self.model, self.tokenizer = load_for_inference(ckpt_dir, base=BASE_MODEL, device="cuda")
        self._enc = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(self, text: str) -> int:
        return len(self._enc.encode(text, disallowed_special=()))

    @modal.batched(max_batch_size=32, wait_ms=50)
    def score(self, reqs: list[dict]) -> list[dict]:
        import time

        import torch
        from snippets_common.segment import segment
        from snippets_common.windowing import render_snippet, select_under_budget
        from snippets_runtime.inference import DocRequest, compress_long_batch

        # The per-batch knob being swept (same value across a batch in practice).
        max_batch_tokens = int(reqs[0].get("max_batch_tokens", 16384))
        report_gpu = bool(reqs[0].get("report_gpu", False))

        docs_units: list[list[str]] = []
        doc_reqs: list[DocRequest] = []
        for r in reqs:
            document = r["document"]
            units = segment(document) if isinstance(document, str) else list(document)
            docs_units.append(units)
            doc_reqs.append(
                DocRequest(
                    query=r["query"],
                    document=units,
                    token_threshold=float(r.get("token_threshold", 0.5)),
                    sentence_threshold=float(r.get("sentence_threshold", 0.5)),
                )
            )

        if report_gpu:
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        results = compress_long_batch(
            self.model,
            self.tokenizer,
            doc_reqs,
            device="cuda",
            max_batch_tokens=max_batch_tokens,
        )
        gpu_s = time.perf_counter() - t0
        peak_mib = torch.cuda.max_memory_allocated() / (1024 * 1024) if report_gpu else 0.0

        out: list[dict] = []
        for r, units, res in zip(reqs, docs_units, results):
            if not units:
                out.append(
                    {
                        "snippet": "",
                        "units": [],
                        "kept_indices": [],
                        "batch_size": len(reqs),
                        "max_batch_tokens": max_batch_tokens,
                        "gpu_batch_s": gpu_s,
                        "gpu_peak_mib": peak_mib,
                    }
                )
                continue
            budget = r.get("budget_tokens")
            if budget is not None:
                kept = select_under_budget(
                    units,
                    res.sentence_mean_probs,
                    int(budget),
                    self._count_tokens,
                    min_score=float(r.get("min_score", 0.1)),
                )
            else:
                kept = res.kept_indices
            out.append(
                {
                    "snippet": render_snippet(units, kept),
                    "units": units,
                    "kept_indices": kept,
                    # Telemetry the sweep reads:
                    "batch_size": len(reqs),
                    "max_batch_tokens": max_batch_tokens,
                    "gpu_batch_s": gpu_s,  # wall time of the whole batched score
                    "gpu_peak_mib": peak_mib,  # peak CUDA memory for this batch
                }
            )
        return out


@app.function(image=image)
@modal.concurrent(max_inputs=100)
@modal.fastapi_endpoint(method="POST")
async def compress(req: dict):
    from fastapi import HTTPException

    if not req.get("query") or not req.get("document"):
        raise HTTPException(400, "both 'query' and 'document' are required")
    return await BenchCompressor().score.remote.aio(req)

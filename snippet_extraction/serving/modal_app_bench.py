"""Parameterized copy of modal_app.py for capacity benchmarking. BENCH ONLY.

Same pipeline as modal_app.py (segment on CPU endpoint -> @modal.batched GPU
scorer -> select/render), but APP_NAME / GPU / MAX_CONTAINERS come from env
vars so we can deploy throwaway variants side by side without touching the
production app:

    BENCH_APP_NAME=qac-bench-l40s-c1 BENCH_GPU=L40S BENCH_MAX_CONTAINERS=1 \
        modal deploy modal_app_bench.py

Why variants matter: benchmark/loadgen.py measures whatever fleet the app is
allowed to scale to. max_containers=1 gives the clean per-GPU capacity (mu);
max_containers=3 against the same GPU checks that throughput scales ~linearly.

Clean up when done (idle apps cost $0 with min_containers=0, but still):

    modal app stop qac-bench-l40s-c1
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import modal

if TYPE_CHECKING:
    from snippets_runtime.inference import CompressResult, DocRequest

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "runtime"))
sys.path.insert(0, str(_REPO / "common"))

APP_NAME = os.environ.get("BENCH_APP_NAME", "qac-bench")
BENCH_GPU = os.environ.get("BENCH_GPU", "L40S")
MAX_CONTAINERS = int(os.environ.get("BENCH_MAX_CONTAINERS", "1"))
CHECKPOINT_NAME = "run9"  # keep in sync with modal_app.py
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
        "numpy>=1.26",
    )
    .run_function(_prefetch_base_assets)
    .add_local_python_source("snippets_runtime")
    .add_local_python_source("snippets_common")
)

volume = modal.Volume.from_name("compressor-checkpoints", create_if_missing=True)

app = modal.App(APP_NAME)


@app.cls(
    image=image,
    gpu=BENCH_GPU,
    volumes={VOLUME_MOUNT: volume},
    scaledown_window=120,
    min_containers=0,
    max_containers=MAX_CONTAINERS,
    enable_memory_snapshot=True,
)
class Compressor:
    @modal.enter(snap=True)
    def load(self):
        from snippets_runtime.inference import load_for_inference

        ckpt_dir = f"{VOLUME_MOUNT}/{CHECKPOINT_NAME}"
        self.model, self.tokenizer = load_for_inference(ckpt_dir, base=BASE_MODEL, device="cpu")

    @modal.enter(snap=False)
    def to_gpu(self):
        self.model.to("cuda")

    @modal.batched(max_batch_size=6, wait_ms=50)
    def score(self, doc_reqs: list[DocRequest]) -> list[CompressResult]:
        from snippets_runtime.inference import compress_long_batch

        return compress_long_batch(self.model, self.tokenizer, doc_reqs, device="cuda")


def _segment_doc(document: str) -> list[str]:
    from snippets_common.segment import segment

    return segment(document)


_ENC = None


def _count_tokens(text: str) -> int:
    global _ENC
    if _ENC is None:
        import tiktoken

        _ENC = tiktoken.get_encoding("cl100k_base")
    return len(_ENC.encode(text, disallowed_special=()))


_EMPTY_RESPONSE = {
    "kept_indices": [],
    "kept_sentences": [],
    "snippet": "",
    "units": [],
    "sentence_scores": [],
    "sentence_mean_probs": [],
    "compression_ratio": 0.0,
}


_SEG_POOL = None


def _seg_pool():
    global _SEG_POOL
    if _SEG_POOL is None:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        _SEG_POOL = ProcessPoolExecutor(max_workers=8, mp_context=mp.get_context("forkserver"))
    return _SEG_POOL


@app.function(image=image, cpu=8.0)
@modal.concurrent(max_inputs=100)
@modal.fastapi_endpoint(method="POST")
async def compress(req: dict):
    import asyncio

    from fastapi import HTTPException
    from snippets_common.windowing import render_snippet, select_under_budget
    from snippets_runtime.inference import DocRequest

    if not req.get("query") or not req.get("document"):
        raise HTTPException(400, "both 'query' and 'document' are required")

    document = req["document"]
    if isinstance(document, str):
        loop = asyncio.get_running_loop()
        units = await loop.run_in_executor(_seg_pool(), _segment_doc, document)
    else:
        units = list(document)

    if not units:
        return dict(_EMPTY_RESPONSE)

    dreq = DocRequest(
        query=req["query"],
        document=units,
        token_threshold=float(req.get("token_threshold", 0.5)),
        sentence_threshold=float(req.get("sentence_threshold", 0.5)),
    )
    result = await Compressor().score.remote.aio(dreq)

    budget = req.get("budget_tokens")
    if budget is not None:
        kept = select_under_budget(
            units,
            result.sentence_mean_probs,
            int(budget),
            _count_tokens,
            min_score=float(req.get("min_score", 0.1)),
        )
    else:
        kept = result.kept_indices

    kept_sentences = [units[j] for j in kept]
    snippet = render_snippet(units, kept)
    total_chars = sum(len(u) for u in units) or 1
    return {
        "kept_indices": kept,
        "kept_sentences": kept_sentences,
        "snippet": snippet,
        "units": units,
        "sentence_scores": result.sentence_scores,
        "sentence_mean_probs": result.sentence_mean_probs,
        "compression_ratio": sum(len(s) for s in kept_sentences) / total_chars,
    }

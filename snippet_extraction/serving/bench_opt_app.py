"""Isolated A/B benchmark deployment for serving optimizations.

This is a DISPOSABLE benchmark app. It is a *separate* ``modal.App`` (its name
includes ``bench-opt``) so deploying/running it NEVER touches the production
``query-aware-compressor-l4`` deployment or any in-flight eval.

It mirrors the production code path (the ``compress`` web endpoint + the batched
``Compressor`` GPU class) exactly, with two knobs read from environment variables
at deploy time so a single file produces both arms of the A/B:

    BENCH_VARIANT = baseline | snapshot | gpusnap   (default: baseline)
    BENCH_MAXC    = max_containers                  (default: 4)
    BENCH_MAXBATCH= score max_batch_size            (default: 6)

  baseline : current production config (no memory snapshot).
  snapshot : CPU memory snapshot — load weights on CPU inside @enter(snap=True),
             move to CUDA in @enter(snap=False). Skips torch/transformers import
             + model construction + weight load on every cold start.
  gpusnap  : same but also enable_gpu_snapshot (alpha) so CUDA state is captured.

Deploy an arm:
    BENCH_VARIANT=baseline modal deploy bench_opt_app.py
    BENCH_VARIANT=snapshot modal deploy bench_opt_app.py

Drive it with bench_loadtest.py. Tear down with:
    modal app stop query-aware-compressor-bench-opt-baseline
    modal app stop query-aware-compressor-bench-opt-snapshot
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import modal

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "runtime"))
sys.path.insert(0, str(_REPO / "common"))

VARIANT = os.environ.get("BENCH_VARIANT", "baseline").strip()
MAX_CONTAINERS = int(os.environ.get("BENCH_MAXC", "4"))
MAX_BATCH = int(os.environ.get("BENCH_MAXBATCH", "6"))
GPU = os.environ.get("BENCH_GPU", "L4").strip()

assert VARIANT in {"baseline", "snapshot", "gpusnap"}, VARIANT

_gpu_slug = GPU.replace(":", "x").replace("-", "").lower()
APP_NAME = f"query-aware-compressor-bench-opt-{VARIANT}-{_gpu_slug}"
CHECKPOINT_NAME = "run9"
VOLUME_MOUNT = "/checkpoints"
BASE_MODEL = "answerdotai/ModernBERT-base"

SNAPSHOT = VARIANT in {"snapshot", "gpusnap"}
GPU_SNAPSHOT = VARIANT == "gpusnap"


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

_cls_kwargs: dict = dict(
    image=image,
    gpu=GPU,
    volumes={VOLUME_MOUNT: volume},
    scaledown_window=120,
    min_containers=0,
    max_containers=MAX_CONTAINERS,
    enable_memory_snapshot=SNAPSHOT,
)
if GPU_SNAPSHOT:
    _cls_kwargs["experimental_options"] = {"enable_gpu_snapshot": True}


@app.cls(**_cls_kwargs)
class Compressor:
    if SNAPSHOT and not GPU_SNAPSHOT:
        # CPU memory snapshot: no GPU during snap. Load weights on CPU so the
        # snapshot captures the (expensive) torch/transformers import + model
        # construction + state_dict load. On restore only .to("cuda") runs.
        @modal.enter(snap=True)
        def load_cpu(self):
            import tiktoken
            from snippets_runtime.inference import load_for_inference

            ckpt_dir = f"{VOLUME_MOUNT}/{CHECKPOINT_NAME}"
            self.model, self.tokenizer = load_for_inference(ckpt_dir, base=BASE_MODEL, device="cpu")
            self._enc = tiktoken.get_encoding("cl100k_base")

        @modal.enter(snap=False)
        def to_gpu(self):
            self.model.to("cuda")

    else:
        # baseline + gpusnap: load directly on the GPU. For gpusnap this runs
        # inside snap=True with the GPU present, so CUDA state is snapshotted.
        @modal.enter(snap=GPU_SNAPSHOT)
        def load(self):
            import tiktoken
            from snippets_runtime.inference import load_for_inference

            ckpt_dir = f"{VOLUME_MOUNT}/{CHECKPOINT_NAME}"
            self.model, self.tokenizer = load_for_inference(
                ckpt_dir, base=BASE_MODEL, device="cuda"
            )
            self._enc = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(self, text: str) -> int:
        return len(self._enc.encode(text, disallowed_special=()))

    @modal.batched(max_batch_size=MAX_BATCH, wait_ms=50)
    def score(self, reqs: list[dict]) -> list[dict]:
        from snippets_common.windowing import render_snippet, select_under_budget
        from snippets_runtime.inference import DocRequest, compress_long_batch

        docs_units: list[list[str]] = []
        doc_reqs: list[DocRequest] = []
        for r in reqs:
            from snippets_common.segment import segment

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

        results = compress_long_batch(self.model, self.tokenizer, doc_reqs, device="cuda")

        out: list[dict] = []
        for r, units, res in zip(reqs, docs_units, results):
            if not units:
                out.append(
                    {
                        "kept_indices": [],
                        "snippet": "",
                        "compression_ratio": 0.0,
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
            kept_sentences = [units[j] for j in kept]
            total_chars = sum(len(u) for u in units) or 1
            out.append(
                {
                    "kept_indices": kept,
                    "snippet": render_snippet(units, kept),
                    "compression_ratio": sum(len(s) for s in kept_sentences) / total_chars,
                }
            )
        return out


def _segment_doc(document: str) -> list[str]:
    from snippets_common.segment import segment

    return segment(document)


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

    if not req.get("query") or not req.get("document"):
        raise HTTPException(400, "both 'query' and 'document' are required")

    document = req["document"]
    if isinstance(document, str):
        loop = asyncio.get_running_loop()
        units = await loop.run_in_executor(_seg_pool(), _segment_doc, document)
        req = {**req, "document": units}
    return await Compressor().score.remote.aio(req)

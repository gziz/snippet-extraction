"""Serverless compression endpoint on Modal.

A generic ``query + document -> snippet`` API. The endpoint owns the full
pipeline so callers need nothing but an HTTP client:

  segment  -> the block-aware segmenter from snippets_common (applied when
              ``document`` is a raw string; pass a list to score pre-split units).
  score    -> windowed ModernBERT inference on the GPU. The GPU class does *only*
              the forward pass (``compress_long_batch``); everything off the
              critical GPU path lives on the CPU endpoint.
  select   -> dual-threshold keeping or, when ``budget_tokens`` is set,
              score-ranked selection under a per-document cl100k token budget.
  render   -> kept units joined in document order, ``[...]`` marking gaps.

Deploy (CHECKPOINT_NAME must match the uploaded volume dir):

    modal volume put compressor-checkpoints ./checkpoints/<run> /<run>
    modal deploy modal_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import modal

if TYPE_CHECKING:
    # Type-only: keeps the runtime module scope importing nothing but ``modal``
    # (torch/transformers stay out of a local ``modal deploy``).
    from snippets_runtime.inference import CompressResult, DocRequest

# Let add_local_python_source() discover the sibling packages at deploy time
# even when they aren't installed in the local venv (their deps only need to
# exist inside the container image). Keep module scope importing only ``modal``:
# torch/transformers/snippets_* are imported lazily inside the functions below
# so a local ``modal deploy`` never needs the heavy deps. Annotations referring
# to those types stay strings thanks to ``from __future__ import annotations``.
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "runtime"))
sys.path.insert(0, str(_REPO / "common"))

APP_NAME = "query-aware-compressor"
CHECKPOINT_NAME = "run9"  # directory name inside the volume
VOLUME_MOUNT = "/checkpoints"
BASE_MODEL = "answerdotai/ModernBERT-base"


def _prefetch_base_assets() -> None:
    """Bake the HF base model into the image so a cold start only reads the
    fine-tuned weights from the Volume."""
    from transformers import AutoModel, AutoTokenizer

    AutoModel.from_pretrained(BASE_MODEL)
    AutoTokenizer.from_pretrained(BASE_MODEL)


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4",
        "transformers>=4.45",
        "fastapi[standard]",
        # snippets_common.segment (the training-faithful segmenter):
        "syntok>=1.4",
        "markdown-it-py>=3.0",
        "tiktoken>=0.7",
        # vectorized per-token pooling in snippets_runtime.inference:
        "numpy>=1.26",
    )
    .run_function(_prefetch_base_assets)
    .add_local_python_source("snippets_runtime")
    .add_local_python_source("snippets_common")
)

volume = modal.Volume.from_name("compressor-checkpoints", create_if_missing=True)

app = modal.App(APP_NAME)


# GPU + autoscaling rationale (all levers keep min_containers=0 -> $0 idle):
#  - gpu="L40S": the forward is GPU-compute-bound on long docs; L40S is ~4x
#    faster per 8192-token window than L4 and finishes bursts fast enough to be
#    cheaper PER REQUEST despite the higher hourly rate.
#  - enable_memory_snapshot: the cold path is dominated by importing
#    torch/transformers, building the ModernBERT graph, and loading the
#    checkpoint. Doing that load ON CPU inside @enter(snap=True) lets Modal
#    snapshot the post-import state; a later cold start restores the snapshot
#    and only runs @enter(snap=False) to move weights to CUDA (~30s -> ~20s).
#  - max_containers=3 caps fan-out; max_batch_size=6 bounds per-batch work so a
#    burst spreads across containers instead of piling onto one.
@app.cls(
    image=image,
    gpu="L40S",
    volumes={VOLUME_MOUNT: volume},
    scaledown_window=120,
    min_containers=0,
    max_containers=3,
    enable_memory_snapshot=True,
)
class Compressor:
    # No GPU is attached during snapshotting, so load the weights on CPU here;
    # this captures the expensive torch/transformers import + graph build +
    # checkpoint load into the snapshot.
    @modal.enter(snap=True)
    def load(self):
        from snippets_runtime.inference import load_for_inference

        ckpt_dir = f"{VOLUME_MOUNT}/{CHECKPOINT_NAME}"
        self.model, self.tokenizer = load_for_inference(ckpt_dir, base=BASE_MODEL, device="cpu")

    # Runs on every (restored) cold start, once a GPU is attached. The snapshot
    # was taken with no GPU, so the move to CUDA can't happen until now.
    @modal.enter(snap=False)
    def to_gpu(self):
        self.model.to("cuda")

    @modal.batched(max_batch_size=6, wait_ms=50)
    def score(self, doc_reqs: list[DocRequest]) -> list[CompressResult]:
        """GPU-only forward pass. Modal batches the per-document requests from a
        search fan-out into one call; every window of every doc is flattened
        into shared minibatches by ``compress_long_batch``. Segmentation and
        selection/render happen off this critical path on the CPU endpoint."""
        from snippets_runtime.inference import compress_long_batch

        return compress_long_batch(self.model, self.tokenizer, doc_reqs, device="cuda")


def _segment_doc(document: str) -> list[str]:
    """Top-level (picklable) worker run inside the segmentation process pool."""
    from snippets_common.segment import segment

    return segment(document)


_ENC = None


def _count_tokens(text: str) -> int:
    """cl100k token count, computed on the CPU endpoint (not the GPU container).

    cl100k matches the budget metric the eval reports (context_tokens), so
    budgeted selection stays calibrated against it.
    """
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
    """Lazily build a per-container ProcessPoolExecutor for syntok segmentation.

    syntok holds the GIL, so true parallelism needs separate processes. The pool
    is created once per warm endpoint container and reused across requests. It
    uses a ``forkserver`` context rather than the default ``fork``: forking a
    live asyncio/threaded web server can copy held locks into the child and
    deadlock, whereas ``forkserver`` forks workers from a clean, single-threaded
    server process. Workers re-import this module to find ``_segment_doc`` (a
    top-level, picklable target), which is cheap (only ``modal`` is imported at
    module scope; torch/transformers are imported lazily inside the GPU class).
    """
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
    """Thin web endpoint: validate, segment off the GPU, then forward into the
    batched scorer.

    The endpoint is a lightweight async forwarder; ``@modal.concurrent`` lets a
    single container hold many simultaneous requests open so their ``score``
    calls land in the same dynamic batch instead of each spinning a container.

    Raw-string documents are segmented *here*, on the CPU endpoint, instead of
    inside the batched scorer (which would block the L4 on pure-Python work).
    Critically, segmentation runs in a **process pool**, not ``asyncio.to_thread``:
    syntok is GIL-bound, so under a real burst (a search fan-out fires ~15 docs
    at once) thread-offloaded segmentation still serialized on one interpreter
    and dominated latency (measured: a 15-doc burst was ~31 s wall, but only
    ~12 s with pre-segmented input — ~20 s was serial syntok). A
    ``ProcessPoolExecutor`` segments several docs truly in parallel across cores,
    collapsing that tail. The scorer already accepts a pre-split unit list, so
    the request/response JSON is unchanged and existing callers work unmodified.
    """
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
        # Pre-split units: score them as given, skipping segmentation.
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


@app.local_entrypoint()
def smoke_test():
    """`modal run modal_app.py` — round-trip a tiny request through the scorer."""
    import json
    from dataclasses import asdict

    from snippets_runtime.inference import DocRequest

    out = Compressor().score.remote(
        DocRequest(
            query="what is the capital of france",
            document=[
                "Paris is the capital of France.",
                "The Eiffel Tower was completed in 1889.",
                "Bananas are a good source of potassium.",
            ],
        )
    )
    print(json.dumps(asdict(out), indent=2))

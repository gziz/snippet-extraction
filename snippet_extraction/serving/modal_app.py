"""Serverless compression endpoint on Modal.

A generic ``query + document -> snippet`` API. The endpoint owns the full
pipeline so callers need nothing but an HTTP client:

  segment  -> the block-aware segmenter from snippets_common (syntok +
              markdown-it), i.e. the exact unit definition the training labels
              were built with. Applied when ``document`` is a raw string; pass
              a list to skip segmentation and score pre-split units.
  score    -> windowed ModernBERT inference. Requests are dynamically batched
              (``@modal.batched``): every window of every document in a batch is
              flattened into shared GPU minibatches (``compress_long_batch``).
  select   -> dual-threshold keeping (training-time rule) or, when
              ``budget_tokens`` is set, score-ranked selection under a
              per-document cl100k token budget.
  render   -> kept units joined in document order, non-adjacent runs marked
              with a ``[...]`` gap.

The trained checkpoint lives in a Modal Volume (uploaded after training):

    modal volume create compressor-checkpoints
    modal volume put compressor-checkpoints ./checkpoints/<run> /<run>

Then deploy (CHECKPOINT_NAME below must match the uploaded dir name):

    modal deploy modal_app.py

POST the endpoint URL printed on deploy:

    {
      "query": "...",
      "document": "raw markdown" | ["unit 0", "unit 1", ...],
      "token_threshold": 0.5,       # optional
      "sentence_threshold": 0.5,    # optional
      "budget_tokens": 400,         # optional; score-ranked selection under cap
      "min_score": 0.1              # optional; budget-mode floor
    }

Long documents are handled transparently: `compress_long` packs sentence
units into encoder-sized windows, so a single endpoint covers both cases.
"""

from __future__ import annotations

import sys
from pathlib import Path

import modal

# Let add_local_python_source() discover the sibling packages at deploy time
# even when they aren't installed in the local venv (their deps only need to
# exist inside the container image).
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "runtime"))
sys.path.insert(0, str(_REPO / "common"))

APP_NAME = "query-aware-compressor-l4"
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


# Burst behaviour, not single-doc cost, drives the tail latency. A real search
# fan-out fires ~15 docs at once (5 extracts x 3 parallel searches).
#
# Levers below, all keeping min_containers=0 -> $0 idle (scale-to-zero):
#
#  1. GPU = L40S. The forward is GPU-compute-bound on long docs (a 4-window
#     extract is ~16 windows of 8192 tokens). The forward is already optimal
#     (sdpa+bf16; eager is 8x slower, torch.compile/flash-attn add nothing), so
#     the only way to cut per-window time is the GPU itself. Measured per 8192-
#     token window: L4 0.29 s vs L40S 0.070 s (~4.2x), which also lands a mixed
#     ~15-doc burst at ~4 s wall instead of ~10 s on L4. L40S is even slightly
#     cheaper PER REQUEST than L4 here (it finishes fast enough to offset the
#     higher hourly rate); the higher hourly rate is the cost-control knob to
#     watch, not $/request. (APP_NAME still says "l4" for URL stability.)
#
#  2. CPU memory snapshot (enable_memory_snapshot + split @enter). The cold path
#     is dominated by container boot: importing torch/transformers, building the
#     ModernBERT graph, and loading the checkpoint. Loading the weights ON CPU
#     inside @enter(snap=True) lets Modal snapshot that whole post-import state;
#     a subsequent cold start RESTORES the snapshot and only runs @enter(snap=
#     False) to move the weights to CUDA. Measured end-to-end, scale-from-zero:
#     ~30.6 s -> ~20.4 s. (The residual ~20 s is infra — GPU provisioning +
#     snapshot restore + the CPU endpoint hop — and is GPU-type-independent.)
#     GPU snapshotting (experimental enable_gpu_snapshot) added nothing on top.
#
#  3. max_containers=3 caps the fan-out (costs nothing idle). max_batch_size=6
#     bounds the work any one batch commits to so a burst spreads across the
#     available containers instead of piling onto one.
#
# Burst-from-cold floor: the ~20 s cold start is longer than a ~4 s burst, so
# newly-autoscaled containers don't arrive in time to help THAT burst — it runs
# on the already-warm container(s). To cut the cold-burst tail further you'd add
# a small warm buffer (buffer_containers=1-2): warm headroom kept only while the
# app is active (drops to 0 when fully idle, so still $0 at rest). Left off by
# default; turn it on if cold-burst latency matters more than active-period cost.
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
        import tiktoken
        from snippets_runtime.inference import load_for_inference

        ckpt_dir = f"{VOLUME_MOUNT}/{CHECKPOINT_NAME}"
        self.model, self.tokenizer = load_for_inference(ckpt_dir, base=BASE_MODEL, device="cpu")
        # cl100k matches the budget metric the eval reports (context_tokens).
        self._enc = tiktoken.get_encoding("cl100k_base")

    # Runs on every (restored) cold start, once a GPU is attached.
    @modal.enter(snap=False)
    def to_gpu(self):
        self.model.to("cuda")

    def _count_tokens(self, text: str) -> int:
        return len(self._enc.encode(text, disallowed_special=()))

    @modal.batched(max_batch_size=6, wait_ms=50)
    def score(self, reqs: list[dict]) -> list[dict]:
        """Dynamically-batched scorer.

        Modal accumulates per-document requests — the 5-way fan-out of a single
        search and any concurrent searches — into one call here. We segment each
        doc, flatten *every window of every doc* into shared GPU minibatches via
        ``compress_long_batch`` (one padded forward pass instead of one window at
        a time), then select + render per document. No lock is needed: a batched
        method runs one batch at a time per container, so the HF tokenizer is
        never touched concurrently.
        """
        from snippets_common.segment import segment
        from snippets_common.windowing import render_snippet, select_under_budget
        from snippets_runtime.inference import DocRequest, compress_long_batch

        # Raw string -> training-faithful units; a list is taken as pre-split.
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

        results = compress_long_batch(self.model, self.tokenizer, doc_reqs, device="cuda")

        out: list[dict] = []
        for r, units, res in zip(reqs, docs_units, results):
            if not units:
                out.append(
                    {
                        "kept_indices": [],
                        "kept_sentences": [],
                        "snippet": "",
                        "units": [],
                        "sentence_scores": [],
                        "sentence_mean_probs": [],
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
            snippet = render_snippet(units, kept)
            total_chars = sum(len(u) for u in units) or 1
            out.append(
                {
                    "kept_indices": kept,
                    "kept_sentences": kept_sentences,
                    "snippet": snippet,
                    "units": units,
                    "sentence_scores": res.sentence_scores,
                    "sentence_mean_probs": res.sentence_mean_probs,
                    "compression_ratio": sum(len(s) for s in kept_sentences) / total_chars,
                }
            )
        return out


def _segment_doc(document: str) -> list[str]:
    """Top-level (picklable) worker run inside the segmentation process pool."""
    from snippets_common.segment import segment

    return segment(document)


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

    if not req.get("query") or not req.get("document"):
        raise HTTPException(400, "both 'query' and 'document' are required")

    document = req["document"]
    if isinstance(document, str):
        loop = asyncio.get_running_loop()
        units = await loop.run_in_executor(_seg_pool(), _segment_doc, document)
        req = {**req, "document": units}
    return await Compressor().score.remote.aio(req)


@app.local_entrypoint()
def smoke_test():
    """`modal run modal_app.py` — round-trip a tiny request through the batched scorer."""
    import json

    out = Compressor().score.remote(
        {
            "query": "what is the capital of france",
            "document": [
                "Paris is the capital of France.",
                "The Eiffel Tower was completed in 1889.",
                "Bananas are a good source of potassium.",
            ],
        }
    )
    print(json.dumps(out, indent=2))
